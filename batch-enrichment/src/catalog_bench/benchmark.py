"""The pipeline benchmark: Parquet in, enriched Parquet out, instrumented throughout.

A deliberately plain Python pipeline — template, tokenize, submit, drain, validate — with
no threads, no async and no cleverness beyond the engine seam. The plainness is the
experiment: the question is whether a straightforward pipeline can keep a GPU saturated,
and a heavily optimized one would answer a question nobody asked.

Four numbers come out of it, and each exists for a reason:

* **products/sec and tokens/sec** — throughput, as a fraction of what the bare engine can
  do (`catalog-bench sol` measures the denominator).
* **Per-stage host CPU**, attributed to template / tokenize / submit / drain / validate
  separately. "Python used 40% of a core" is not actionable; "tokenization is 60% of host
  CPU and the feeder stalls behind it" tells you what to fix.
* **Feeder starvation** — how often the loop had room to submit but nothing ready. Near
  zero means the host is keeping up and the pipeline is not the bottleneck. This is the
  single most diagnostic number here.
* **Goodput** — first-pass schema validity. With guided decoding this is ~100% by
  construction; running with `--grammar off` is what quantifies what the grammar buys.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import statistics
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import psutil
import pyarrow as pa
import pyarrow.parquet as pq
from jsonschema import Draft202012Validator

from .engines import GRAMMARS, GenRequest, build_engine
from .prompts import (
    DEFAULT_SCHEMA,
    SCHEMA_MODES,
    encode,
    load_schema,
    load_tokenizer,
    render_system,
    render_user,
)

# The enrichment output contract. The raw JSON string carries the same information a
# per-schema struct column would, without having to derive one.
OUTPUT_SCHEMA = pa.schema([
    pa.field("product_id", pa.string(), nullable=False),
    pa.field("enrichment_json", pa.string(), nullable=True),
    pa.field("status", pa.string(), nullable=False),      # ok | failed
    pa.field("error", pa.string(), nullable=True),
    pa.field("input_tokens", pa.int32(), nullable=False),
    pa.field("output_tokens", pa.int32(), nullable=False),
    pa.field("worker_id", pa.string(), nullable=False),
])


@dataclass
class StageTimer:
    """Per-stage wall AND CPU time, accumulated across the run.

    Conflating the two is the easy mistake. `drain` spends nearly all its wall time
    blocked on the GPU, which is the pipeline working exactly as intended — charging that
    to host cost would make a perfectly saturated pipeline look host-bound. Wall time says
    where the loop is; CPU time says what the host is burning.

    This is single-threaded, so `time.process_time()` deltas are honest per-stage host
    cost. If the loop ever grows a thread pool, this accounting has to become per-thread
    and the numbers stop being comparable across configurations.
    """

    wall: dict[str, float] = field(default_factory=dict)
    cpu: dict[str, float] = field(default_factory=dict)

    def add(self, stage: str, dwall: float, dcpu: float) -> None:
        self.wall[stage] = self.wall.get(stage, 0.0) + dwall
        self.cpu[stage] = self.cpu.get(stage, 0.0) + dcpu

    def summary(self, wall: float, cpu_total: float) -> dict[str, Any]:
        stages = sorted(self.wall)
        return {
            "wall_seconds": {k: round(self.wall[k], 3) for k in stages},
            "cpu_seconds": {k: round(self.cpu[k], 3) for k in stages},
            "wall_share": {k: round(self.wall[k] / wall, 4) for k in stages},
            "cpu_share_of_host": {
                k: round(self.cpu[k] / cpu_total, 4) if cpu_total else 0.0 for k in stages
            },
            "blocked_on_engine_share": round(
                max(0.0, self.wall.get("drain", 0.0) - self.cpu.get("drain", 0.0)) / wall, 4
            ),
        }


class _Stage:
    """Times one stage on both clocks. Used as a context manager in the hot loop."""

    __slots__ = ("c0", "name", "timer", "w0")

    def __init__(self, timer: StageTimer, name: str) -> None:
        self.timer, self.name = timer, name

    def __enter__(self) -> None:
        self.w0, self.c0 = time.perf_counter(), time.process_time()

    def __exit__(self, *exc: object) -> None:
        self.timer.add(self.name,
                       time.perf_counter() - self.w0,
                       time.process_time() - self.c0)


def write_outputs(results: list[dict[str, Any]], path: Path, model: str) -> None:
    """Write enrichment results so there is something to inspect and score.

    Without this the benchmark measures throughput and then discards everything it
    produced, which makes accuracy unevaluable and gives an integrator nothing to look at.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "product_id": [r["product_id"] for r in results],
        "enrichment_json": [r.get("enrichment_json") for r in results],
        "status": ["ok" if r["valid"] else "failed" for r in results],
        "error": [r["error"] for r in results],
        "input_tokens": [r["input_tokens"] for r in results],
        "output_tokens": [r["output_tokens"] for r in results],
        "worker_id": [model] * len(results),
    }, schema=OUTPUT_SCHEMA)
    pq.write_table(table, path, row_group_size=2048, compression="snappy")


def host_description() -> dict[str, Any]:
    """Who ran this, on what. A throughput number without a host is not reproducible."""
    cpu = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        # A missing CPU name must not fail a run.
        with contextlib.suppress(Exception):
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, check=True,
                                 timeout=5).stdout.strip() or cpu
    return {
        "host_cpu": cpu,
        "host_cores_logical": psutil.cpu_count(logical=True),
        "host_cores_physical": psutil.cpu_count(logical=False),
        "host_ram_bytes": psutil.virtual_memory().total,
        "host_platform": platform.platform(),
    }


def _engine_kwargs(args: argparse.Namespace, schema: dict[str, Any]) -> dict[str, Any]:
    """Assemble engine kwargs, gated by engine.

    The gating is load-bearing, not tidiness. Surplus kwargs on the trtllm path are
    forwarded to `LLM()`, whose args model is strict (`extra="forbid"`), so an unknown key
    is a hard ValidationError at startup. That is the failure mode we want — a
    mis-spelled memory limit that got silently dropped would surface as an OOM twenty
    minutes into a model load — but it means mock-only and http-only knobs must not leak
    across.
    """
    kwargs: dict[str, Any] = {
        "model": args.model,
        "grammar": args.grammar,
        "schema": schema,
        "max_tokens": args.max_tokens,
        "max_in_flight_tokens": args.max_in_flight_tokens,
        "enable_thinking": False,
    }

    if args.engine == "mock":
        kwargs["sim_tokens_per_sec"] = args.sim_tokens_per_sec
        kwargs["invalid_rate"] = 0.0 if args.grammar != "off" else args.mock_invalid_rate

    elif args.engine == "trtllm":
        for kv in args.trtllm_kwarg:
            if "=" not in kv:
                raise SystemExit(f"--trtllm-kwarg expects K=V, got {kv!r}")
            k, _, v = kv.partition("=")
            try:
                kwargs[k] = json.loads(v)
            except json.JSONDecodeError:
                kwargs[k] = v

    elif args.engine == "http":
        kwargs["base_url"] = args.http_base_url
        kwargs["prefix_reuse"] = args.http_prefix_reuse
        # The thread pool must exceed in-flight so it never becomes the concurrency cap.
        # Undersizing it here silently runs the whole sweep at the pool size.
        kwargs["max_workers"] = args.in_flight + 8

    return kwargs


def _steady_state(results: list[dict[str, Any]], warmup_end: float | None,
                  last_submit: float | None, n_at_warmup_end: int,
                  n_at_last_submit: int) -> dict[str, Any]:
    """The measured window: end of warmup to last submit.

    Timing has three phases and only the middle one is a throughput measurement.

      warmup  the first `--warmup` products. Grammar compilation, CUDA graph capture and
              kernel selection all happen here and none of them recur. Discarded.
      steady  from the end of warmup until the last product is *submitted*. Concurrency
              is at target throughout. This is the number.
      drain   after the last submit, in-flight depth falls to zero. Real work, but at
              declining concurrency, so it is not a rate.

    Timing the whole run instead understates throughput, and understates it worst at high
    in-flight — with 1024 in flight over 2000 products the drain tail is half the run, so
    the sweep would appear to saturate exactly where peak throughput is expected. The bare
    engine baseline warms up too, and the efficiency ratio divides one by the other, so
    both sides have to be measured the same way.
    """
    if warmup_end is None or last_submit is None:
        return {"valid": False, "reason": "run too short for a steady-state window"}
    window = last_submit - warmup_end
    n_window = n_at_last_submit - n_at_warmup_end
    if window <= 0 or n_window <= 0:
        return {"valid": False, "reason": "run too short for a steady-state window"}

    measured = results[n_at_warmup_end:n_at_last_submit]
    w_in = sum(r["input_tokens"] for r in measured)
    w_out = sum(r["output_tokens"] for r in measured)
    return {
        "valid": True,
        "seconds": round(window, 3),
        "n_products": n_window,
        "products_per_sec": round(n_window / window, 2),
        "output_tokens_per_sec": round(w_out / window, 1),
        "total_tokens_per_sec": round((w_in + w_out) / window, 1),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    schema = load_schema(args.schema)
    validator = Draft202012Validator(schema)

    rows = pq.read_table(args.catalog).to_pylist()
    if args.n:
        rows = rows[: args.n]

    engine = build_engine(args.engine, **_engine_kwargs(args, schema))

    # Use the engine's own tokenizer where it has one, so the ids we count are exactly the
    # ids the model sees. `is None` rather than `or`: truthiness on a Hugging Face
    # tokenizer calls __len__, which the base class does not implement, so the `or` form
    # raises on a perfectly valid tokenizer after a full model load.
    tok = getattr(engine, "tokenizer", None)
    if tok is None:
        tok = load_tokenizer(args.tokenizer)
    if args.engine == "http":
        engine.tokenizer = tok

    timer = StageTimer()
    system = render_system(schema, args.schema_mode)

    proc = psutil.Process()
    peak_rss = proc.memory_info().rss

    results: list[dict[str, Any]] = []
    in_flight_samples: list[int] = []
    starved_polls = total_polls = 0
    next_idx = req_id = 0
    n_total = len(rows)

    warmup_end: float | None = None
    last_submit: float | None = None
    n_at_warmup_end = n_at_last_submit = 0

    cpu_t0 = time.process_time()
    t0 = time.perf_counter()

    while len(results) < n_total:
        # ---- feed ----------------------------------------------------------------
        submitted_this_round = 0
        while next_idx < n_total and engine.pending < args.in_flight:
            row = rows[next_idx]

            with _Stage(timer, "template"):
                user = render_user(row)

            with _Stage(timer, "tokenize"):
                if hasattr(engine, "apply_chat_template"):
                    ids = encode(tok, engine.apply_chat_template(system, user))
                else:
                    ids = encode(tok, system) + encode(tok, user)

            with _Stage(timer, "submit"):
                engine.submit(GenRequest(
                    req_id, row["product_id"], ids, args.max_tokens,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}]
                    if args.engine == "http" else None,
                ))

            req_id += 1
            next_idx += 1
            submitted_this_round += 1

            if next_idx == n_total and last_submit is None:
                last_submit = time.perf_counter()
                n_at_last_submit = len(results)

        # ---- drain ---------------------------------------------------------------
        total_polls += 1
        if next_idx < n_total and engine.pending < args.in_flight and not submitted_this_round:
            starved_polls += 1

        in_flight_samples.append(engine.pending)
        peak_rss = max(peak_rss, proc.memory_info().rss)

        with _Stage(timer, "drain"):
            drained = engine.drain(max_wait=args.drain_wait)

        # ---- validate ------------------------------------------------------------
        with _Stage(timer, "validate"):
            for r in drained:
                valid, err = False, r.error
                if not err:
                    try:
                        validator.validate(json.loads(r.text))
                        valid = True
                    except Exception as exc:
                        err = type(exc).__name__
                results.append({
                    "product_id": r.product_id,
                    "valid": valid,
                    "error": err,
                    "input_tokens": r.prompt_tokens,
                    "output_tokens": r.output_tokens,
                    "finish_reason": r.finish_reason,
                    "enrichment_json": r.text,
                })

            if warmup_end is None and len(results) >= args.warmup:
                warmup_end = time.perf_counter()
                n_at_warmup_end = len(results)

    wall = time.perf_counter() - t0
    cpu_time = time.process_time() - cpu_t0
    peak_rss = max(peak_rss, proc.memory_info().rss)
    # Read before shutdown: afterwards the engine is entitled to have dropped whatever it
    # would describe.
    engine_info = engine.describe() if hasattr(engine, "describe") else {}
    engine.shutdown()

    if args.save_outputs:
        write_outputs(results, args.save_outputs, args.model)

    n_valid = sum(1 for r in results if r["valid"])
    in_tok = sum(r["input_tokens"] for r in results)
    out_tok = sum(r["output_tokens"] for r in results)

    steady = _steady_state(results, warmup_end, last_submit,
                           n_at_warmup_end, n_at_last_submit)

    # Headline rates are the steady-state ones where a window exists. The bare-engine
    # baseline is warmed up too, so both sides of the efficiency ratio measure the same
    # thing or the ratio is meaningless.
    if steady["valid"]:
        rate_pps = steady["products_per_sec"]
        rate_tps = steady["total_tokens_per_sec"]
        rate_otps = steady["output_tokens_per_sec"]
    else:
        rate_pps = round(len(results) / wall, 2)
        rate_tps = round((in_tok + out_tok) / wall, 1)
        rate_otps = round(out_tok / wall, 1)

    osl_mean = round(out_tok / max(1, len(results)), 1)

    record = {
        "kind": "probe",
        "engine": args.engine,
        "device": {"trtllm": "gpu", "mock": "none", "http": "gpu"}[args.engine],
        "mock": bool(getattr(engine, "is_mock", False)),
        "model": args.model,
        "grammar": args.grammar,
        "schema_mode": args.schema_mode,
        "in_flight": args.in_flight,
        "max_tokens": args.max_tokens,
        "n_products": len(results),
        "warmup": args.warmup,

        "products_per_sec": rate_pps,
        "output_tokens_per_sec": rate_otps,
        "total_tokens_per_sec": rate_tps,
        "rate_basis": "steady_state" if steady["valid"] else "end_to_end (NO WARMUP WINDOW)",
        "steady_state": steady,

        # What the job actually took, warmup and drain included. The honest number for a
        # wall-clock SLA question, as opposed to a rate.
        "wall_seconds": round(wall, 3),
        "end_to_end_products_per_sec": round(len(results) / wall, 2),
        "end_to_end_total_tokens_per_sec": round((in_tok + out_tok) / wall, 1),

        "isl_mean": round(in_tok / max(1, len(results)), 1),
        "osl_mean": osl_mean,
        "tokens_per_product": round((in_tok + out_tok) / max(1, len(results)), 1),

        "goodput": round(n_valid / max(1, len(results)), 5),
        "n_invalid": len(results) - n_valid,

        "host_cpu_seconds": round(cpu_time, 3),
        "host_cpu_share_of_wall": round(cpu_time / wall, 4),
        "stages": timer.summary(wall, cpu_time),
        "feeder_starved_polls": starved_polls,
        "feeder_total_polls": total_polls,
        "feeder_starvation_rate": round(starved_polls / max(1, total_polls), 4),
        "in_flight_mean": round(statistics.fmean(in_flight_samples), 1)
        if in_flight_samples else 0,

        "peak_rss_bytes": peak_rss,
        **host_description(),
        **engine_info,
    }

    # A mean output length sitting exactly on the cap is not a result, it is a prompt
    # contract failure: the model reasoned or rambled until it ran out of budget on every
    # single product. Flagged in the record so it cannot be read as a measurement.
    if len(results) > 1 and osl_mean >= args.max_tokens - 0.5:
        record["osl_saturated"] = True
        print(f"!! osl_mean ({osl_mean}) is at --max-tokens ({args.max_tokens}) across "
              "the whole run. Every output hit the cap, which means they were truncated "
              "mid-JSON. Check that reasoning is disabled and that max_tokens exceeds "
              "the schema's worst-case output length.")

    return record


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--catalog", type=Path, default=Path("data/catalog"))
    ap.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    ap.add_argument("--engine", default="mock", choices=["mock", "trtllm", "http"])
    ap.add_argument("--model", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
                    help="TRT-LLM engine dir or HF id (ignored by the mock engine)")
    ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-1.5B-Instruct",
                    help="only used when the engine has no tokenizer of its own")
    ap.add_argument("--grammar", default="xgrammar", choices=list(GRAMMARS))
    ap.add_argument("--schema-mode", default="compact", choices=list(SCHEMA_MODES))
    ap.add_argument("--n", type=int, default=1000, help="products to process; 0 = all")
    ap.add_argument("--in-flight", type=int, default=256, help="max concurrent requests")
    ap.add_argument("--max-in-flight-tokens", type=int, default=1 << 20)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=100,
                    help="products excluded from the rate; covers grammar compilation, "
                         "CUDA graph capture and kernel selection")
    ap.add_argument("--drain-wait", type=float, default=0.005)
    ap.add_argument("--save-outputs", type=Path, default=None,
                    help="write enrichment results as Parquet, for `catalog-bench accuracy`")
    ap.add_argument("--out", type=Path, default=None)

    mock = ap.add_argument_group("mock engine")
    mock.add_argument("--sim-tokens-per-sec", type=float, default=20_000.0,
                      help="simulated aggregate decode rate")
    mock.add_argument("--mock-invalid-rate", type=float, default=0.07,
                      help="with --grammar off only: simulated schema-invalid share")

    trt = ap.add_argument_group("trtllm engine")
    trt.add_argument("--trtllm-kwarg", action="append", default=[], metavar="K=V",
                     help="extra kwarg forwarded verbatim to LLM(); repeatable, value "
                          "parsed as JSON with a string fallback. Needed because LLM() "
                          "otherwise sizes its executor from model defaults, which on a "
                          "Mamba hybrid asks for 92 GiB of KV cache on an 80 GiB card. "
                          "Executor sizing describes the host, not the measurement, so it "
                          "belongs on the command line next to --in-flight")

    http = ap.add_argument_group("http engine")
    http.add_argument("--http-base-url", default="http://localhost:8000",
                      help="base URL of the OpenAI-compatible server")
    http.add_argument("--http-prefix-reuse", action="store_true",
                      help="record that prefix caching is enabled server-side; the "
                           "server enables it, not this flag")


def main(args: argparse.Namespace) -> int:
    record = run(args)

    out = args.out or Path(
        f"reports/probe_{args.engine}_{args.grammar}_{args.schema_mode}"
        f"_if{args.in_flight}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2))

    if record["mock"]:
        print("!! engine=mock — harness check only. These are NOT measurements.")
    print(f"{record['n_products']} products in {record['wall_seconds']}s end-to-end")
    print(f"  rate basis         {record['rate_basis']}"
          + (f" ({record['steady_state']['n_products']} products over "
             f"{record['steady_state']['seconds']}s)" if record["steady_state"]["valid"] else ""))
    print(f"  products/s         {record['products_per_sec']}  "
          f"(end-to-end {record['end_to_end_products_per_sec']})")
    print(f"  tok/s              {record['total_tokens_per_sec']}  "
          f"(end-to-end {record['end_to_end_total_tokens_per_sec']})")
    print(f"  goodput            {record['goodput']:.4%}  ({record['n_invalid']} invalid)")
    print(f"  tokens/product     {record['tokens_per_product']}  "
          f"(ISL {record['isl_mean']}, OSL {record['osl_mean']})")
    print(f"  host CPU           {record['host_cpu_share_of_wall']:.1%} of wall "
          f"({record['host_cpu_seconds']}s)")
    print(f"  feeder starvation  {record['feeder_starvation_rate']:.1%} of polls")
    print(f"  blocked on engine  {record['stages']['blocked_on_engine_share']:.1%} of wall")
    print(f"    {'stage':<10} {'wall':>8} {'host CPU':>10}")
    for stage in record["stages"]["wall_share"]:
        print(f"    {stage:<10} {record['stages']['wall_share'][stage]:>7.2%} "
              f"{record['stages']['cpu_share_of_host'][stage]:>9.2%}")
    print(f"wrote {out}")
    return 0
