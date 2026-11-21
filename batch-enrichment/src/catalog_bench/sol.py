"""Bare-engine baseline: a thin wrapper around `trtllm-bench throughput`.

The ceiling the pipeline is measured against. *Standard* is the operative word — a
bespoke baseline would let us publish a flattering denominator, and the credibility of
any "X% of the engine" claim rests on the denominator being one you can reproduce
independently. So this runs the real tool and does nothing but normalize its output.

GPU-only. Off-GPU it fails loudly rather than estimating: a fabricated denominator would
silently invalidate every efficiency number downstream.

Two caveats, both recorded in the output record so they reach the report:

1. `trtllm-bench` measures the engine WITHOUT guided decoding, so a grammar-on run has no
   like-for-like bare-engine baseline. The honest denominator for a guided run is the
   pipeline's own `--grammar off` configuration; this number bounds what the *pipeline*
   costs, not what the grammar costs.

2. The dataset is generated from our real prompts, so input length matches exactly.
   Output length is pinned per request, whereas real guided decoding emits a variable
   number. Uniform decode lengths batch more neatly, so this baseline is slightly
   optimistic for the engine — which biases efficiency down, the safe direction for a
   claim, but worth saying rather than letting a reader assume otherwise.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def find_key(obj: Any, key: str) -> Any:
    """Recursively locate a key in trtllm-bench's report JSON.

    Structure-agnostic on purpose: the report's nesting is not a stable contract, and the
    field names have been stable far longer than their location.
    """
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = find_key(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_key(v, key)
            if found is not None:
                return found
    return None


def parse_report(raw: dict[str, Any]) -> tuple[float, float, float, int]:
    """Pull (latency_ns, output_tokens, input_tokens, n_requests) across report schemas.

    Version risk lives in serialized formats as much as in function signatures, and this
    is what it looks like. rc9 renamed every key this needs except `num_requests`:

        pre-rc9                 rc9
        total_latency_ns    ->  performance.total_latency_ms
        total_output_tokens ->  request_info.avg_output_length   (per request, not total)
        total_input_tokens  ->  request_info.avg_input_length    (per request, not total)

    The last two are not renames but a change of *meaning* — totals became per-request
    means. A parser matching on names alone would silently produce numbers thousands of
    times too small. Old names are tried first so this keeps working if the pin moves back.
    """
    n_requests = find_key(raw, "num_requests")
    latency_ns = find_key(raw, "total_latency_ns")
    out_tokens = find_key(raw, "total_output_tokens")
    in_tokens = find_key(raw, "total_input_tokens")

    if not latency_ns:
        latency_ms = find_key(raw, "total_latency_ms")
        if latency_ms:
            latency_ns = latency_ms * 1e6
        # Multiply the per-request means back up so everything downstream keeps its
        # "total" meaning. With a fixed output length the mean is exact, not an estimate.
        avg_out = find_key(raw, "avg_output_length")
        avg_in = find_key(raw, "avg_input_length")
        if out_tokens is None and avg_out is not None and n_requests:
            out_tokens = avg_out * n_requests
        if in_tokens is None and avg_in is not None and n_requests:
            in_tokens = avg_in * n_requests

    if not latency_ns:
        raise SystemExit(
            "could not find total_latency_ns or total_latency_ms in the trtllm-bench "
            "report; its schema may have changed again for this version")

    return latency_ns, out_tokens or 0, in_tokens or 0, n_requests or 0


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--model", required=True, help="HF id or engine dir")
    ap.add_argument("--dataset", type=Path, required=True,
                    help="JSONL from `catalog-bench tokens --bench-dataset`, so the token "
                         "shape matches the pipeline run exactly")
    ap.add_argument("--concurrency", type=int, default=256,
                    help="in-flight requests; mirror the pipeline's --in-flight")
    ap.add_argument("--num-requests", type=int, default=2000)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--backend", default="pytorch")
    ap.add_argument("--bench-arg", action="append", default=[], metavar="FLAG=VALUE",
                    help="extra flag forwarded to `trtllm-bench throughput`, without the "
                         "leading dashes (e.g. max_batch_size=256). Repeatable. Needed for "
                         "the same reason `run` has --trtllm-kwarg: left to its own "
                         "defaults the executor asks for 92 GiB on an 80 GiB card")
    ap.add_argument("--out", type=Path, default=Path("reports/sol_baseline.json"))
    ap.add_argument("--raw-report", type=Path,
                    default=Path("reports/trtllm_bench_raw.json"))


def main(args: argparse.Namespace) -> int:
    if shutil.which("trtllm-bench") is None:
        raise SystemExit(
            "trtllm-bench not found. This is the bare-engine baseline and it needs a GPU\n"
            "node with TensorRT-LLM installed. Refusing to estimate: a fabricated\n"
            "denominator would invalidate every efficiency number in the report.")
    if not args.dataset.exists():
        raise SystemExit(f"dataset not found: {args.dataset}\nGenerate it with: "
                         f"catalog-bench tokens --bench-dataset {args.dataset}")

    args.raw_report.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "trtllm-bench", "--model", args.model, "throughput",
        "--dataset", str(args.dataset),
        "--backend", args.backend,
        "--concurrency", str(args.concurrency),
        "--num_requests", str(args.num_requests),
        "--warmup", str(args.warmup),
        "--report_json", str(args.raw_report),
    ]
    for ba in args.bench_arg:
        if "=" not in ba:
            raise SystemExit(f"--bench-arg expects FLAG=VALUE, got {ba!r}")
        flag, _, value = ba.partition("=")
        cmd += [f"--{flag.lstrip('-')}", value]

    print("$ " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise SystemExit(f"trtllm-bench exited {proc.returncode}; see output above")

    latency_ns, out_tokens, in_tokens, n_requests = parse_report(
        json.loads(args.raw_report.read_text()))
    seconds = latency_ns / 1e9

    record = {
        "kind": "sol_baseline",
        "engine": "trtllm-bench",
        "mock": False,
        "model": args.model,
        # Unguided by construction; the key exists so the report can join this against
        # pipeline rows without special-casing.
        "grammar": "off",
        "concurrency": args.concurrency,
        "n_products": n_requests,
        "wall_seconds": round(seconds, 3),
        "products_per_sec": round(n_requests / seconds, 2),
        "output_tokens_per_sec": round(out_tokens / seconds, 1),
        "total_tokens_per_sec": round((out_tokens + in_tokens) / seconds, 1),
        "isl_mean": round(in_tokens / max(1, n_requests), 1),
        "osl_mean": round(out_tokens / max(1, n_requests), 1),
        "caveats": [
            "Measures the engine WITHOUT guided decoding; not a like-for-like denominator "
            "for grammar-on runs.",
            "Dataset output length is fixed per request while real guided decoding varies. "
            "Uniform lengths batch more neatly, so this is slightly optimistic for the "
            "engine and biases efficiency downward.",
        ],
        "raw_report": str(args.raw_report),
        "command": cmd,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))
    print(f"bare engine: {record['total_tokens_per_sec']:,.0f} tok/s total, "
          f"{record['output_tokens_per_sec']:,.0f} tok/s output, "
          f"{record['products_per_sec']} req/s")
    print(f"wrote {args.out}")
    return 0
