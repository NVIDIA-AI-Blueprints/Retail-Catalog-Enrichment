"""Token accounting: the one measurement you can take without a GPU.

Tokens per product is the denominator of every headline figure — products/sec,
products/GPU-hour, cost per million, and the GPU-count sizing formula all divide by it.
Tokenization is pure CPU, so this can be settled before the GPU session rather than
during it.

Two outputs:

1. **Input length distribution**, split into the invariant system prefix and the
   per-product suffix. That split tells you whether prefix caching is worth anything for
   your `schema_mode`, and it is the input to the `schema_mode` decision.

2. **A `trtllm-bench` dataset** built from these exact prompts. This is the
   methodological point: pipeline efficiency is `pipeline_tok_s / bare_engine_tok_s`, and
   that ratio is meaningless unless both sides ran the same token shape. Generating the
   baseline's dataset from the pipeline's own prompts makes them identical by
   construction rather than by assumption.

Output length is *estimated* here, not measured — only a real decode can say how many
tokens the model actually emits. The estimate is derived from the schema's bounded
structure and is labelled as an assumption everywhere it appears.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .prompts import (
    DEFAULT_SCHEMA,
    SCHEMA_MODES,
    encode,
    load_schema,
    load_tokenizer,
    render_system,
    render_user,
)

# Stand-in values for output-length estimation. Free-text fields have no length bound in
# the schema by design, so their cost has to be modelled with representative content
# rather than derived.
_SHORT_TEXT = {0.0: "navy", 0.66: "full-grain leather",
               1.0: "certified full-grain vegetable-tanned leather"}
_LONG_TEXT = {
    0.0: "A merino base layer.",
    0.66: ("A lightweight merino base layer built for cold-weather layering, with "
           "flatlock seams and a trim fit."),
    1.0: ("A lightweight merino wool base layer built for cold-weather layering, with "
          "flatlock seams, a trim athletic fit, and naturally antimicrobial fibers that "
          "keep odor down across multiple wears between washes."),
}
_PROSE_HINTS = ("description", "summary", "copy", "instructions", "text")

# Assumed cap for an array with no maxItems, so the estimate stays finite. An unbounded
# array genuinely has no worst case; `check-schema` reports that separately as an error.
_UNBOUNDED_ARRAY_ASSUMPTION = 10


def _percentile(xs: list[int], q: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    return s[min(len(s) - 1, int(len(s) * q))]


def _instance(schema: dict[str, Any], fill: float) -> dict[str, Any]:
    """Build a schema-conforming instance at a given 'fullness'.

    fill=0 is the cheapest legal output (empty arrays, nulls, terse values); fill=1 is the
    most expensive (every array at maxItems, verbose free text). The gap between them is
    the output-length range the in-flight batcher has to absorb.
    """
    out: dict[str, Any] = {}
    for name, spec in schema.get("properties", {}).items():
        types = spec.get("type")
        types = types if isinstance(types, list) else [types]
        if "enum" in spec:
            out[name] = spec["enum"][0]
        elif "array" in types:
            cap = spec.get("maxItems", _UNBOUNDED_ARRAY_ASSUMPTION)
            out[name] = ["moisture wicking fabric"] * round(cap * fill)
        elif "boolean" in types:
            out[name] = False
        elif "integer" in types or "number" in types:
            out[name] = 1
        elif "null" in types and fill == 0.0:
            out[name] = None
        elif any(h in name.lower() for h in _PROSE_HINTS):
            out[name] = _LONG_TEXT[fill]
        else:
            out[name] = _SHORT_TEXT[fill]
    return out


def osl_estimates(schema: dict[str, Any], tokenizer: Any) -> dict[str, int]:
    """Min / typical / max output tokens, plus the part the grammar actually fixes.

    Guided decoding constrains *structure*, not *length*: keys, punctuation and enum
    members are pinned, but free-text values and any array with maxItems > minItems are
    the model's choice. `fixed` quantifies how much of the output is genuinely
    deterministic — the rest is variance the batcher has to absorb, and it is usually the
    larger share.
    """
    skeleton = {n: ([] if "array" in str(s.get("type")) else "")
                for n, s in schema.get("properties", {}).items()}

    def n_tokens(obj: Any) -> int:
        return len(encode(tokenizer, json.dumps(obj)))

    return {
        "min": n_tokens(_instance(schema, 0.0)),
        "typical": n_tokens(_instance(schema, 0.66)),
        "max": n_tokens(_instance(schema, 1.0)),
        "fixed": n_tokens(skeleton),
    }


def estimate_osl(schema: dict[str, Any], tokenizer: Any) -> int:
    """Typical-case output length — the single number the report and dataset use."""
    return osl_estimates(schema, tokenizer)["typical"]


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--catalog", type=Path, default=Path("data/catalog"))
    ap.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    ap.add_argument("--tokenizer", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
                    help="HF id or path to tokenizer.json. Any tokenizer gives a usable "
                         "shape estimate; the model's own is the one that counts")
    ap.add_argument("--schema-mode", default="compact", choices=list(SCHEMA_MODES))
    ap.add_argument("--limit", type=int, default=0, help="0 = whole catalog")
    ap.add_argument("--bench-dataset", type=Path, default=None,
                    help="write a trtllm-bench dataset JSONL here")
    ap.add_argument("--out", type=Path, default=Path("reports/tokens.json"))


def main(args: argparse.Namespace) -> int:
    schema = load_schema(args.schema)
    tok = load_tokenizer(args.tokenizer)

    rows = pq.read_table(args.catalog).to_pylist()
    if args.limit:
        rows = rows[: args.limit]

    system = render_system(schema, args.schema_mode)
    system_ids = encode(tok, system)
    n_system = len(system_ids)
    osl_est = estimate_osl(schema, tok)

    user_lens: list[int] = []
    total_lens: list[int] = []
    bench_records: list[dict[str, Any]] = []

    for i, row in enumerate(rows):
        user_ids = encode(tok, render_user(row))
        user_lens.append(len(user_ids))
        total_lens.append(n_system + len(user_ids))
        if args.bench_dataset:
            # The chat template's own control tokens are not included. They are a small
            # fixed addition applied identically to both sides of the ratio; recorded as a
            # known small bias rather than silently absorbed.
            bench_records.append({
                "task_id": i,
                "input_ids": system_ids + user_ids,
                "output_tokens": osl_est,
            })

    mean_isl = statistics.fmean(total_lens)
    summary = {
        "kind": "tokens",
        "catalog": str(args.catalog),
        "n_products": len(rows),
        "tokenizer": args.tokenizer,
        "schema_mode": args.schema_mode,
        "system_prefix_tokens": n_system,
        "isl_mean": round(mean_isl, 1),
        "isl_p50": _percentile(total_lens, 0.50),
        "isl_p95": _percentile(total_lens, 0.95),
        "isl_max": max(total_lens),
        "user_suffix_mean": round(statistics.fmean(user_lens), 1),
        "prefix_share": round(n_system / mean_isl, 3),
        "osl_estimate": osl_est,
        "osl_source": "estimated from schema bounds; NOT measured",
        "tokens_per_product_mean": round(mean_isl + osl_est, 1),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    if args.bench_dataset:
        args.bench_dataset.parent.mkdir(parents=True, exist_ok=True)
        with args.bench_dataset.open("w") as fh:
            for rec in bench_records:
                fh.write(json.dumps(rec) + "\n")

    print(f"tokenizer: {args.tokenizer}   schema_mode: {args.schema_mode}")
    print(f"  system prefix     {n_system:>6} tok  ({summary['prefix_share']:.0%} of mean ISL)")
    print(f"  user suffix mean  {summary['user_suffix_mean']:>6} tok")
    print(f"  ISL  mean {summary['isl_mean']:.0f}  p50 {summary['isl_p50']}  "
          f"p95 {summary['isl_p95']}  max {summary['isl_max']}")
    print(f"  OSL  ~{osl_est} tok (estimated from schema, not measured)")
    print(f"  tokens/product ~{summary['tokens_per_product_mean']:.0f}")
    print(f"wrote {args.out}" + (f" and {args.bench_dataset}" if args.bench_dataset else ""))
    return 0
