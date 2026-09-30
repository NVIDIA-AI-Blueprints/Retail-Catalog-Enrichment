"""Capacity and cost: catalog size + SLA -> GPU count -> dollars.

    GPUs = catalog_size / (valid_products_per_sec * SLA_seconds * scaling_efficiency)

Arithmetic on two inputs. One of them — tokens per product — is measured on CPU and
already known before you book anything. The other — valid products/sec/GPU — needs the
GPU run, and until you have it this falls back to an explicit, labelled assumption.

The word *valid* is load-bearing. Throughput and goodput are different numbers, and a
fleet sized from raw throughput is a fleet sized to produce output that fails schema
validation.

That labelling is the whole discipline of this file. A sizing model that silently mixes
measured and guessed inputs is how someone ends up buying 25 GPUs for a job that needs 60.
Every number printed is tagged `measured` or `ASSUMED`, and the assumed ones are echoed
back with the flag that would replace them.

    # before the GPU run — assumed rate, measured tokens
    catalog-bench capacity --catalog-size 10_000_000 --sla-hours 4

    # after it — everything measured
    catalog-bench capacity --catalog-size 10_000_000 --sla-hours 4 --from-reports reports/

Scaling is linear by construction — no central queue, no inter-worker coordination — so
GPU count is a division rather than a simulation. `--scaling-efficiency` applies a derate
for the coordination that a real deployment does need.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

# Order-of-magnitude placeholders so the model produces a number out of the box. Public
# list prices vary by region, term and negotiation; pass --gpu-cost-per-hour from your own
# contract before sizing a real purchase.
GPU_HOURLY_USD = {"h100": 3.50, "h200": 4.50, "b200": 6.00, "l40s": 1.20}

# Pre-measurement placeholder for aggregate engine throughput, for a ~30B model at this
# token shape. It exists only so the formula runs before the GPU does; it is the number
# the benchmark replaces.
ASSUMED_TOKENS_PER_SEC = 50_000.0
ASSUMED_TOKENS_PER_PRODUCT = 750.0


def load_measured(reports: Path) -> dict[str, Any]:
    """Pull measured tokens/product and, if a real run exists, measured throughput."""
    found: dict[str, Any] = {}
    for f in sorted(reports.glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue

        if rec.get("kind") == "tokens" and "tokens_per_product" not in found:
            found["tokens_per_product"] = rec["tokens_per_product_mean"]
            found["tokens_source"] = f"measured ({rec['schema_mode']}, {rec['tokenizer']})"

        # Only non-mock records may set a throughput number. The mock's tok/s is a
        # constant someone typed on a laptop, and letting it reach a cost model is exactly
        # the failure the report's verdict refuses elsewhere.
        #
        # Rank by **valid** products/sec, never by raw tok/s. Ranking by tok/s always
        # selects grammar=off, because unguided decoding produces tokens faster and a
        # sizeable share of what it produces is schema-invalid. Sizing a fleet from that
        # rate bills for output that gets thrown away.
        if rec.get("kind") == "probe" and not rec.get("mock"):
            valid_pps = rec["products_per_sec"] * rec.get("goodput", 1.0)
            if valid_pps > found.get("products_per_sec", 0):
                found["products_per_sec"] = valid_pps
                found["tokens_per_sec"] = rec["total_tokens_per_sec"]
                found["goodput"] = rec.get("goodput", 1.0)
                found["rate_source"] = (f"measured (grammar={rec['grammar']}, "
                                        f"goodput {rec.get('goodput', 1.0):.1%})")
    return found


def size(catalog: int, sla_hours: float, products_per_sec: float,
         scaling_efficiency: float) -> dict[str, float]:
    """Sizing takes **valid** products/sec directly.

    Not tok/s divided by tokens/product — that quietly assumes every generated token lands
    in a schema-valid product, which is true of a guided configuration and of nothing else.
    """
    ideal = catalog / (products_per_sec * sla_hours * 3600)
    gpus = math.ceil(ideal / scaling_efficiency)
    # Actual wall time on the integer GPU count, which is what the SLA is judged on.
    hours = catalog / (gpus * products_per_sec * scaling_efficiency * 3600)
    return {
        "products_per_sec_per_gpu": products_per_sec,
        "products_per_gpu_hour": products_per_sec * 3600,
        "gpus": gpus,
        "hours": hours,
        "gpu_hours": gpus * hours,
    }


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--catalog-size", type=lambda s: int(s.replace("_", "")),
                    default=10_000_000)
    ap.add_argument("--sla-hours", type=float, default=4.0)
    ap.add_argument("--tokens-per-product", type=float, default=None,
                    help="override; default is measured from --from-reports")
    ap.add_argument("--tokens-per-sec", type=float, default=None,
                    help="aggregate engine tok/s per GPU; the input the benchmark measures")
    ap.add_argument("--gpu", default="h200", choices=sorted(GPU_HOURLY_USD))
    ap.add_argument("--gpu-cost-per-hour", type=float, default=None,
                    help="your contract rate; overrides the built-in placeholder")
    ap.add_argument("--scaling-efficiency", type=float, default=0.98,
                    help="derate for multi-GPU coordination")
    ap.add_argument("--from-reports", type=Path, default=Path("reports"))


def main(args: argparse.Namespace) -> int:
    measured = load_measured(args.from_reports) if args.from_reports.exists() else {}

    if args.tokens_per_product is not None:
        tpp, tpp_src = args.tokens_per_product, "ASSUMED (--tokens-per-product)"
    elif "tokens_per_product" in measured:
        tpp, tpp_src = measured["tokens_per_product"], measured["tokens_source"]
    else:
        tpp = ASSUMED_TOKENS_PER_PRODUCT
        tpp_src = "ASSUMED (run `catalog-bench tokens` to measure)"

    # Sizing runs on valid products/sec. A measured run reports it directly; the pre-GPU
    # path has to derive it from tok/s, and that derivation carries an implicit goodput of
    # 1.0 — true of a guided configuration, optimistic for anything else.
    if args.tokens_per_sec is not None:
        tps, tps_src = args.tokens_per_sec, "ASSUMED (--tokens-per-sec)"
        pps, pps_src = tps / tpp, "derived from --tokens-per-sec; goodput ASSUMED 1.0"
    elif "products_per_sec" in measured:
        tps, tps_src = measured["tokens_per_sec"], "measured (same run)"
        pps, pps_src = measured["products_per_sec"], measured["rate_source"]
    else:
        tps, tps_src = ASSUMED_TOKENS_PER_SEC, "ASSUMED (needs the GPU run)"
        pps, pps_src = tps / tpp, "derived from the placeholder; goodput ASSUMED 1.0"

    rate = args.gpu_cost_per_hour or GPU_HOURLY_USD[args.gpu]
    rate_src = "your rate" if args.gpu_cost_per_hour else f"placeholder for {args.gpu}"

    print("inputs")
    print(f"  tokens/product        {tpp:>12,.0f}   {tpp_src}")
    print(f"  engine tok/s per GPU  {tps:>12,.0f}   {tps_src}")
    print(f"  valid products/s/GPU  {pps:>12,.2f}   {pps_src}")
    print(f"  $/GPU-hour            {rate:>12,.2f}   {rate_src}")
    print(f"  scaling efficiency    {args.scaling_efficiency:>12.0%}")

    if "ASSUMED" in pps_src:
        print("\n  !! Throughput is ASSUMED. Every figure below is a projection, not a\n"
              "     capacity commitment. Run scripts/run_benchmark.sh to replace it.")

    r = size(args.catalog_size, args.sla_hours, pps, args.scaling_efficiency)
    cost = r["gpu_hours"] * rate

    print(f"\nsizing — {args.catalog_size:,} products in {args.sla_hours}h")
    print(f"  valid products/s/GPU  {r['products_per_sec_per_gpu']:>12,.1f}")
    print(f"  valid products/GPU-hr {r['products_per_gpu_hour']:>12,.0f}")
    print(f"  GPUs required         {r['gpus']:>12,}")
    print(f"  actual run time       {r['hours']:>12,.2f} h")
    print(f"  GPU-hours             {r['gpu_hours']:>12,.1f}")
    print(f"  total cost            {cost:>12,.2f} USD")
    print(f"  cost per 1M products  {cost / (args.catalog_size / 1e6):>12,.2f} USD")

    print("\nworked examples (same inputs)")
    print(f"  {'catalog':>12} {'SLA':>6} {'GPUs':>6} {'run time':>10} "
          f"{'GPU-hours':>10} {'$/1M':>10}")
    for n in (100_000, 1_000_000, 10_000_000, 100_000_000):
        for sla in (1.0, 8.0):
            e = size(n, sla, pps, args.scaling_efficiency)
            print(f"  {n:>12,} {sla:>5.0f}h {e['gpus']:>6,} {e['hours']:>9,.2f}h "
                  f"{e['gpu_hours']:>10,.1f} {e['gpu_hours'] * rate / (n / 1e6):>10,.2f}")

    print("\nformula:  GPUs = catalog / (valid_products_per_sec * SLA_seconds * scaling_eff)")
    return 0
