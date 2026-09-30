"""Score enrichment output against ground truth.

## Read this before quoting any number from here

The labels are **synthetic**. `catalog-bench catalog --ground-truth` plants attributes and
then writes a listing around them, in its own vocabulary. So this measures whether a model
can recover what we planted, in text we wrote. That makes it:

  * a genuine **regression harness** — swap the model, schema or prompt and see
    immediately whether extraction got worse;
  * a genuine **smoke test** — a model scoring 40% here is broken, and you want to know
    that before a ten-hour run;
  * **not** evidence about a real catalog. Real listings carry vocabulary, ambiguity and
    noise this generator does not reproduce. Shipping these as an accuracy claim would be
    dishonest. Point `--ground-truth` at your own labelled set and the numbers mean
    something.

## The metric that matters most

Per-field accuracy is the obvious number; `hallucination_rate` is the one a catalog team
will actually ask about. The generator marks a field null when its value never made it
into the listing text — an unstated material is genuinely unstated. Those rows are the
test: a well-behaved model returns null, and a model that invents a plausible material is
doing the one thing a catalog team cannot tolerate.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

SCORED_FIELDS = ["category", "brand", "primary_color", "material", "sizes"]


def norm(v: Any) -> str | None:
    """Normalize for comparison: case, whitespace, trivial punctuation.

    Deliberately shallow. A deeper normalizer — synonyms, stemming, colour ontologies —
    would inflate the score by encoding our own opinion of what counts as correct, and the
    point of this file is to be a regression signal, not a flattering one.
    """
    if v is None:
        return None
    s = re.sub(r"[^a-z0-9 ]+", " ", str(v).strip().lower())
    return re.sub(r"\s+", " ", s).strip() or None


def score_field(pred: Any, gold: Any) -> str:
    p, g = norm(pred), norm(gold)
    if g is None:
        # Nothing in the listing supported an answer.
        return "correct_abstain" if p is None else "hallucinated"
    if p is None:
        return "missed"
    return "correct" if p == g else "wrong"


def score_sizes(pred: Any, gold: list[str]) -> str:
    """Sizes is a set, so exact match is the wrong shape. Scored as set equality."""
    g = {norm(x) for x in (gold or []) if norm(x)}
    p = {norm(x) for x in (pred or []) if norm(x)} if isinstance(pred, list) else set()
    if not g:
        return "correct_abstain" if not p else "hallucinated"
    if not p:
        return "missed"
    return "correct" if p == g else "wrong"


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--outputs", type=Path, required=True,
                    help="Parquet written by `catalog-bench run --save-outputs`")
    ap.add_argument("--ground-truth", type=Path, default=Path("data/ground_truth.parquet"))
    ap.add_argument("--out", type=Path, default=Path("reports/accuracy.json"))


def main(args: argparse.Namespace) -> int:
    truth = {r["product_id"]: r for r in pq.read_table(args.ground_truth).to_pylist()}
    outputs = pq.read_table(args.outputs).to_pylist()

    buckets: dict[str, dict[str, int]] = {f: {} for f in SCORED_FIELDS}
    n_scored = n_unparseable = 0

    for row in outputs:
        gold = truth.get(row["product_id"])
        if gold is None:
            continue
        try:
            pred = json.loads(row["enrichment_json"] or "")
        except (json.JSONDecodeError, TypeError):
            n_unparseable += 1
            continue
        n_scored += 1
        for f in SCORED_FIELDS:
            b = (score_sizes(pred.get(f), gold[f]) if f == "sizes"
                 else score_field(pred.get(f), gold[f]))
            buckets[f][b] = buckets[f].get(b, 0) + 1

    if not n_scored:
        raise SystemExit("no scoreable rows — do the product_ids match?")

    print(f"scored {n_scored} products"
          + (f"  ({n_unparseable} unparseable outputs skipped)" if n_unparseable else ""))
    print("\n  NOTE: synthetic labels. A regression signal, not an accuracy claim.\n")
    print(f"  {'field':<15} {'accuracy':>9} {'wrong':>7} {'missed':>7} "
          f"{'halluc.':>8}  {'(n stated)':>11}")

    summary: dict[str, Any] = {
        "kind": "accuracy",
        "labels": "synthetic unless --ground-truth points at real labels",
        "ground_truth": str(args.ground_truth),
        "n_scored": n_scored,
        "n_unparseable": n_unparseable,
        "fields": {},
    }

    for f in SCORED_FIELDS:
        b = buckets[f]
        stated = b.get("correct", 0) + b.get("wrong", 0) + b.get("missed", 0)
        unstated = b.get("correct_abstain", 0) + b.get("hallucinated", 0)
        acc = b.get("correct", 0) / stated if stated else float("nan")
        halluc = b.get("hallucinated", 0) / unstated if unstated else 0.0
        print(f"  {f:<15} {acc:>8.1%} {b.get('wrong', 0):>7} {b.get('missed', 0):>7} "
              f"{halluc:>7.1%}  {stated:>11}")
        summary["fields"][f] = {
            "accuracy_on_stated": None if stated == 0 else round(acc, 4),
            "hallucination_rate_on_unstated": round(halluc, 4),
            "n_stated": stated,
            "n_unstated": unstated,
            **b,
        }

    macro = [summary["fields"][f]["accuracy_on_stated"] for f in SCORED_FIELDS
             if summary["fields"][f]["accuracy_on_stated"] is not None]
    summary["macro_accuracy"] = round(sum(macro) / len(macro), 4) if macro else None
    if summary["macro_accuracy"] is not None:
        print(f"\n  macro accuracy   {summary['macro_accuracy']:.1%}")
    print("  accuracy is over listings that STATE the attribute; hallucination rate is")
    print("  over those that do not. The second is the one a catalog team cares about.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out}")
    return 0
