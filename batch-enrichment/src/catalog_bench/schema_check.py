"""Check an enrichment schema before you spend GPU hours on it.

Swapping the enrichment schema is the first thing anyone integrating this does. Two ways
it goes wrong, both expensive, neither obvious from reading your schema:

1. **The grammar will not compile.** Guided decoding compiles your JSON Schema to a
   grammar. Not every keyword is supported, and an unsupported one surfaces as a failure
   at engine start — after a 30B model has loaded.

2. **It compiles, and quietly costs you 3x throughput.** The worse case. Throughput moves
   inversely with tokens per product, and schema shape alone drives a multiple-fold
   spread. An unbounded array or a chatty free-text field does not fail; it just makes
   your target unreachable, and you find out at the end of the run.

So this reports both: what will break, and what it will cost.

    catalog-bench check-schema my_attributes.schema.json

Exit code is non-zero if any ERROR is found, so it drops straight into CI. It is the
cheapest thing in this repo and prevents the most expensive mistake.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .prompts import (
    DEFAULT_SCHEMA,
    SCHEMA_MODES,
    encode,
    load_schema,
    load_tokenizer,
    render_system,
)
from .tokens import osl_estimates

# Keywords that JSON-Schema-to-grammar compilers do not reliably support. Support has been
# growing release to release, so these are warnings with a reason rather than hard
# failures.
RISKY_KEYWORDS = {
    "pattern": "regex constraints on strings are not reliably supported",
    "format": "semantic formats (date-time, email, uri) are advisory and usually ignored",
    "minLength": "string length bounds have varied in support across versions",
    "maxLength": "string length bounds have varied in support across versions",
    "allOf": "schema composition is not reliably supported; inline the result instead",
    "anyOf": 'schema composition is not reliably supported; use a ["a","b"] type union',
    "oneOf": 'schema composition is not reliably supported; use a ["a","b"] type union',
    "not": "negation is not supported",
    "$ref": "references may not resolve during grammar compilation; inline them",
    "patternProperties": "dynamic key names make the grammar unbounded",
    "propertyNames": "dynamic key names make the grammar unbounded",
}

MAX_REASONABLE_ATTRS = 40
MIN_REASONABLE_ATTRS = 5
# Above this ratio against the reference schema, the token cost is worth flagging.
COST_WARN_RATIO = 1.25


@dataclass
class Findings:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    unbounded_arrays: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def inspect(schema: dict[str, Any]) -> Findings:
    """Walk every subschema and collect findings. Pure — no printing, no globals."""
    f = Findings()
    _walk(schema, "$", f)

    n_attrs = len(schema.get("properties", {}))
    if n_attrs > MAX_REASONABLE_ATTRS:
        f.warnings.append(
            f"$: {n_attrs} attributes. Grammar complexity and output length both scale "
            "with this; consider two passes if throughput matters more than a single "
            "round trip.")
    if n_attrs < MIN_REASONABLE_ATTRS:
        f.notes.append(
            f"$: only {n_attrs} attributes — benchmark numbers from a schema this small "
            "will not generalize to a realistic one.")
    return f


def _walk(node: Any, path: str, f: Findings) -> None:
    if isinstance(node, list):
        for i, item in enumerate(node):
            _walk(item, f"{path}[{i}]", f)
        return
    if not isinstance(node, dict):
        return

    for kw, why in RISKY_KEYWORDS.items():
        if kw in node:
            f.warnings.append(f"{path}: `{kw}` — {why}")

    node_type = node.get("type")
    types = node_type if isinstance(node_type, list) else [node_type]

    if "object" in types or "properties" in node:
        if node.get("additionalProperties") is not False:
            f.errors.append(
                f"{path}: `additionalProperties` is not false. The grammar will permit "
                "keys you did not define, so output length has no ceiling and validation "
                "cannot catch invented fields.")
        props = node.get("properties", {})
        required = set(node.get("required", []))
        optional = [k for k in props if k not in required]
        if optional:
            f.warnings.append(
                f"{path}: {len(optional)} optional field(s) ({', '.join(optional[:4])}"
                f"{'...' if len(optional) > 4 else ''}). Optional keys make the emitted "
                "structure vary per product, which widens output length and costs prefix "
                'determinism. Prefer required + a ["type","null"] union.')
        for name, sub in props.items():
            _walk(sub, f"{path}.{name}", f)

    if "array" in types:
        if "maxItems" not in node:
            f.unbounded_arrays.append(path)
            f.errors.append(
                f"{path}: array has no `maxItems`. This is the single most expensive "
                "mistake available here — an unbounded array is an unbounded token "
                "budget, and one runaway list can dominate your output length.")
        _walk(node.get("items", {}), f"{path}[]", f)

    if "string" in types and "enum" not in node and len(types) == 1:
        f.notes.append(f"{path}: free-text string. If the value set is closed, an `enum` "
                       "is cheaper to decode and trivially validatable.")


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("schema", type=Path, help="the enrichment schema to check")
    ap.add_argument("--tokenizer", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8")
    ap.add_argument("--compare-to", type=Path, default=DEFAULT_SCHEMA,
                    help="reference schema to compare token cost against")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="the max_tokens you plan to run with; checked against worst-case "
                         "output length")


def main(args: argparse.Namespace) -> int:
    schema = load_schema(args.schema)
    print(f"checking {args.schema}\n")

    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:
        print(f"ERROR  not a valid JSON Schema: {exc}")
        return 1

    f = inspect(schema)

    tok = load_tokenizer(args.tokenizer)
    reference = load_schema(args.compare_to)
    ours = osl_estimates(schema, tok)
    ref = osl_estimates(reference, tok)

    prefix = {m: len(encode(tok, render_system(schema, m))) for m in SCHEMA_MODES}
    ref_prefix = {m: len(encode(tok, render_system(reference, m))) for m in SCHEMA_MODES}

    print("token cost")
    print(f"  {'':16} {'this schema':>12} {'reference':>12}")
    for m in SCHEMA_MODES:
        print(f"  prefix ({m:<7}) {prefix[m]:>12} {ref_prefix[m]:>12}")

    # An unbounded array has no worst case. The estimator assumes 10 items so the number
    # stays finite, but printing that as a max would be worse than printing nothing — a
    # schema with two unbounded arrays would score *cheaper* than the reference, which is
    # exactly backwards.
    unbounded = bool(f.unbounded_arrays)
    print(f"  {'OSL min':16} {ours['min']:>12} {ref['min']:>12}")
    print(f"  {'OSL typical':16} {ours['typical']:>12} {ref['typical']:>12}"
          + ("   (understated: unbounded arrays assumed at 10 items)" if unbounded else ""))
    print(f"  {'OSL max':16} {'unbounded' if unbounded else ours['max']:>12} {ref['max']:>12}")
    print(f"  {'OSL fixed':16} {ours['fixed']:>12} {ref['fixed']:>12}"
          "   (the part the grammar pins; the rest is the model's choice)")

    ratio = ours["typical"] / ref["typical"] if ref["typical"] else 1.0
    if ratio > COST_WARN_RATIO:
        f.warnings.append(
            f"$: typical output length is {ratio:.1f}x the reference schema's. Throughput "
            f"moves inversely with tokens per product, so expect roughly {1 / ratio:.0%} "
            "of the reference products/sec on identical hardware.")
    if ours["max"] > args.max_tokens:
        f.errors.append(
            f"$: worst-case output length ({ours['max']}) exceeds --max-tokens "
            f"({args.max_tokens}). Fully-populated outputs will be truncated mid-JSON and "
            f"fail validation. Raise max_tokens above {ours['max']} or tighten the bounds.")

    print()
    for label, items in (("ERROR", f.errors), ("WARN", f.warnings), ("NOTE", f.notes)):
        for msg in items:
            print(f"{label:<6} {msg}")

    print()
    if f.errors:
        print(f"FAILED — {len(f.errors)} error(s), {len(f.warnings)} warning(s)")
        return 1
    print(f"OK — {len(f.warnings)} warning(s), {len(f.notes)} note(s)")
    return 0
