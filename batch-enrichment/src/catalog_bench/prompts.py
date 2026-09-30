"""Prompt rendering, shared by every command that touches a prompt.

Small on purpose. It exists so the token accounting and the benchmark render prompts
through exactly the same code — if they diverged, the measured input length would not be
the length the benchmark actually paid for, and every throughput ratio computed from it
would be wrong in a way nobody would notice.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

PACKAGE_ROOT = Path(__file__).resolve().parent
TEMPLATE_DIR = PACKAGE_ROOT / "templates"
DEFAULT_SCHEMA = PACKAGE_ROOT / "schemas" / "product_attributes.schema.json"

SCHEMA_MODES = ("full", "compact", "none")


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        undefined=StrictUndefined,   # a silently-empty prompt field would corrupt ISL
        trim_blocks=True,
        lstrip_blocks=True,
    )


def load_schema(path: Path | str = DEFAULT_SCHEMA) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def load_tokenizer(name_or_path: str):
    """Load a tokenizer from a local tokenizer.json or a Hugging Face id.

    Token accounting is the one measurement here obtainable without a GPU, so a failure
    to load a tokenizer must not read like a bug in the harness — it needs to say what
    to pass instead.
    """
    from tokenizers import Tokenizer

    p = Path(name_or_path)
    if p.exists():
        return Tokenizer.from_file(str(p))
    try:
        return Tokenizer.from_pretrained(name_or_path)
    except Exception as exc:
        raise SystemExit(
            f"could not load tokenizer {name_or_path!r}: {type(exc).__name__}\n\n"
            "If this is a gated repo, either authenticate (`hf auth login`) or pass a\n"
            "public tokenizer for a shape estimate, e.g.\n"
            "  --tokenizer Qwen/Qwen2.5-1.5B-Instruct\n\n"
            "Tokenizers of similar vocabulary size agree on input length to within a\n"
            "couple of percent, so the shape estimate is usable. Published numbers must\n"
            "ultimately use the model's own tokenizer."
        ) from exc


def encode(tokenizer: Any, text: str) -> list[int]:
    """Normalize two different `encode` contracts into one.

    `tokenizers.Tokenizer` returns an Encoding whose ids live on `.ids`. TRT-LLM's
    `TransformersTokenizer.encode` returns a plain `list[int]`. Same method name,
    different return type — so the obvious `.ids` works on a laptop and raises
    AttributeError on the GPU.
    """
    out = tokenizer.encode(text, add_special_tokens=False)
    return out.ids if hasattr(out, "ids") else out


def compact_fields(schema: dict[str, Any]) -> list[str]:
    """One terse line per field: name, type or enum, and the first clause of its description.

    This is the `schema_mode='compact'` payload. Guided decoding already enforces syntax,
    so restating the JSON Schema verbatim in the prompt pays twice for the same
    constraint — once in the grammar, once in tokens. What the model still needs is the
    semantics: which field means what.
    """
    lines = []
    for name, spec in schema["properties"].items():
        if "enum" in spec:
            kind = "|".join(spec["enum"])
        elif spec.get("type") == "array":
            kind = f"array[str] max {spec.get('maxItems', '-')}"
        else:
            t = spec.get("type")
            kind = "/".join(t) if isinstance(t, list) else str(t)
        desc = spec.get("description", "").split(".")[0]
        lines.append(f"{name} ({kind}): {desc}" if desc else f"{name} ({kind})")
    return lines


def render_system(schema: dict[str, Any], schema_mode: str = "compact") -> str:
    """The invariant prefix. Rendered once per run, never per product."""
    if schema_mode not in SCHEMA_MODES:
        raise ValueError(f"unknown schema_mode: {schema_mode!r} (expected {SCHEMA_MODES})")
    return _env().get_template("system.j2").render(
        schema_mode=schema_mode,
        schema_json=json.dumps(schema, indent=2) if schema_mode == "full" else "",
        compact_fields=compact_fields(schema) if schema_mode == "compact" else [],
    )


def render_user(product: dict[str, Any]) -> str:
    """The per-product suffix."""
    return _env().get_template("enrich.j2").render(product=product)
