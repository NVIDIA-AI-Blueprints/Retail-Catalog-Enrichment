"""`catalog-bench` — one entrypoint for every command.

Each subcommand lives in its own module and exposes `add_args(parser)` and `main(args)`,
so the module is usable as a library and the CLI is a thin dispatch table.

    catalog-bench check-schema my.schema.json   # will my schema work, and what will it cost?
    catalog-bench capacity --catalog-size 40e6  # how many GPUs, and how much?
    catalog-bench catalog --n 10000             # generate a synthetic catalog
    catalog-bench tokens                        # tokens per product (no GPU needed)
    catalog-bench run --engine mock             # the pipeline benchmark
    catalog-bench sol --model ...               # the bare-engine ceiling (GPU only)
    catalog-bench report                        # render the benchmark report
    catalog-bench accuracy --outputs ...        # score enrichment against ground truth

The first two need no GPU and no model, and answer the sizing question *before* you book
hardware.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable

# (command, module, one-line help). Modules are imported lazily so that a command needing
# only jsonschema does not pay for pyarrow, and so `--help` works with a partial install.
COMMANDS: list[tuple[str, str, str]] = [
    ("check-schema", "schema_check",
     "lint an enrichment schema for grammar compatibility and token cost"),
    ("capacity", "capacity", "catalog size + SLA -> GPU count -> cost"),
    ("catalog", "catalog", "generate a synthetic catalog (and optional ground truth)"),
    ("tokens", "tokens", "measure tokens per product; build the bare-engine dataset"),
    ("run", "benchmark", "run the pipeline benchmark against an engine"),
    ("sol", "sol", "measure the bare-engine ceiling with trtllm-bench (GPU only)"),
    ("report", "report", "render the benchmark report from collected records"),
    ("accuracy", "accuracy", "score enrichment output against ground truth"),
]


def _load(module: str) -> tuple[Callable, Callable]:
    mod = importlib.import_module(f".{module}", package=__package__)
    return mod.add_args, mod.main


def _usage() -> str:
    width = max(len(name) for name, _, _ in COMMANDS)
    lines = [__doc__.splitlines()[0], "", "commands:"]
    lines += [f"  {name:<{width}}  {help_text}" for name, _, help_text in COMMANDS]
    lines += ["", "Run `catalog-bench COMMAND --help` for a command's options."]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    known = {name: module for name, module, _ in COMMANDS}

    # Dispatch on the command word before building a parser, so only the selected
    # command's module is imported. Otherwise `catalog-bench check-schema` — which needs
    # nothing but jsonschema — would pull in pyarrow and every engine backend.
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0 if argv else 1
    if argv[0] not in known:
        print(f"unknown command: {argv[0]!r}\n\n{_usage()}")
        return 2

    name, rest = argv[0], argv[1:]
    add_args, run = _load(known[name])
    help_text = next(h for n, _, h in COMMANDS if n == name)
    ap = argparse.ArgumentParser(
        prog=f"catalog-bench {name}", description=help_text,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(ap)
    return run(ap.parse_args(rest))


if __name__ == "__main__":
    raise SystemExit(main())
