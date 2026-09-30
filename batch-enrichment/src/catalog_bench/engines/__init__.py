"""The engine seam: `submit` / `drain` / `capacity_hint`.

One small interface, three implementations, so the same benchmark harness can measure
in-process TensorRT-LLM, any OpenAI-compatible server, and a laptop simulation without
the pipeline code knowing which it is talking to.

The mock exists so the harness can be *known* to work before a GPU is on the clock,
rather than debugged during a booked session.

A note carried over from building the TRT-LLM binding, because it will bite anyone
porting this seam elsewhere: `submit`/`drain` mirrors an enqueue/await-responses shape,
but the TRT-LLM Python API does not have that shape. `LLM.generate_async()` returns one
future per request; there is no queue of completed responses to drain. So `drain()` is
reconstructed as a poll over a pending set, which makes it O(in-flight) per call. The
interface survives across backends; its cost model does not.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

GRAMMARS = ("off", "xgrammar", "llguidance")


@dataclass(slots=True)
class GenRequest:
    """One product's worth of work.

    Carries token ids rather than text: tokenization is the pipeline's job, not the
    engine's. That split is deliberate — the benchmark attributes host CPU time to
    tokenization specifically, and an engine that tokenizes internally hides exactly the
    number we are trying to measure.
    """

    req_id: int
    product_id: str
    prompt_token_ids: list[int]
    max_tokens: int
    # Raw ChatCompletions messages, populated only for the HTTP engine so it can send
    # structured system/user roles rather than a single decoded string.
    messages: list[dict] | None = None


@dataclass(slots=True)
class GenResult:
    req_id: int
    product_id: str
    text: str
    prompt_tokens: int
    output_tokens: int
    finish_reason: str | None = None
    error: str | None = None


@dataclass(slots=True)
class TokenBudget:
    """Scheduler headroom, in tokens.

    Backpressure is measured in tokens rather than requests because a batch of
    2000-token products and a batch of 200-token products are not the same load. The
    benchmark reports the budget it was configured with; a production implementation
    would query the executor's live KV-cache headroom instead.
    """

    max_in_flight_tokens: int
    in_flight_tokens: int

    @property
    def free(self) -> int:
        return max(0, self.max_in_flight_tokens - self.in_flight_tokens)


class Engine(Protocol):
    is_mock: bool
    pending: int

    def submit(self, req: GenRequest) -> int: ...
    def drain(self, max_wait: float) -> list[GenResult]: ...
    def capacity_hint(self) -> TokenBudget: ...
    def shutdown(self) -> None: ...


def validate_grammar(grammar: str, schema: dict[str, Any] | None) -> None:
    """Shared construction-time guard for every engine."""
    if grammar not in GRAMMARS:
        raise ValueError(f"unknown grammar backend: {grammar!r} (expected one of {GRAMMARS})")
    if grammar != "off" and schema is None:
        raise ValueError("guided decoding requested but no schema supplied")


def build_engine(kind: str, **kwargs: Any) -> Engine:
    """Construct an engine by name, importing its backend lazily.

    Lazy so that `import catalog_bench.engines` works on a laptop with neither
    tensorrt_llm nor requests installed.
    """
    if kind == "mock":
        from .mock import MockEngine

        return MockEngine(**kwargs)
    if kind == "trtllm":
        from .trtllm import TrtllmEngine

        return TrtllmEngine(**kwargs)
    if kind == "http":
        from .http import HttpEngine

        return HttpEngine(**kwargs)
    raise ValueError(f"unknown engine: {kind!r} (expected mock, trtllm or http)")


__all__ = [
    "GRAMMARS",
    "Engine",
    "GenRequest",
    "GenResult",
    "TokenBudget",
    "build_engine",
    "validate_grammar",
]
