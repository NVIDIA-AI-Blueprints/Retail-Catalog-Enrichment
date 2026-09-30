"""The TensorRT-LLM binding, against stubs shaped like the real API.

The GPU path cannot run on a laptop, but most of what breaks in it is not GPU behaviour —
it is API contract mismatches, and those are checkable here. Two live bugs were caught
this way before a GPU session was ever booked:

  1. mock-only kwargs reached `LLM()`, whose args model sets `extra="forbid"` — an
     immediate ValidationError at startup;
  2. `encode()` returns `list[int]` on TRT-LLM's tokenizer but an object with `.ids` on
     `tokenizers.Tokenizer`, so the laptop path worked and the GPU path would have raised
     AttributeError on the first product.

The stubs deliberately reproduce the awkward parts of the real API: `LLM` rejects unknown
kwargs, `encode` returns a bare list, and the result future has NO `prompt_token_ids`
attribute (TRT-LLM's own code reaches for it via getattr). A politely permissive stub
would test nothing.
"""

from __future__ import annotations

import json
import sys
import types
from typing import ClassVar

import pytest

from catalog_bench.engines import GenRequest
from catalog_bench.engines.mock import SAMPLE_ENRICHMENT

STUB_OUTPUT_TOKENS = 37


class _Out:
    def __init__(self) -> None:
        self.text = json.dumps(SAMPLE_ENRICHMENT)
        self.token_ids = [1] * STUB_OUTPUT_TOKENS
        self.finish_reason = "stop"


class _Fut:
    def __init__(self) -> None:
        self.finished = True
        self.outputs = [_Out()]
        # Deliberately no `prompt_token_ids`.


class _Tok:
    def encode(self, text, **kw):
        return [7] * 11          # a bare list, as TRT-LLM's tokenizer returns

    def apply_chat_template(self, conversation, **kw):
        return "CHAT:" + conversation[-1]["content"]


class _LLM:
    ALLOWED: ClassVar[set[str]] = {"model", "guided_decoding_backend", "tokenizer"}

    def __init__(self, **kw):
        extra = set(kw) - self.ALLOWED
        if extra:
            raise TypeError(f"LLM: unexpected kwargs (extra='forbid'): {sorted(extra)}")
        self.kw = kw
        self.tokenizer = _Tok()

    def generate_async(self, inputs, sampling_params=None):
        assert "prompt_token_ids" in inputs, "must submit pre-tokenized input"
        return _Fut()

    def shutdown(self) -> None:
        pass


class _SamplingParams:
    def __init__(self, **kw):
        self.__dict__.update(kw)


@pytest.fixture
def TrtllmEngine(monkeypatch):
    """Install the stub modules, then import the engine against them."""
    trt = types.ModuleType("tensorrt_llm")
    trt.LLM, trt.SamplingParams = _LLM, _SamplingParams
    api = types.ModuleType("tensorrt_llm.llmapi")
    api.GuidedDecodingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "tensorrt_llm", trt)
    monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi", api)

    from catalog_bench.engines.trtllm import TrtllmEngine as cls

    return cls


def test_grammar_backend_is_set_once(TrtllmEngine, schema):
    e = TrtllmEngine("fake/model", grammar="xgrammar", schema=schema, max_tokens=512)
    assert e.llm.kw.get("guided_decoding_backend") == "xgrammar"


def test_grammar_off_leaves_the_backend_unset(TrtllmEngine):
    e = TrtllmEngine("fake/model", grammar="off", max_tokens=512)
    assert e.llm.kw.get("guided_decoding_backend") is None


def test_reasoning_is_disabled(TrtllmEngine, schema):
    e = TrtllmEngine("fake/model", grammar="xgrammar", schema=schema)
    assert e.chat_template_kwargs == {"enable_thinking": False}


def test_submit_and_drain_round_trip(TrtllmEngine, schema):
    e = TrtllmEngine("fake/model", grammar="xgrammar", schema=schema, max_tokens=512)
    ids = e.tokenizer.encode(e.apply_chat_template("SYS", "USR"))

    e.submit(GenRequest(0, "SKU-1", ids, 512))
    assert e.pending == 1
    assert e.capacity_hint().in_flight_tokens == len(ids) + 512

    results = e.drain(max_wait=0.0)
    assert len(results) == 1
    assert results[0].error is None
    assert results[0].output_tokens == STUB_OUTPUT_TOKENS
    # The future has no prompt_token_ids, so this must fall back to our own count.
    assert results[0].prompt_tokens == len(ids)

    assert e.pending == 0
    assert e.capacity_hint().in_flight_tokens == 0


@pytest.mark.parametrize("kwargs, why", [
    ({"grammar": "xgrammar"}, "guided decoding without a schema"),
    ({"grammar": "bogus", "schema": {}}, "unknown grammar backend"),
])
def test_invalid_construction_is_rejected(TrtllmEngine, kwargs, why):
    with pytest.raises(ValueError):
        TrtllmEngine("fake/model", **kwargs)


def test_surplus_kwargs_reach_llm_and_are_rejected(TrtllmEngine, schema):
    """The strict-kwargs failure mode is wanted: a mis-spelled memory limit must fail at
    startup, not surface as an OOM twenty minutes into a model load."""
    with pytest.raises(TypeError, match="unexpected kwargs"):
        TrtllmEngine("fake/model", grammar="off", sim_tokens_per_sec=20_000)
