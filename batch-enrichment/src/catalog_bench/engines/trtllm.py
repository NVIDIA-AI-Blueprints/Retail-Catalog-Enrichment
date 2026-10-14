"""In-process TensorRT-LLM. No HTTP anywhere in this path.

Written against TensorRT-LLM 1.3.0rc9. The source references in the comments below are
there so a reviewer can check the binding against their own checkout without a GPU.
"""

from __future__ import annotations

import json
from typing import Any

from . import GenRequest, GenResult, TokenBudget, validate_grammar


class TrtllmEngine:
    is_mock = False

    def __init__(
        self,
        model: str,
        *,
        grammar: str = "xgrammar",
        schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
        max_in_flight_tokens: int = 1 << 20,
        enable_thinking: bool = False,
        stop: list[str] | None = None,
        **llm_kwargs: Any,
    ) -> None:
        from tensorrt_llm import LLM, SamplingParams
        from tensorrt_llm.llmapi import GuidedDecodingParams

        validate_grammar(grammar, schema)

        self.grammar = grammar
        self.max_tokens = max_tokens
        self._budget = TokenBudget(max_in_flight_tokens=max_in_flight_tokens,
                                   in_flight_tokens=0)

        # The backend is chosen once, at construction (llm_args.py:4489). It cannot be
        # switched per request, which is why the benchmark varies it across runs rather
        # than within one.
        if grammar != "off":
            llm_kwargs["guided_decoding_backend"] = grammar

        self.llm = LLM(model=model, **llm_kwargs)
        self.tokenizer = self.llm.tokenizer

        # Reasoning off. `enable_thinking` is read from chat_template_kwargs by the
        # nemotron reasoning parser (reasoning_parser.py:446-470), so this is the
        # supported control rather than a prompt-level plea. max_tokens and the stop list
        # are the independent backstop: a checkpoint that will not fully suppress its
        # reasoning trace can at least be prevented from spending 20x the token budget
        # on one.
        self.chat_template_kwargs = {"enable_thinking": enable_thinking}

        # One SamplingParams shared by every request. The grammar compiles once per
        # distinct schema, and sharing this object is what keeps that true.
        self.sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            stop=stop or [],
            guided_decoding=(
                GuidedDecodingParams(json=json.dumps(schema)) if grammar != "off" else None
            ),
        )

        self._pending: dict[int, Any] = {}
        self._req_tokens: dict[int, int] = {}      # prompt + max_tokens, for the budget
        self._prompt_tokens: dict[int, int] = {}   # prompt only, for reporting
        self._products: dict[int, str] = {}

    @property
    def pending(self) -> int:
        return len(self._pending)

    def apply_chat_template(self, system: str, user: str) -> str:
        """Render the chat template with reasoning disabled.

        Lives on the engine rather than in the pipeline because the template and its
        `enable_thinking` handling are properties of the model, not of the pipeline.
        """
        return self.tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )

    def submit(self, req: GenRequest) -> int:
        # TokensPrompt (inputs/data.py:39) takes pre-tokenized input directly, so the
        # engine never sees our prompt text and never re-tokenizes it.
        out = self.llm.generate_async(
            {"prompt_token_ids": req.prompt_token_ids},
            sampling_params=self.sampling_params,
        )
        self._pending[req.req_id] = out
        self._req_tokens[req.req_id] = len(req.prompt_token_ids) + req.max_tokens
        self._prompt_tokens[req.req_id] = len(req.prompt_token_ids)
        self._products[req.req_id] = req.product_id
        self._budget.in_flight_tokens += self._req_tokens[req.req_id]
        return req.req_id

    def drain(self, max_wait: float) -> list[GenResult]:
        """Reap finished futures.

        Non-blocking first pass over `.finished` (result.py:1039). Only if nothing has
        completed do we block, and then on a single future with a timeout — blocking on
        all of them would stall the feeder and starve the executor.
        """
        done = [rid for rid, fut in self._pending.items() if fut.finished]

        if not done and max_wait > 0 and self._pending:
            rid = next(iter(self._pending))
            try:
                self._pending[rid].result(timeout=max_wait)
                done = [rid]
            except TimeoutError:
                return []

        results = []
        for rid in done:
            fut = self._pending.pop(rid)
            self._budget.in_flight_tokens -= self._req_tokens.pop(rid)
            submitted_tokens = self._prompt_tokens.pop(rid)
            product_id = self._products.pop(rid)
            try:
                out = fut.outputs[0]
                # TRT-LLM's own code reaches for this via getattr (result.py:692), i.e.
                # it is not a guaranteed attribute. We submitted the ids, so fall back to
                # our own count rather than risk an AttributeError mid-run.
                prompt_ids = getattr(fut, "prompt_token_ids", None)
                results.append(GenResult(
                    req_id=rid,
                    product_id=product_id,
                    text=out.text,
                    prompt_tokens=len(prompt_ids) if prompt_ids else submitted_tokens,
                    output_tokens=len(out.token_ids),
                    finish_reason=out.finish_reason,
                ))
            except Exception as exc:
                results.append(GenResult(rid, product_id, "", 0, 0, error=repr(exc)))
        return results

    def capacity_hint(self) -> TokenBudget:
        return self._budget

    def shutdown(self) -> None:
        self.llm.shutdown()

    def describe(self) -> dict[str, Any]:
        return {"device": "gpu", "serving": "in_process", "grammar_backend": self.grammar}
