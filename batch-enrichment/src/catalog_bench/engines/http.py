"""OpenAI-compatible ChatCompletions engine, for NIM / vLLM / SGLang.

Puts the same benchmark harness behind an HTTP server so in-process and served
inference can be compared with one variable changed at a time.

Two things about this path are easy to get wrong and both silently corrupt results:

* **Concurrency.** The thread pool caps how many requests are actually in flight. If it
  is smaller than `--in-flight`, the sweep runs entirely at the pool size and produces
  flat throughput that reads like a scheduler finding. `benchmark.py` sizes
  `max_workers` from `--in-flight` for exactly this reason. Verify against the server's
  own counter, not ours — see `describe()`.
* **Chat template flags.** The server applies the model's chat template, not us, so
  `enable_thinking` and `stop` have to be sent on the wire. Left off, this model reasons
  until it exhausts `max_tokens` on every product. The tell is `osl_mean` landing exactly
  on `max_tokens` with goodput at zero.

ISL and OSL come from the server's `usage` fields rather than our own tokenization. The
server re-tokenizes the text we send, so it sees a different count from the same prompt;
its number is the right one for the comparison, and ours is kept for attribution.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

from . import GenRequest, GenResult, TokenBudget, validate_grammar


class HttpEngine:
    is_mock = False

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:8000",
        grammar: str = "xgrammar",
        schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
        max_in_flight_tokens: int = 1 << 20,
        enable_thinking: bool = False,
        stop: list[str] | None = None,
        prefix_reuse: bool = False,
        max_workers: int = 256,  # set from --in-flight; must never cap concurrency
        tokenizer: Any = None,   # set after construction; used to decode ids -> text
        request_timeout: float = 300.0,
        **_ignored: Any,
    ) -> None:
        import requests

        validate_grammar(grammar, schema)

        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        self._chat_url = base_url.rstrip("/") + "/v1/chat/completions"
        self._models_url = base_url.rstrip("/") + "/v1/models"
        self._model = model
        self._timeout = request_timeout
        self.grammar = grammar
        self.max_tokens = max_tokens
        self.prefix_reuse = prefix_reuse
        self.tokenizer = tokenizer
        self.max_workers = max_workers

        # json_schema, not json_object. json_object constrains JSON syntax only, and the
        # failures this workload actually produces are schema-valid JSON with the wrong
        # cardinality or an out-of-enum value — which json_object does not catch.
        if grammar != "off" and schema is not None:
            self._response_format: dict[str, Any] = {
                "type": "json_schema",
                "json_schema": {"name": "product_attributes", "schema": schema,
                                "strict": True},
            }
        else:
            self._response_format = {"type": "text"}

        self._stop = stop or []
        self._enable_thinking = enable_thinking

        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._futures: dict[int, concurrent.futures.Future] = {}
        self._future_meta: dict[int, tuple[str, int, int]] = {}
        self._budget = TokenBudget(max_in_flight_tokens=max_in_flight_tokens,
                                   in_flight_tokens=0)
        self._server_info = self._probe_server()

    def _probe_server(self) -> dict[str, Any]:
        """Capture server metadata once at startup. Failure is non-fatal."""
        try:
            r = self._session.get(self._models_url, timeout=10)
            return r.json() if r.ok else {}
        except Exception:
            return {}

    def _decode_ids(self, token_ids: list[int]) -> str:
        tok = self.tokenizer
        if tok is None:
            raise RuntimeError("HttpEngine.tokenizer was never set")
        if not hasattr(tok, "decode"):
            raise RuntimeError(f"cannot decode with tokenizer type {type(tok).__name__}")
        return tok.decode(token_ids, skip_special_tokens=False)

    @property
    def pending(self) -> int:
        return len(self._futures)

    def submit(self, req: GenRequest) -> int:
        # Prefer pre-built messages so the server sees proper system/user roles; fall
        # back to decoding the ids into one user turn.
        messages = req.messages if req.messages is not None else [
            {"role": "user", "content": self._decode_ids(req.prompt_token_ids)}
        ]

        our_isl = len(req.prompt_token_ids)
        self._future_meta[req.req_id] = (req.product_id, our_isl, req.max_tokens)
        self._budget.in_flight_tokens += our_isl + req.max_tokens

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": 0.0,
            "response_format": self._response_format,
            "stream": False,
            # Must be sent: the server applies the chat template, so this is the only
            # place reasoning can be turned off on this path.
            "chat_template_kwargs": {"enable_thinking": self._enable_thinking},
        }
        if self._stop:
            payload["stop"] = self._stop

        session, url, timeout = self._session, self._chat_url, self._timeout

        def _call() -> dict[str, Any]:
            try:
                r = session.post(url, json=payload, timeout=timeout)
                r.raise_for_status()
                return r.json()
            except Exception as exc:
                return {"_error": repr(exc)}

        self._futures[req.req_id] = self._executor.submit(_call)
        return req.req_id

    def drain(self, max_wait: float) -> list[GenResult]:
        done = [rid for rid, fut in self._futures.items() if fut.done()]

        if not done and max_wait > 0 and self._futures:
            rid = next(iter(self._futures))
            try:
                self._futures[rid].result(timeout=max_wait)
                done = [rid]
            except concurrent.futures.TimeoutError:
                return []

        results = []
        for rid in done:
            fut = self._futures.pop(rid)
            product_id, our_isl, req_max_tokens = self._future_meta.pop(rid)
            self._budget.in_flight_tokens -= our_isl + req_max_tokens
            try:
                data = fut.result()
                if "_error" in data:
                    results.append(GenResult(rid, product_id, "", 0, 0,
                                             error=data["_error"]))
                    continue
                choice = data["choices"][0]
                usage = data.get("usage", {})
                results.append(GenResult(
                    req_id=rid,
                    product_id=product_id,
                    text=(choice.get("message") or {}).get("content") or "",
                    # The server's own count: it re-tokenized from text.
                    prompt_tokens=usage.get("prompt_tokens", our_isl),
                    output_tokens=usage.get("completion_tokens", 0),
                    finish_reason=choice.get("finish_reason"),
                ))
            except Exception as exc:
                results.append(GenResult(rid, product_id, "", 0, 0, error=repr(exc)))
        return results

    def capacity_hint(self) -> TokenBudget:
        return self._budget

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
        self._session.close()

    def describe(self) -> dict[str, Any]:
        import psutil

        return {
            "device": "gpu",
            "serving": "http",
            "server": self._model,
            "server_models_response": self._server_info,
            "grammar_backend": self.grammar,
            # There is no runtime API for which backend the server actually selected;
            # it has to be read from the server's startup log.
            "grammar_backend_used": "check_server_log",
            # Recorded so the concurrency cap lands in the artifact rather than only in
            # the source. Cross-check against the server's own "Running: N reqs" counter.
            "client_max_workers": self.max_workers,
            "prefix_reuse": self.prefix_reuse,
            "client_cpu_pct": psutil.cpu_percent(interval=None),
        }
