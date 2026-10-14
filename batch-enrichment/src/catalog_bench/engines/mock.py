"""CPU-only stand-in. Proves the harness runs; produces NO valid measurements.

Every record it feeds is stamped `mock: true`, and `report.py` refuses to render a
verdict from one. That refusal is the point — a mock that can be mistaken for a
measurement is worse than no mock at all.

It models decode as a token budget shared across in-flight requests, which reproduces
the one behaviour the feeder loop actually has to cope with: per-request latency rises
with occupancy while aggregate throughput stays flat.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any

from . import GenRequest, GenResult, TokenBudget

# A schema-valid enrichment result, used as the simulated model output. Public because
# the tests use it as their canonical valid instance — one definition, so a schema change
# cannot leave the mock producing output the validator rejects.
SAMPLE_ENRICHMENT = {
    "normalized_title": "Brand01 Merino Base Layer",
    "brand": "Brand01",
    "category": "apparel",
    "subcategory": "base layers",
    "primary_color": "navy",
    "secondary_colors": [],
    "material": "merino wool",
    "sizes": ["S", "M", "L"],
    "gender": "unisex",
    "age_group": "adult",
    "season": "winter",
    "style_tags": ["athletic", "minimalist"],
    "key_features": ["moisture wicking", "naturally antimicrobial"],
    "care_instructions": "Machine wash cold, tumble dry low.",
    "country_of_origin": None,
    "is_bundle": False,
    "unit_count": 1,
    "condition": "new",
    "search_keywords": ["merino", "base layer", "thermal"],
    "short_description": "A lightweight merino base layer for cold-weather layering.",
    "extraction_confidence": "high",
}


class MockEngine:
    is_mock = True

    def __init__(
        self,
        *,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 512,
        max_in_flight_tokens: int = 1 << 20,
        sim_tokens_per_sec: float = 20_000.0,
        invalid_rate: float = 0.0,
        seed: int = 0,
        **_ignored: Any,
    ) -> None:
        self.schema = schema
        self.max_tokens = max_tokens
        self.sim_tokens_per_sec = sim_tokens_per_sec
        self.invalid_rate = invalid_rate
        self._rng = random.Random(seed)
        self._budget = TokenBudget(max_in_flight_tokens=max_in_flight_tokens,
                                   in_flight_tokens=0)
        self._pending: dict[int, dict[str, Any]] = {}
        self._last_tick = time.perf_counter()

    @property
    def pending(self) -> int:
        return len(self._pending)

    def _sample_output(self) -> tuple[str, int]:
        text = json.dumps(SAMPLE_ENRICHMENT)
        if self.invalid_rate and self._rng.random() < self.invalid_rate:
            text = text[: len(text) // 2]  # truncated JSON, as an unguided model produces
        # Output length varies. A constant OSL would make the feeder's job artificially
        # easy and hide the behaviour this mock exists to reproduce.
        n_tokens = max(32, int(self._rng.gauss(self.max_tokens * 0.45,
                                               self.max_tokens * 0.12)))
        return text, min(n_tokens, self.max_tokens)

    def _advance(self) -> None:
        """Spend elapsed wall-clock as decode tokens, split across in-flight work."""
        now = time.perf_counter()
        elapsed, self._last_tick = now - self._last_tick, now
        if not self._pending:
            return
        share = (self.sim_tokens_per_sec * elapsed) / len(self._pending)
        for state in self._pending.values():
            state["remaining"] -= share

    def submit(self, req: GenRequest) -> int:
        text, n_out = self._sample_output()
        self._pending[req.req_id] = {
            "product_id": req.product_id,
            "text": text,
            "prompt_tokens": len(req.prompt_token_ids),
            "output_tokens": n_out,
            "remaining": n_out,
        }
        self._budget.in_flight_tokens += len(req.prompt_token_ids) + req.max_tokens
        return req.req_id

    def drain(self, max_wait: float) -> list[GenResult]:
        self._advance()
        done = [rid for rid, s in self._pending.items() if s["remaining"] <= 0]

        if not done and max_wait > 0 and self._pending:
            # Sleep just long enough for the soonest request to finish, capped at max_wait.
            soonest = min(s["remaining"] for s in self._pending.values())
            wait = min(max_wait, soonest * len(self._pending) / self.sim_tokens_per_sec)
            time.sleep(max(0.0, wait))
            self._advance()
            done = [rid for rid, s in self._pending.items() if s["remaining"] <= 0]

        results = []
        for rid in done:
            s = self._pending.pop(rid)
            self._budget.in_flight_tokens -= s["prompt_tokens"] + self.max_tokens
            results.append(GenResult(
                req_id=rid,
                product_id=s["product_id"],
                text=s["text"],
                prompt_tokens=s["prompt_tokens"],
                output_tokens=s["output_tokens"],
                finish_reason="stop",
            ))
        return results

    def capacity_hint(self) -> TokenBudget:
        return self._budget

    def shutdown(self) -> None:
        self._pending.clear()

    def describe(self) -> dict[str, Any]:
        return {"device": "none", "serving": "simulated"}
