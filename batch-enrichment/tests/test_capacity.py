"""The sizing model.

The failure this guards against is a fleet sized from raw throughput rather than valid
throughput — buying GPUs to produce output that fails schema validation.
"""

from __future__ import annotations

import json

from catalog_bench.capacity import load_measured, size


def test_sizing_is_linear_in_catalog_size():
    a = size(1_000_000, sla_hours=4, products_per_sec=38.31, scaling_efficiency=1.0)
    b = size(10_000_000, sla_hours=4, products_per_sec=38.31, scaling_efficiency=1.0)
    assert b["gpus"] >= 9 * a["gpus"]


def test_tighter_sla_needs_more_gpus():
    slow = size(10_000_000, 8, products_per_sec=38.31, scaling_efficiency=1.0)
    fast = size(10_000_000, 4, products_per_sec=38.31, scaling_efficiency=1.0)
    assert fast["gpus"] > slow["gpus"]


def test_integer_gpu_count_meets_the_sla():
    """GPUs round up, so actual run time must land at or under the SLA."""
    r = size(10_000_000, sla_hours=4, products_per_sec=38.31, scaling_efficiency=0.98)
    assert r["hours"] <= 4.0


def test_measured_rate_is_ranked_by_valid_products(tmp_path):
    """Unguided is faster on raw products/sec and fails 18% of validation. The guided row
    must win, or the fleet gets sized to produce output that is thrown away."""
    (tmp_path / "a.json").write_text(json.dumps({
        "kind": "probe", "mock": False, "grammar": "off", "products_per_sec": 42.76,
        "goodput": 0.824, "total_tokens_per_sec": 44_000.0}))
    (tmp_path / "b.json").write_text(json.dumps({
        "kind": "probe", "mock": False, "grammar": "xgrammar", "products_per_sec": 38.31,
        "goodput": 1.0, "total_tokens_per_sec": 40_000.0}))

    m = load_measured(tmp_path)
    assert m["products_per_sec"] == 38.31          # not 42.76 * 0.824 == 35.23
    assert "grammar=xgrammar" in m["rate_source"]


def test_mock_records_never_set_a_rate(tmp_path):
    """A constant someone typed on a laptop must not reach a cost model."""
    (tmp_path / "m.json").write_text(json.dumps({
        "kind": "probe", "mock": True, "grammar": "xgrammar", "products_per_sec": 9_999.0,
        "goodput": 1.0, "total_tokens_per_sec": 1e9}))
    assert "products_per_sec" not in load_measured(tmp_path)
