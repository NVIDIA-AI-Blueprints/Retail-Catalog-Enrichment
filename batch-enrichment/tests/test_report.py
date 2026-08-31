"""The report's refusals.

Each of these prevents a number that looks like a measurement and is not. They are the
reason the report is code rather than a document someone updates by hand.
"""

from __future__ import annotations

from catalog_bench.report import build_report, verdict


def probe(**kw):
    base = {
        "kind": "probe", "engine": "trtllm", "device": "gpu", "mock": False,
        "grammar": "xgrammar", "schema_mode": "compact", "in_flight": 1024,
        "total_tokens_per_sec": 40_000.0, "products_per_sec": 38.0, "goodput": 1.0,
        "host_cpu_share_of_wall": 0.09, "feeder_starvation_rate": 0.0,
        "stages": {"cpu_share_of_host": {"tokenize": 0.6}, "blocked_on_engine_share": 0.9},
    }
    return {**base, **kw}


def baseline(tok_s=42_188.0):
    return {"kind": "sol_baseline", "concurrency": 1024, "total_tokens_per_sec": tok_s,
            "products_per_sec": 40.0, "mock": False}


def test_no_verdict_without_a_baseline():
    assert "NO VERDICT" in verdict([probe()], sol=None)


def test_no_verdict_from_mock_records_alone():
    assert "NO VERDICT" in verdict([probe(mock=True, engine="mock")], sol=42_188.0)


def test_served_records_alone_produce_no_verdict():
    """The bare-engine baseline measures the in-process engine. Dividing a served rate by
    it would compare two different engines and render the result as an efficiency."""
    v = verdict([probe(engine="http", serving="http")], sol=42_188.0)
    assert "NO VERDICT" in v
    assert "not their denominator" in v


def test_served_rows_get_no_efficiency_column():
    report = build_report([probe(engine="http", serving="http"), baseline()])
    # The in-process row would print a ratio; the served one must print an em dash.
    assert "http (http)" in report
    assert "1.000 (defines the ceiling)" in report


def test_pipeline_not_bottleneck_verdict():
    v = verdict([probe(total_tokens_per_sec=41_460.0)], sol=42_188.0)
    assert "not the bottleneck" in v


def test_pipeline_is_bottleneck_verdict():
    v = verdict([probe(total_tokens_per_sec=20_000.0)], sol=42_188.0)
    assert "IS the bottleneck" in v


def test_inconclusive_verdict():
    v = verdict([probe(total_tokens_per_sec=36_000.0)], sol=42_188.0)
    assert "Inconclusive" in v


def test_mock_records_are_banner_flagged():
    assert "NOT A MEASUREMENT" in build_report([probe(mock=True, engine="mock")])


def test_engines_are_distinguishable_in_the_matrix():
    """Two engines at the same grammar and in-flight must not render as duplicate rows."""
    report = build_report([probe(engine="trtllm"),
                           probe(engine="http", serving="http"), baseline()])
    assert "trtllm (in-process)" in report
    assert "http (http)" in report


def test_valid_products_ranking_prefers_guided():
    """Unguided decoding produces tokens faster and fails validation more, so raw tok/s
    and valid products/sec rank configurations differently."""
    report = build_report([
        probe(grammar="off", products_per_sec=42.76, goodput=0.824,
              total_tokens_per_sec=44_000.0),
        probe(grammar="xgrammar", products_per_sec=38.31, goodput=1.0),
        baseline(),
    ])
    section = report.split("### Valid products/sec")[1]
    guided_pos = section.index("xgrammar")
    unguided_pos = section.index("off")
    assert guided_pos < unguided_pos, "guided config should rank first on valid products/s"
