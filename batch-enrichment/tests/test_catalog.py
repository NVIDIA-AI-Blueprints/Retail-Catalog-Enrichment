"""The catalog generator.

Determinism is the load-bearing property: the benchmark compares configurations, so the
catalog must not be a variable between runs.
"""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from catalog_bench.catalog import GROUND_TRUTH_SCHEMA, INPUT_SCHEMA, generate_rows, write_shards

N = 200


@pytest.fixture(scope="module")
def generated():
    return generate_rows(N, seed=42)


def test_same_seed_produces_identical_bytes(tmp_path):
    paths = []
    for name in ("a", "b"):
        rows, _ = generate_rows(N, seed=42)
        paths.append(write_shards(rows, tmp_path / name, shards=1)[0])
    assert paths[0].read_bytes() == paths[1].read_bytes()


def test_different_seed_produces_different_data(tmp_path):
    a, _ = generate_rows(N, seed=42)
    b, _ = generate_rows(N, seed=43)
    pa_ = write_shards(a, tmp_path / "a", shards=1)[0]
    pb = write_shards(b, tmp_path / "b", shards=1)[0]
    assert pa_.read_bytes() != pb.read_bytes()


def test_output_matches_the_input_contract(generated, tmp_path):
    rows, _ = generated
    path = write_shards(rows, tmp_path / "c", shards=1)[0]
    assert pq.read_table(path).schema.equals(INPUT_SCHEMA)


def test_ground_truth_is_a_separate_artifact(generated):
    """The input contract is what a customer hands the pipeline; it must not acquire
    columns that exist only because our input happens to be synthetic."""
    catalog, truth = generated
    assert set(catalog) == {"product_id", "title", "description", "metadata"}
    assert set(truth) == set(GROUND_TRUTH_SCHEMA.names)


def test_product_ids_are_unique_and_stable(generated):
    rows, truth = generated
    assert len(set(rows["product_id"])) == N
    assert rows["product_id"] == truth["product_id"]


def test_unstated_attributes_are_labelled_null(generated):
    """A value the generator chose is only a valid label if it survived into the emitted
    text. Scoring a model against information absent from its input measures nothing."""
    rows, truth = generated
    for pid, desc, meta, material in zip(rows["product_id"], rows["description"],
                                         rows["metadata"], truth["material"],
                                         strict=True):
        if material:
            assert material.lower() in (desc + meta).lower(), (
                f"{pid}: labelled material {material!r} is not in the listing text")


def test_shards_partition_the_catalog(generated, tmp_path):
    rows, _ = generated
    paths = write_shards(rows, tmp_path / "shards", shards=4)
    assert len(paths) == 4
    total = sum(pq.read_table(p).num_rows for p in paths)
    assert total == N


def test_catalog_is_messy_enough_to_be_a_workload(generated):
    """A clean catalog understates both token count and extraction difficulty."""
    rows, _ = generated
    assert any(not d for d in rows["description"]), "no empty descriptions"
    lengths = sorted(len(t) + len(d)
                     for t, d in zip(rows["title"], rows["description"], strict=True))
    p50, p95 = lengths[len(lengths) // 2], lengths[int(len(lengths) * 0.95)]
    assert p95 > 2 * p50, f"length distribution is not long-tailed (p50={p50}, p95={p95})"
