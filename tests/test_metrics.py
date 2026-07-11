"""Hand-computable cases for every metric — no fixtures, no randomness."""

import math

import pytest

from rag_retriever_bench import metrics


class TestDedupe:
    def test_keeps_first_occurrence(self):
        assert metrics.dedupe(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    def test_empty(self):
        assert metrics.dedupe([]) == []

    def test_no_duplicates_unchanged(self):
        assert metrics.dedupe(["a", "b", "c"]) == ["a", "b", "c"]


class TestRecallAtK:
    def test_all_positives_found(self):
        assert metrics.recall_at_k(["a", "b", "c"], {"a", "b"}, k=10) == 1.0

    def test_half_found(self):
        assert metrics.recall_at_k(["a", "x", "y"], {"a", "b"}, k=10) == 0.5

    def test_cutoff_at_k(self):
        # "b" sits at rank 3, outside k=2
        assert metrics.recall_at_k(["a", "x", "b"], {"a", "b"}, k=2) == 0.5

    def test_uncapped_denominator(self):
        # Standard recall: |positives| > k means recall@k cannot reach 1.0.
        # 12 positives, top-10 all hits -> 10/12.
        positives = {f"p{i}" for i in range(12)}
        retrieved = [f"p{i}" for i in range(10)]
        assert metrics.recall_at_k(retrieved, positives, k=10) == pytest.approx(10 / 12)

    def test_empty_positives(self):
        assert metrics.recall_at_k(["a"], set(), k=10) == 0.0


class TestHitAtK:
    def test_hit(self):
        assert metrics.hit_at_k(["x", "a"], {"a"}, k=10) == 1.0

    def test_miss(self):
        assert metrics.hit_at_k(["x", "y"], {"a"}, k=10) == 0.0

    def test_hit_outside_k(self):
        assert metrics.hit_at_k(["x", "y", "a"], {"a"}, k=2) == 0.0


class TestMrrAtK:
    def test_first_rank(self):
        assert metrics.mrr_at_k(["a", "x"], {"a"}, k=10) == 1.0

    def test_third_rank(self):
        assert metrics.mrr_at_k(["x", "y", "a"], {"a"}, k=10) == pytest.approx(1 / 3)

    def test_only_first_hit_counts(self):
        # "a" at rank 2, "b" at rank 4 — MRR uses the first hit only
        assert metrics.mrr_at_k(["x", "a", "y", "b"], {"a", "b"}, k=10) == 0.5

    def test_no_hit(self):
        assert metrics.mrr_at_k(["x", "y"], {"a"}, k=10) == 0.0


class TestNdcgAtK:
    def test_perfect_ranking(self):
        assert metrics.ndcg_at_k(["a", "b"], {"a", "b"}, k=10) == pytest.approx(1.0)

    def test_no_hits(self):
        assert metrics.ndcg_at_k(["x", "y"], {"a"}, k=10) == 0.0

    def test_single_hit_at_rank_2(self):
        # DCG = 1/log2(3), IDCG = 1/log2(2) = 1
        expected = 1 / math.log2(3)
        assert metrics.ndcg_at_k(["x", "a"], {"a"}, k=10) == pytest.approx(expected)

    def test_ideal_hits_capped_by_k(self):
        # 3 positives but k=2: IDCG uses only 2 ideal ranks, so a perfect
        # top-2 scores 1.0 even though one positive is unreachable.
        assert metrics.ndcg_at_k(["a", "b"], {"a", "b", "c"}, k=2) == pytest.approx(1.0)

    def test_empty_positives(self):
        assert metrics.ndcg_at_k(["a"], set(), k=10) == 0.0


class TestPercentile:
    def test_empty(self):
        assert metrics.percentile([], 50) == 0.0

    def test_single_value(self):
        assert metrics.percentile([7.0], 99) == 7.0

    def test_median_odd(self):
        assert metrics.percentile([1.0, 2.0, 3.0], 50) == 2.0

    def test_median_interpolated(self):
        assert metrics.percentile([1.0, 2.0], 50) == 1.5

    def test_p0_and_p100(self):
        values = [1.0, 5.0, 9.0]
        assert metrics.percentile(values, 0) == 1.0
        assert metrics.percentile(values, 100) == 9.0

    def test_p95_interpolation(self):
        # idx = 9 * 0.95 = 8.55 -> 0.45*v[8] + 0.55*v[9]
        values = [float(i) for i in range(10)]
        assert metrics.percentile(values, 95) == pytest.approx(8.55)
