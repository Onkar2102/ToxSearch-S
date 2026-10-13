"""NLI cost counters used for the d_N wall-clock / call-count outcome."""

from speciation.distance_metrics import nli


def test_nli_stats_count_passes_and_cache_hits():
    nli.set_nli_scorer(lambda a, b: (0.2, 0.5, 0.3))
    try:
        before = nli.get_nli_stats()
        nli.nli_distance("a", "b")
        nli.nli_distance("a", "b")
        nli.nli_distance("c", "c")
        after = nli.get_nli_stats()
    finally:
        nli.set_nli_scorer(None)

    assert after["distance_calls"] - before["distance_calls"] == 3
    assert after["model_passes"] - before["model_passes"] == 2
    assert after["cache_hits"] - before["cache_hits"] == 2
