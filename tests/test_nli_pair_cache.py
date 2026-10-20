"""On-disk NLI pair cache shared across clustering processes."""

from speciation.distance_metrics import nli


def test_nli_pair_cache_roundtrip_across_memory_clear(tmp_path):
    db = tmp_path / "nli_pairs.sqlite"
    nli.set_nli_pair_cache_path(str(db))
    nli.set_nli_scorer(lambda a, b: (0.2, 0.5, 0.3))
    try:
        nli.clear_nli_cache()
        before_first = nli.get_nli_stats()
        nli.nli_entail_distance("hello prompt", "world prompt")
        after_first = nli.get_nli_stats()
        nli.clear_nli_cache()
        before_second = nli.get_nli_stats()
        nli.nli_entail_distance("hello prompt", "world prompt")
        after_second = nli.get_nli_stats()
    finally:
        nli.set_nli_scorer(None)
        nli.close_nli_pair_cache()
        nli.clear_nli_cache()

    assert after_first["model_passes"] - before_first["model_passes"] >= 2
    assert after_second["model_passes"] == before_second["model_passes"]
    assert after_second["disk_hits"] - before_second["disk_hits"] >= 2


def test_nli_pair_cache_reopen_file(tmp_path):
    db = tmp_path / "nli_pairs.sqlite"
    nli.set_nli_pair_cache_path(str(db))
    nli.set_nli_scorer(lambda a, b: (0.8, 0.1, 0.1))
    try:
        nli.clear_nli_cache()
        d1 = nli.nli_entail_distance("p", "q")
        nli.close_nli_pair_cache()
        nli.clear_nli_cache()
        nli.set_nli_pair_cache_path(str(db))
        before = nli.get_nli_stats()
        d2 = nli.nli_entail_distance("p", "q")
        after = nli.get_nli_stats()
    finally:
        nli.set_nli_scorer(None)
        nli.close_nli_pair_cache()
        nli.clear_nli_cache()

    assert d1 == d2
    assert after["model_passes"] == before["model_passes"]
    assert after["disk_hits"] - before["disk_hits"] >= 2
