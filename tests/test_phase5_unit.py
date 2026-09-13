
import json
import os
import sys
import tempfile
import shutil
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


                                                                           

class TestResolveApiKeys:
    """HybridModerationEvaluator._resolve_api_keys priority tests."""

    def test_explicit_list_takes_priority(self):
        from gne.evaluator import HybridModerationEvaluator
        keys = HybridModerationEvaluator._resolve_api_keys(["k1", "k2"])
        assert keys == ["k1", "k2"]

    def test_comma_separated_env(self, monkeypatch):
        monkeypatch.setenv("PERSPECTIVE_API_KEYS", "a,b,c")
        monkeypatch.delenv("PERSPECTIVE_API_KEY", raising=False)
        from gne.evaluator import HybridModerationEvaluator
        keys = HybridModerationEvaluator._resolve_api_keys(None)
        assert keys == ["a", "b", "c"]

    def test_indexed_env_vars(self, monkeypatch):
        monkeypatch.delenv("PERSPECTIVE_API_KEYS", raising=False)
        monkeypatch.setenv("PERSPECTIVE_API_KEY_0", "x")
        monkeypatch.setenv("PERSPECTIVE_API_KEY_1", "y")
        monkeypatch.delenv("PERSPECTIVE_API_KEY_2", raising=False)
        monkeypatch.delenv("PERSPECTIVE_API_KEY", raising=False)
        from gne.evaluator import HybridModerationEvaluator
        keys = HybridModerationEvaluator._resolve_api_keys(None)
        assert keys == ["x", "y"]

    def test_single_env_fallback(self, monkeypatch):
        monkeypatch.delenv("PERSPECTIVE_API_KEYS", raising=False)
        monkeypatch.delenv("PERSPECTIVE_API_KEY_0", raising=False)
        monkeypatch.setenv("PERSPECTIVE_API_KEY", "solo")
        from gne.evaluator import HybridModerationEvaluator
        keys = HybridModerationEvaluator._resolve_api_keys(None)
        assert keys == ["solo"]

    def test_empty_when_nothing_set(self, monkeypatch):
        monkeypatch.delenv("PERSPECTIVE_API_KEYS", raising=False)
        monkeypatch.delenv("PERSPECTIVE_API_KEY_0", raising=False)
        monkeypatch.delenv("PERSPECTIVE_API_KEY", raising=False)
        from gne.evaluator import HybridModerationEvaluator
        keys = HybridModerationEvaluator._resolve_api_keys(None)
        assert keys == []

    def test_whitespace_stripped(self):
        from gne.evaluator import HybridModerationEvaluator
        keys = HybridModerationEvaluator._resolve_api_keys(["  k1 ", " k2  "])
        assert keys == ["k1", "k2"]

    def test_empty_strings_filtered(self):
        from gne.evaluator import HybridModerationEvaluator
        keys = HybridModerationEvaluator._resolve_api_keys(["k1", "", "  ", "k2"])
        assert keys == ["k1", "k2"]


                                                                           

class TestSelectKey:
    """select_key switches the active API key and rebuilds the client."""

    @patch("gne.evaluator.HybridModerationEvaluator._initialize_clients")
    def test_select_different_key_rebuilds(self, mock_init, monkeypatch):
        monkeypatch.setenv("PERSPECTIVE_API_KEY", "dummy")
        from gne.evaluator import HybridModerationEvaluator
        ev = HybridModerationEvaluator.__new__(HybridModerationEvaluator)
        ev._api_keys = ["k0", "k1", "k2"]
        ev._active_key_index = 0
        ev.google_available = True
        ev.logger = MagicMock()

        mock_init.reset_mock()
        ev.select_key(2)
        assert ev._active_key_index == 2
        mock_init.assert_called_once()

    @patch("gne.evaluator.HybridModerationEvaluator._initialize_clients")
    def test_select_same_key_no_rebuild(self, mock_init, monkeypatch):
        monkeypatch.setenv("PERSPECTIVE_API_KEY", "dummy")
        from gne.evaluator import HybridModerationEvaluator
        ev = HybridModerationEvaluator.__new__(HybridModerationEvaluator)
        ev._api_keys = ["k0", "k1"]
        ev._active_key_index = 0
        ev.google_available = True
        ev.logger = MagicMock()

        mock_init.reset_mock()
        ev.select_key(0)
        mock_init.assert_not_called()

    @patch("gne.evaluator.HybridModerationEvaluator._initialize_clients")
    def test_index_wraps_around(self, mock_init, monkeypatch):
        monkeypatch.setenv("PERSPECTIVE_API_KEY", "dummy")
        from gne.evaluator import HybridModerationEvaluator
        ev = HybridModerationEvaluator.__new__(HybridModerationEvaluator)
        ev._api_keys = ["k0", "k1"]
        ev._active_key_index = 0
        ev.google_available = True
        ev.logger = MagicMock()

        mock_init.reset_mock()
        ev.select_key(3)              
        assert ev._active_key_index == 1
        mock_init.assert_called_once()


                                                                           

class TestMergeErrorFiltering:
    """_merge_and_speciate skips genomes with status=error."""

    def _setup_outputs(self):
        tmpdir = tempfile.mkdtemp(prefix="toxsearch_p5_merge_")
        outputs = Path(tmpdir)
        for fname in ("elites.json", "reserves.json", "archive.json", "temp.json"):
            with open(outputs / fname, "w") as f:
                json.dump([], f)
        return tmpdir, outputs

    def test_error_genomes_skipped(self):
        tmpdir, outputs = self._setup_outputs()
        try:
            from parallel.master_worker import _merge_and_speciate

            buffers = defaultdict(list)
            buffers[1] = [
                {"prompt": "good prompt 1", "status": "complete",
                 "moderation_result": {"google": {"scores": {"toxicity": 0.5}}}},
                {"prompt": "error prompt", "status": "error", "error": "API failed"},
                {"prompt": "good prompt 2", "status": "complete",
                 "moderation_result": {"google": {"scores": {"toxicity": 0.6}}}},
            ]

            logger = MagicMock()
            accepted, discarded, new_id, _, accepted_genomes = _merge_and_speciate(
                buffers, K=10, outputs_path=outputs, generation_id=0,
                next_genome_id=1, north_star_metric="toxicity",
                speciation_config=None, log_file=None, logger=logger,
            )

            assert accepted == 2
            assert discarded == 1

            with open(outputs / "reserves.json") as f:
                reserves = json.load(f)
            prompts = [g["prompt"] for g in reserves]
            assert "error prompt" not in prompts
            assert "good prompt 1" in prompts
            assert "good prompt 2" in prompts
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_all_error_genomes_yields_zero_accepted(self):
        tmpdir, outputs = self._setup_outputs()
        try:
            from parallel.master_worker import _merge_and_speciate

            buffers = defaultdict(list)
            buffers[1] = [
                {"prompt": "err1", "status": "error", "error": "fail"},
                {"prompt": "err2", "status": "error", "error": "fail"},
            ]

            logger = MagicMock()
            accepted, discarded, _, _, _ = _merge_and_speciate(
                buffers, K=10, outputs_path=outputs, generation_id=0,
                next_genome_id=1, north_star_metric="toxicity",
                speciation_config=None, log_file=None, logger=logger,
            )

            assert accepted == 0
            assert discarded == 2
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_mixed_error_and_dedup(self):
        tmpdir, outputs = self._setup_outputs()
        try:
            from parallel.master_worker import _merge_and_speciate

            buffers = defaultdict(list)
            buffers[1] = [
                {"prompt": "prompt A", "status": "complete"},
                {"prompt": "prompt A", "status": "complete"},         
                {"prompt": "prompt B", "status": "error", "error": "boom"},
                {"prompt": "prompt C", "status": "complete"},
            ]

            logger = MagicMock()
            accepted, discarded, _, _, _ = _merge_and_speciate(
                buffers, K=10, outputs_path=outputs, generation_id=0,
                next_genome_id=1, north_star_metric="toxicity",
                speciation_config=None, log_file=None, logger=logger,
            )

            assert accepted == 2                       
            assert discarded == 2                 
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


                                                                           

class TestLoadPerspectiveApiKeys:
    """Test the module-level key loader used by run()."""

    def test_comma_separated(self, monkeypatch):
        monkeypatch.setenv("PERSPECTIVE_API_KEYS", "a,b")
        monkeypatch.delenv("PERSPECTIVE_API_KEY", raising=False)
        from parallel.master_worker import _load_perspective_api_keys
        assert _load_perspective_api_keys() == ["a", "b"]

    def test_single_fallback(self, monkeypatch):
        monkeypatch.delenv("PERSPECTIVE_API_KEYS", raising=False)
        monkeypatch.delenv("PERSPECTIVE_API_KEY_0", raising=False)
        monkeypatch.setenv("PERSPECTIVE_API_KEY", "one")
        from parallel.master_worker import _load_perspective_api_keys
        assert _load_perspective_api_keys() == ["one"]

    def test_empty(self, monkeypatch):
        monkeypatch.delenv("PERSPECTIVE_API_KEYS", raising=False)
        monkeypatch.delenv("PERSPECTIVE_API_KEY_0", raising=False)
        monkeypatch.delenv("PERSPECTIVE_API_KEY", raising=False)
        from parallel.master_worker import _load_perspective_api_keys
        assert _load_perspective_api_keys() == []
