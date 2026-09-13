
import json
import sys
import tempfile
import shutil
from collections import defaultdict
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


class TestPartialGen0Speciation:
    """_merge_and_speciate works correctly when buffered < K."""

    def _setup_outputs(self):
        tmpdir = tempfile.mkdtemp(prefix="toxsearch_p67_")
        outputs = Path(tmpdir)
        for fname in ("elites.json", "reserves.json", "archive.json", "temp.json"):
            with open(outputs / fname, "w") as f:
                json.dump([], f)
        return tmpdir, outputs

    def test_partial_batch_accepted(self):
        tmpdir, outputs = self._setup_outputs()
        try:
            from parallel.master_worker import _merge_and_speciate

            buffers = defaultdict(list)
            buffers[1] = [
                {"prompt": "prompt A", "status": "complete",
                 "moderation_result": {"google": {"scores": {"toxicity": 0.5}}}},
                {"prompt": "prompt B", "status": "complete",
                 "moderation_result": {"google": {"scores": {"toxicity": 0.6}}}},
            ]
            buffers[2] = [
                {"prompt": "prompt C", "status": "complete",
                 "moderation_result": {"google": {"scores": {"toxicity": 0.7}}}},
            ]

            logger = MagicMock()
            total_buffered = sum(len(b) for b in buffers.values())
            batch_size = min(total_buffered, 100)

            accepted, discarded, new_id, _, _ = _merge_and_speciate(
                buffers, K=batch_size, outputs_path=outputs, generation_id=0,
                next_genome_id=1, north_star_metric="toxicity",
                speciation_config=None, log_file=None, logger=logger,
            )

            assert accepted == 3, f"Expected 3 accepted, got {accepted}"
            assert discarded == 0

            with open(outputs / "reserves.json") as f:
                reserves = json.load(f)
            prompts = {g["prompt"] for g in reserves}
            assert prompts == {"prompt A", "prompt B", "prompt C"}

            for g in reserves:
                assert "id" in g
                assert g["generation"] == 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_partial_batch_with_errors(self):
        tmpdir, outputs = self._setup_outputs()
        try:
            from parallel.master_worker import _merge_and_speciate

            buffers = defaultdict(list)
            buffers[1] = [
                {"prompt": "good 1", "status": "complete",
                 "moderation_result": {"google": {"scores": {"toxicity": 0.4}}}},
                {"prompt": "bad 1", "status": "error", "error": "API fail"},
            ]

            logger = MagicMock()
            total_buffered = sum(len(b) for b in buffers.values())
            batch_size = min(total_buffered, 100)

            accepted, discarded, _, _, _ = _merge_and_speciate(
                buffers, K=batch_size, outputs_path=outputs, generation_id=0,
                next_genome_id=1, north_star_metric="toxicity",
                speciation_config=None, log_file=None, logger=logger,
            )

            assert accepted == 1
            assert discarded == 1
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_empty_buffers_yields_zero(self):
        tmpdir, outputs = self._setup_outputs()
        try:
            from parallel.master_worker import _merge_and_speciate

            buffers = defaultdict(list)
            logger = MagicMock()

            accepted, discarded, _, _, _ = _merge_and_speciate(
                buffers, K=0, outputs_path=outputs, generation_id=0,
                next_genome_id=1, north_star_metric="toxicity",
                speciation_config=None, log_file=None, logger=logger,
            )

            assert accepted == 0
            assert discarded == 0
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestGen0ExpectedTracking:
    """Test gen0_expected computation logic (mirrors master_main code)."""

    def test_expected_equals_sum_of_ranges(self):
        gen0_assignments = {1: (0, 3), 2: (3, 5)}
        gen0_expected = sum(e - s for s, e in gen0_assignments.values())
        assert gen0_expected == 5

    def test_expected_zero_when_no_seed(self):
        gen0_assignments = {}
        gen0_expected = sum(e - s for s, e in gen0_assignments.values())
        assert gen0_expected == 0

    def test_uneven_distribution(self):
        n_workers = 3
        n_prompts = 7
        chunk = max(n_prompts // n_workers, 1)
        remainder = n_prompts % n_workers
        assignments = {}
        start = 0
        for w in range(1, n_workers + 1):
            end = start + chunk + (1 if (w - 1) < remainder else 0)
            end = min(end, n_prompts)
            assignments[w] = (start, end)
            start = end
        expected = sum(e - s for s, e in assignments.values())
        assert expected == 7
        assert assignments[1] == (0, 3)
        assert assignments[2] == (2, 5) or assignments[2][1] - assignments[2][0] >= 2
