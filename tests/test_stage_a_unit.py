"""Unit tests for Stage A distance tables (no API / GGUF)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.stage_analysis import stage_a


def _toy_p0(n: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(n, 8))
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    rows = []
    for i in range(n):
        rows.append({
            "prompt": f"prompt {i}?",
            "response": f"response {i}",
            "emb": emb[i].tolist(),
            "scores_google": rng.random(8).tolist(),
            "scores_openai": rng.random(13).tolist(),
        })
    return pd.DataFrame(rows)


def test_calibration_lf_vs_dbscan_radii():
    mats = stage_a.distance_matrices(_toy_p0(), nli=False)
    calib = stage_a.calibration_table(mats, reference="d_E", reference_radius=0.25)
    assert set(calib["distance"]) >= {"d_E", "d_EP[google]", "d_EP[openai]"}
    # Matching the same neighbour fraction: LF strict radius is >= non-strict eps
    # up to floating nextafter; both must be finite and in [0, 1].
    for _, row in calib.iterrows():
        assert 0.0 <= row["theta_sim"] <= 1.0
        assert 0.0 <= row["eps"] <= 1.0
        assert row["theta_sim"] + 1e-12 >= row["eps"]


def test_output_term_share_in_unit_interval():
    mats = stage_a.distance_matrices(_toy_p0(), nli=False)
    for ev in ("google", "openai"):
        share = stage_a.output_term_share(mats, ev)
        assert 0.0 <= share["mean"] <= 1.0
        assert 0.0 <= share["median"] <= 1.0


def test_spearman_and_knn_square():
    mats = stage_a.distance_matrices(_toy_p0(), nli=False)
    sp = stage_a.spearman_table(mats)
    kn = stage_a.knn_table(mats, k=3)
    assert sp.shape[0] == sp.shape[1] == len(mats)
    assert kn.shape[0] == kn.shape[1]
    assert np.isclose(sp.loc["d_E", "d_E"], 1.0)
    assert np.isclose(kn.loc["d_E", "d_E"], 1.0)


def test_p0_roundtrip_json(tmp_path):
    df = _toy_p0(4)
    path = tmp_path / "p0.json"
    df.to_json(path, orient="records", indent=2)
    loaded = stage_a.load_p0(str(path))
    assert len(loaded) == 4
    assert list(loaded.columns) == list(df.columns)
