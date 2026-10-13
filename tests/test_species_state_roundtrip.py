"""speciation_state.json must round-trip the leader objective vector (needed by d_EP)."""

import numpy as np

from speciation.clustering._io import load_species_state, write_species_state
from speciation.distance import pair_distance
from speciation.species import Individual, Species


def _species(sid: int, pheno) -> Species:
    emb = np.zeros(4, dtype=np.float32)
    emb[0] = 1.0
    leader = Individual(id=sid * 10, prompt=f"p{sid}", fitness=0.5, embedding=emb, phenotype=pheno, species_id=sid)
    return Species(id=sid, leader=leader, members=[leader], radius=0.25)


def test_leader_phenotype_roundtrip(tmp_path):
    pheno = np.linspace(0.0, 0.7, 8, dtype=np.float32)
    path = tmp_path / "speciation_state.json"
    write_species_state(path, {1: _species(1, pheno)}, generation=0)

    loaded = load_species_state(path, theta_sim=0.25)
    leader = loaded[1].leader
    assert leader.phenotype is not None
    np.testing.assert_allclose(leader.phenotype, pheno, atol=1e-6)

    d = pair_distance(
        "embedding+objective",
        embedding_a=leader.embedding,
        embedding_b=leader.embedding,
        objective_a=leader.phenotype,
        objective_b=np.zeros(8, dtype=np.float32),
    )
    assert 0.0 < d <= 1.0


def test_leader_without_phenotype_serializes_none(tmp_path):
    path = tmp_path / "speciation_state.json"
    write_species_state(path, {2: _species(2, None)}, generation=0)
    loaded = load_species_state(path, theta_sim=0.25)
    assert loaded[2].leader.phenotype is None
