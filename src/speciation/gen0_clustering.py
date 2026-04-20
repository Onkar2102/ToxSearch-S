

import numpy as np
from typing import List, Dict, Tuple, Set

from .species import Individual, Species, generate_species_id
from .distance import ensemble_distance, ensemble_distances_batch
from .reserves import CLUSTER_0_ID


class Gen0Clustering:
    """Generation 0 clustering: build all (leader, followers) groups over the entire population, then promote to species only groups with |{leader} ∪ followers| >= min_island_size. Flow 2: Two-phase approach with no leader update and no radius enforcement. - Phase 1: Collect all potential leader groups - Phase 2: Form species if group size >= min_island_size (keep all members, no filtering)"""    @staticmethod
    def run(
        individuals: List[Individual],
        theta_sim: float,
        min_island_size: int,
        w_genotype: float,
        w_phenotype: float,
        current_generation: int,
        logger,
    ) -> Tuple[Dict[int, Species], Set[int]]:
        
        if not individuals:
            return {}, set()

        sorted_ind = sorted(individuals, key=lambda x: x.fitness, reverse=True)
        groups: List[Tuple[Individual, List[Individual]]] = [(sorted_ind[0], [])]

        for ind in sorted_ind[1:]:
            leader_embeddings = np.array([g[0].embedding for g in groups])
            leader_phenotypes = [g[0].phenotype for g in groups]
            if len(groups) > 1:
                dists = ensemble_distances_batch(
                    ind.embedding, leader_embeddings,
                    ind.phenotype, leader_phenotypes,
                    w_genotype, w_phenotype,
                )
                min_idx = int(np.argmin(dists))
                d = float(dists[min_idx])
            else:
                min_idx = 0
                d = ensemble_distance(
                    ind.embedding, groups[0][0].embedding,
                    ind.phenotype, groups[0][0].phenotype,
                    w_genotype, w_phenotype,
                )
            if d < theta_sim:
                groups[min_idx][1].append(ind)
            else:
                groups.append((ind, []))

        species: Dict[int, Species] = {}
        species_with_new_members: Set[int] = set()

        for pl_ind, followers in groups:
            all_members = [pl_ind] + followers

            if len(all_members) >= min_island_size:
                new_species_id = generate_species_id()
                new_species = Species(
                    id=new_species_id,
                    leader=pl_ind,
                    members=all_members,
                    radius=theta_sim,
                    created_at=current_generation,
                    last_improvement=current_generation,
                    cluster_origin="natural",
                    parent_ids=None,
                    leader_distance=0.0,
                )
                species[new_species_id] = new_species
                for m in all_members:
                    m.species_id = new_species_id
                species_with_new_members.add(new_species_id)
                logger.info(
                    "Gen 0 species %s: leader %s + %s followers (total=%s, min=%s)",
                    new_species_id, pl_ind.id, len(followers), len(all_members),
                    min_island_size,
                )
            else:
                for m in all_members:
                    m.species_id = CLUSTER_0_ID
                logger.debug(
                    "Gen 0: group with %s members < min_island_size %s → reserves",
                    len(all_members), min_island_size,
                )

        n_reserves = sum(1 for ind in individuals if getattr(ind, "species_id", None) == CLUSTER_0_ID)
        logger.info("Gen 0 two-phase: %s groups, %s species formed, %s in reserves", len(groups), len(species), n_reserves)
        return species, species_with_new_members
