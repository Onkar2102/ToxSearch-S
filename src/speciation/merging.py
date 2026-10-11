

from typing import List, Tuple

from .species import Individual, Species, SpeciesIdGenerator

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def merge_islands(
    sp1: Species,
    sp2: Species,
    current_generation: int,
    theta_sim: float = 0.2,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3,
    logger=None,
) -> Tuple[Species, List[Individual]]:
    """Merge two species onto the oldest ID (``min(sp1.id, sp2.id)``).

    ``w_genotype`` / ``w_phenotype`` are unused (kept for call-site compatibility).
    Returns ``(merged_species, outliers)``; outliers is always empty (radius
    cleanup happens in the orchestration layer).
    """
    del w_genotype, w_phenotype
    if logger is None:
        logger = get_logger("IslandMerging")

    seen = set()
    combined = []
    for m in sp1.members + sp2.members:
        if m.id not in seen:
            combined.append(m)
            seen.add(m.id)

    combined.sort(key=lambda x: x.fitness, reverse=True)

    if not combined:
        if not sp1.leader or not sp2.leader:
            logger.error(
                f"Cannot merge species {sp1.id} and {sp2.id}: "
                "both have no members and at least one has no leader"
            )
            raise ValueError(
                f"Cannot merge species {sp1.id} and {sp2.id}: "
                "insufficient members and leaders"
            )
        new_leader = max([sp1.leader, sp2.leader], key=lambda x: x.fitness)
    else:
        new_leader = combined[0]

    # Survivor keeps the oldest parent ID; absorbed ID is noted so it is not reused.
    survivor_id = min(sp1.id, sp2.id)
    older = sp1 if sp1.id <= sp2.id else sp2
    parent_ids = sorted([sp1.id, sp2.id])
    SpeciesIdGenerator.note_id(max(sp1.id, sp2.id))

    merged = Species(
        id=survivor_id,
        leader=new_leader,
        members=combined,
        radius=theta_sim,
        stagnation=0,
        max_fitness=new_leader.fitness,
        species_state="active",
        created_at=older.created_at,
        last_improvement=current_generation,
        cluster_origin="merge",
        parent_ids=parent_ids,
    )
    
    for m in combined:
        m.species_id = merged.id
    
    try:
        from .run_speciation import _get_state
        state = _get_state()
        genome_tracker = state.get("_genome_tracker")
        if genome_tracker:
            parent1_genome_ids = genome_tracker.get_all_genomes_by_species(sp1.id)
            parent2_genome_ids = genome_tracker.get_all_genomes_by_species(sp2.id)
            all_parent_genome_ids = set(parent1_genome_ids) | set(parent2_genome_ids)
            
            in_memory_ids = {str(m.id) for m in combined}
            all_genome_ids_to_update = all_parent_genome_ids | in_memory_ids
            
            updates = {str(gid): merged.id for gid in all_genome_ids_to_update}
            
            if len(all_parent_genome_ids) > len(in_memory_ids):
                logger.debug(
                    f"Merge {sp1.id}+{sp2.id}->{merged.id}: Updating {len(all_parent_genome_ids)} genomes from tracker "
                    f"(in-memory had {len(in_memory_ids)} members, {len(all_parent_genome_ids) - len(in_memory_ids)} additional from tracker)"
                )
            
            result = genome_tracker.batch_update(updates, current_generation, f"merge_{sp1.id}_{sp2.id}_to_{merged.id}")
            if result["failed"] > 0:
                logger.warning(f"Genome tracker batch update had {result['failed']} failures during merge")
            elif len(all_parent_genome_ids) > len(in_memory_ids):
                logger.info(
                    f"Merge {sp1.id}+{sp2.id}->{merged.id}: Successfully updated {result['succeeded']} genomes "
                    f"({len(all_parent_genome_ids) - len(in_memory_ids)} additional from tracker beyond in-memory members)"
                )
            
            reassigned_from_archive = result.get("reassigned_from_archive", [])
            if reassigned_from_archive:
                events_tracker = state.get("_events_tracker")
                if events_tracker:
                    for genome_id, old_species_id, new_species_id in reassigned_from_archive:
                        events_tracker.log(
                            genome_id, "reassigned_from_archive",
                            {
                                "from_species_id": old_species_id,
                                "to_species_id": new_species_id,
                                "reason": "species_merge",
                                "merged_species": [sp1.id, sp2.id],
                                "result_species": merged.id,
                                "note": "Genome was previously archived but is now in merged species"
                            }
                        )
                    logger.debug(f"Logged {len(reassigned_from_archive)} reassignment events for genomes reactivated from archive during merge")
    except Exception as e:
        logger.debug(f"Could not update genome tracker during merge: {e}")
    
    absorbed = max(sp1.id, sp2.id)
    logger.info(
        f"Merged species {sp1.id} + {sp2.id} -> survivor {merged.id} "
        f"(absorbed {absorbed}, {merged.size} members, no filtering applied - enforced later)"
    )
    return merged, []
