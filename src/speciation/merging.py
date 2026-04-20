

from typing import Dict, List, Tuple, Optional

from .species import Individual, Species, generate_species_id
from .distance import ensemble_distance

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def merge_islands(
    sp1: Species,
    sp2: Species,
    current_generation: int,
    theta_sim: float = 0.2,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3,
    logger=None
) -> Tuple[Species, List[Individual]]:
    
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
            logger.error(f"Cannot merge species {sp1.id} and {sp2.id}: both have no members and at least one has no leader")
            raise ValueError(f"Cannot merge species {sp1.id} and {sp2.id}: insufficient members and leaders")
        new_leader = max([sp1.leader, sp2.leader], key=lambda x: x.fitness)
    else:
        new_leader = combined[0]
    
    
    merged = Species(
        id=generate_species_id(),
        leader=new_leader,
        members=combined,
        radius=theta_sim,
        stagnation=0,
        max_fitness=new_leader.fitness,
        species_state="active",
        created_at=current_generation,
        last_improvement=current_generation,
        cluster_origin="merge",
        parent_ids=[sp1.id, sp2.id]
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
    
    logger.info(f"Merged species {sp1.id} + {sp2.id} -> {merged.id} ({merged.size} members, no filtering applied - enforced later)")
    return merged, []


def process_merges(
    species: Dict[int, Species],
    theta_merge: float = 0.1,
    theta_sim: float = 0.2,
    min_stability_gens: int = 5,
    current_gen: int = 0,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3,
    historical_species: Optional[Dict[int, Species]] = None,
    logger=None
) -> Tuple[Dict[int, Species], List[Dict], List[Individual], Dict[int, Species]]:
    
    if logger is None:
        logger = get_logger("IslandMerging")
    
    events = []
    all_outliers = []
    extinct_parents = {}
    
    all_species_for_merging = {}
    for sid, sp in species.items():
        if sp.leader is not None and sp.species_state != "incubator":
            all_species_for_merging[sid] = sp
        elif sp.species_state == "incubator":
            logger.debug(f"Skipping species {sid} from merge candidates: incubator state (no leader)")
        elif sp.leader is None:
            logger.warning(f"Skipping species {sid} from merge candidates: no leader (state={sp.species_state})")
    
    while True:
        merge_pairs = []
        species_list = list(all_species_for_merging.items())
        for i, (id1, sp1) in enumerate(species_list):
            for j, (id2, sp2) in enumerate(species_list[i + 1:], start=i + 1):
                if not sp1.leader or not sp2.leader:
                    logger.debug(f"Skipping merge check for {id1}+{id2}: one or both species have no leader")
                    continue
                if sp1.leader.embedding is None or sp2.leader.embedding is None:
                    logger.debug(f"Skipping merge check for {id1}+{id2}: one or both leaders have no embedding")
                    continue
                dist = ensemble_distance(
                    sp1.leader.embedding, sp2.leader.embedding,
                    sp1.leader.phenotype, sp2.leader.phenotype,
                    w_genotype, w_phenotype
                )
                if dist < theta_merge:
                    sp1_stable = (current_gen - sp1.created_at) >= min_stability_gens
                    sp2_stable = (current_gen - sp2.created_at) >= min_stability_gens
                    if sp1_stable and sp2_stable:
                        merge_pairs.append((id1, id2, sp1.species_state, sp2.species_state))
        
        if not merge_pairs:
            break
        
        id1, id2, state1, state2 = merge_pairs[0]
        sp1 = all_species_for_merging.get(id1)
        sp2 = all_species_for_merging.get(id2)
        
        if not sp1 or not sp2:
            logger.warning(f"Skipping merge {id1}+{id2}: one or both species not found in all_species_for_merging")
            continue
        
        if not sp1.leader or not sp2.leader:
            logger.warning(f"Skipping merge {id1}+{id2}: one or both species have no leader (sp1.leader={sp1.leader is not None}, sp2.leader={sp2.leader is not None})")
            continue
        
        merged, outliers = merge_islands(sp1, sp2, current_gen, theta_sim, w_genotype, w_phenotype, logger)
        
        species.pop(id1, None)
        species.pop(id2, None)
        all_species_for_merging.pop(id1, None)
        all_species_for_merging.pop(id2, None)
        
        species[merged.id] = merged
        all_species_for_merging[merged.id] = merged
        
        sp1.species_state = "extinct"
        sp2.species_state = "extinct"
        
        extinct_parents[id1] = sp1
        extinct_parents[id2] = sp2
        logger.info(f"Parent species {id1} and {id2} became extinct via merge -> new species {merged.id}")
        
        if outliers:
            all_outliers.extend(outliers)
            logger.debug(f"Merge {id1}+{id2}->{merged.id}: {len(outliers)} outliers need to be moved to cluster 0")
        
        events.append({
            "generation": current_gen,
            "merged": (id1, id2),
            "result_id": merged.id,
            "cluster_origin": "merge",
            "parent_ids": [id1, id2]
        })
        logger.info(f"Completed merge {len(events)}: {id1}+{id2}->{merged.id} (total merges so far: {len(events)})")
    
    logger.info(f"Merge process complete: {len(events)} merges performed in generation {current_gen}")
    if extinct_parents:
        logger.info(f"{len(extinct_parents)} parent species became extinct via merging: {sorted(extinct_parents.keys())}")
    return species, events, all_outliers, extinct_parents
