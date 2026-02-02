"""
merging.py

Island merging logic for speciation.
"""

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
    """
    Merge two islands into a single species.
    
    Merging combines two similar species into one:
    - Members: All members from both species (deduplicated by ID) - NO radius/capacity filtering
    - Leader: Highest-fitness individual from ALL combined members
    - Radius: Constant theta_sim (same for all species)
    - Stagnation: Reset to 0 (fresh start for merged species)
    - Origin: "merge" with parent_ids = [sp1.id, sp2.id]
    
    NOTE: This function does NOT enforce radius or capacity. All combined members are kept.
    Radius enforcement is done in Phase 3 (after merging) and capacity enforcement in Phase 4 of run_speciation.py.
    
    Note: Creates a NEW species ID (not reusing sp1.id or sp2.id) to avoid confusion.
    
    Args:
        sp1: First species to merge
        sp2: Second species to merge
        current_generation: Current generation number
        theta_sim: Constant radius for the merged species (default: 0.2, matches config.py)
        logger: Optional logger instance
    
    Returns:
        Tuple of (merged Species, empty list)
        - merged: New merged Species with cluster_origin="merge" and parent_ids=[sp1.id, sp2.id]
        - outliers: Empty list (no filtering during merge, will be done in Phase 4)
    """
    if logger is None:
        logger = get_logger("IslandMerging")
    
    # Combine members, deduplicating by ID
    seen = set()
    combined = []
    for m in sp1.members + sp2.members:
        if m.id not in seen:
            combined.append(m)
            seen.add(m.id)
    
    # Sort ALL members by fitness score (descending order)
    combined.sort(key=lambda x: x.fitness, reverse=True)
    
    # Select genome with highest fitness score as leader
    # This leader will NOT be changed until a new genome with higher fitness score is added to the species
    if not combined:
        # Fallback: use highest fitness leader from either species (both should exist)
        if not sp1.leader or not sp2.leader:
            logger.error(f"Cannot merge species {sp1.id} and {sp2.id}: both have no members and at least one has no leader")
            raise ValueError(f"Cannot merge species {sp1.id} and {sp2.id}: insufficient members and leaders")
        new_leader = max([sp1.leader, sp2.leader], key=lambda x: x.fitness)
    else:
        new_leader = combined[0]  # Highest fitness (first after sorting)
    
    # Note: Leader selection happens BEFORE radius enforcement (Phase 3 Step 7)
    # Note: Leader will only change if a new genome with higher fitness is added (handled in Phase 1/Phase 3)
    # Note: Capacity enforcement happens in Phase 4 (does NOT change leader for merged species)
    
    # Create merged species with ALL members (NO radius/capacity filtering)
    merged = Species(
        id=generate_species_id(),  # New ID for clarity
        leader=new_leader,
        members=combined,  # ALL members, NO filtering
        radius=theta_sim,  # Constant radius for all species
        stagnation=0,
        max_fitness=new_leader.fitness,
        species_state="active",
        created_at=current_generation,
        last_improvement=current_generation,
        cluster_origin="merge",  # Created via merge
        parent_ids=[sp1.id, sp2.id]  # Both parent IDs
    )
    
    # Update species assignments for all members
    for m in combined:
        m.species_id = merged.id
    
    # Update genome tracker if available
    try:
        from .run_speciation import _get_state
        state = _get_state()
        genome_tracker = state.get("_genome_tracker")
        if genome_tracker:
            # CRITICAL: Update ALL genomes from tracker that belong to parent species, not just in-memory members
            # In-memory members might not include all genomes (e.g., from previous generations, archived genomes)
            # Get all genome IDs from tracker for both parent species
            parent1_genome_ids = genome_tracker.get_all_genomes_by_species(sp1.id)
            parent2_genome_ids = genome_tracker.get_all_genomes_by_species(sp2.id)
            all_parent_genome_ids = set(parent1_genome_ids) | set(parent2_genome_ids)
            
            # Also include in-memory members (in case they're not in tracker yet)
            in_memory_ids = {str(m.id) for m in combined}
            all_genome_ids_to_update = all_parent_genome_ids | in_memory_ids
            
            # Prepare batch update: ALL genomes from both parents -> new merged species_id
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
            
            # Log reassignment events (if any): in current design archive genomes are not moved back; this supports possible future use
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
    
    logger.info(f"Merged species {sp1.id} + {sp2.id} -> {merged.id} ({merged.size} members, no filtering applied - will be enforced in Phase 4)")
    return merged, []  # Return empty outliers list (no filtering during merge)


def process_merges(
    species: Dict[int, Species],
    theta_merge: float = 0.1,
    theta_sim: float = 0.2,
    min_stability_gens: int = 1,
    current_gen: int = 0,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3,
    historical_species: Optional[Dict[int, Species]] = None,
    logger=None
) -> Tuple[Dict[int, Species], List[Dict], List[Individual], Dict[int, Species]]:
    """
    Process all species merges for a generation.
    
    Merging combines similar species to prevent excessive fragmentation.
    Two species merge if:
    1. Leader distance < theta_merge (very similar)
    2. Both species are stable (existed for min_stability_gens; default 1 = can merge if created in last or prior generation)
    
    Merged species:
    - Combines all members (deduplicated)
    - Keeps highest-fitness leader from all combined members
    - Uses constant theta_sim radius
    - Has cluster_origin="merge" and parent_ids=[id1, id2]
    - NO radius/capacity enforcement (deferred to Phase 4 in run_speciation.py)
    
    Frozen species can merge with active or other frozen species.
    When species merge, BOTH parent species become extinct (moved to historical_species).
    The merged species is a new species with a new ID.
    
    Process iteratively until no more merge candidates are found.
    All eligible merges happen in a single generation.
    
    Args:
        species: Dict of active and frozen species (modified in-place)
        theta_merge: Merge distance threshold (must be < theta_sim)
        theta_sim: Constant radius for merged species
        min_stability_gens: Minimum age (generations) for species to be mergeable (default 1: can merge if created in last or prior generation)
        current_gen: Current generation number
        historical_species: Optional dict for storing extinct parent species
        logger: Optional logger instance
    
    Returns:
        Tuple of (updated_species, merge_events, outliers, extinct_parents)
        - updated_species: Dict of species after merging (parents removed, merged species added)
        - merge_events: List of merge event dictionaries
        - outliers: Empty list (no filtering during merge, radius enforcement deferred to Phase 4)
        - extinct_parents: Dict of parent species that became extinct via merging (to be moved to historical_species)
    """
    if logger is None:
        logger = get_logger("IslandMerging")
    
    events = []
    all_outliers = []  # Collect all outliers from merges
    extinct_parents = {}  # Track parent species that became extinct via merging
    
    # Combine active and frozen species for merge candidate search
    # Frozen species can merge with active or other frozen species
    # Both active and frozen species are "alive" - only difference is parent selection preference
    # Note: Frozen species are now in the active species dict (not historical_species)
    # Filter out species without leaders or with incubator state (should not merge)
    all_species_for_merging = {}
    for sid, sp in species.items():
        # Only include species that have a leader and are not incubator
        if sp.leader is not None and sp.species_state != "incubator":
            all_species_for_merging[sid] = sp
        elif sp.species_state == "incubator":
            logger.debug(f"Skipping species {sid} from merge candidates: incubator state (no leader)")
        elif sp.leader is None:
            logger.warning(f"Skipping species {sid} from merge candidates: no leader (state={sp.species_state})")
    
    # Continue merging until no more candidates are found
    while True:
        # Find merge candidates: pairs with leader distance < theta_merge and both stable
        merge_pairs = []
        species_list = list(all_species_for_merging.items())
        for i, (id1, sp1) in enumerate(species_list):
            for j, (id2, sp2) in enumerate(species_list[i + 1:], start=i + 1):
                # Check if leaders exist and have embeddings before accessing
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
        # Get species from appropriate dict (active or historical)
        sp1 = all_species_for_merging.get(id1)
        sp2 = all_species_for_merging.get(id2)
        
        if not sp1 or not sp2:
            logger.warning(f"Skipping merge {id1}+{id2}: one or both species not found in all_species_for_merging")
            continue
        
        # Validate that both species have leaders before merging
        if not sp1.leader or not sp2.leader:
            logger.warning(f"Skipping merge {id1}+{id2}: one or both species have no leader (sp1.leader={sp1.leader is not None}, sp2.leader={sp2.leader is not None})")
            continue
        
        merged, outliers = merge_islands(sp1, sp2, current_gen, theta_sim, w_genotype, w_phenotype, logger)
        
        # Remove old species from active dict
        species.pop(id1, None)
        species.pop(id2, None)
        # Remove from all_species_for_merging too
        all_species_for_merging.pop(id1, None)
        all_species_for_merging.pop(id2, None)
        
        # Add merged species (always active after merge)
        species[merged.id] = merged
        all_species_for_merging[merged.id] = merged
        
        # Mark both parent species as extinct (they will be moved to historical_species by caller)
        sp1.species_state = "extinct"
        sp2.species_state = "extinct"
        
        extinct_parents[id1] = sp1
        extinct_parents[id2] = sp2
        logger.info(f"Parent species {id1} and {id2} became extinct via merge -> new species {merged.id}")
        
        # Collect outliers for caller to handle
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
