

from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

from .species import Individual, Species
from .reserves import CLUSTER_0_ID

if TYPE_CHECKING:
    from .reserves import Cluster0

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def process_extinctions(
    species: Dict[int, Species],
    cluster0: "Cluster0",
    current_generation: int,
    species_stagnation: int = 20,
    min_size: int = 2,
    elites_path: Optional[str] = None,
    logger=None
) -> Tuple[Dict[int, Species], List[Dict], List[Dict], Dict[int, Species]]:
    
    if logger is None:
        logger = get_logger("Extinction")
    
    extinction_events = []
    moved_to_cluster0_events = []
    incubator_species = {}
    
    frozen_ids = []
    for sid, sp in species.items():
        if sp.stagnation >= species_stagnation and sp.species_state != "frozen":
            sp.species_state = "frozen"
            frozen_ids.append(sid)
            extinction_events.append({
                "generation": current_generation,
                "species_id": sid,
                "action": "frozen",
                "stagnation": sp.stagnation,
                "max_fitness": sp.max_fitness
            })
            logger.info(f"Frozen species {sid} (stagnation={sp.stagnation} >= {species_stagnation}) - excluded from parent selection")
    
    small_species_ids = []
    for sid, sp in species.items():
        if sp.species_state not in ["active", "frozen", "incubator"]:
            continue
        current_size = sp.size
        if sp.species_state == "incubator" or current_size < min_size:
            if sp.species_state != "incubator":
                small_species_ids.append(sid)
                logger.debug(f"Species {sid}: current size={current_size} (in-memory), min_size={min_size}, state={sp.species_state} -> will move to incubator")
            else:
                small_species_ids.append(sid)
                logger.debug(f"Species {sid}: already marked as incubator but not yet processed, will complete cleanup")
    
    for sid in small_species_ids:
        if sid not in species:
            continue
        
        if cluster0.size >= cluster0.max_capacity:
            logger.debug(f"Cluster 0 at capacity ({cluster0.max_capacity}), cannot move species {sid}")
            continue
        
        sp = species[sid]
        
        original_size = sp.size
        
        moved_count = 0
        moved_member_ids = []
        member_ids_set = {m.id for m in sp.members}
        
        if sp.leader and sp.leader.id not in member_ids_set:
            if cluster0.size < cluster0.max_capacity:
                cluster0.add(sp.leader, current_generation)
                moved_member_ids.append(sp.leader.id)
                moved_count += 1
        
        for member in sp.members:
            if cluster0.size >= cluster0.max_capacity:
                break
            cluster0.add(member, current_generation)
            moved_member_ids.append(member.id)
            moved_count += 1
        
        try:
            from .run_speciation import _get_state
            state = _get_state()
            genome_tracker = state.get("_genome_tracker")
            if genome_tracker and moved_member_ids:
                updates = {str(mid): 0 for mid in moved_member_ids}
                result = genome_tracker.batch_update(updates, current_generation, f"extinct_to_reserves_species_{sid}")
                if result["failed"] > 0:
                    logger.warning(f"Genome tracker batch update had {result['failed']} failures during extinction")
        except Exception as e:
            logger.debug(f"Could not update genome tracker during extinction: {e}")
        
        sp.species_state = "incubator"
        sp.members = []
        incubator_species[sid] = sp
        
        moved_to_cluster0_events.append({
            "generation": current_generation,
            "species_id": sid,
            "action": "moved_to_cluster0",
            "new_state": "incubator",
            "size": original_size,
            "moved_count": moved_count,
            "moved_member_ids": moved_member_ids
        })
        logger.info(f"Moved species {sid} ({moved_count} members) to cluster 0 - state=incubator")
    
    for sid in incubator_species:
        species.pop(sid, None)
    
    return species, extinction_events, moved_to_cluster0_events, incubator_species
