

from typing import Dict, List, Tuple, Optional

from .species import Individual, Species

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def process_extinctions(
    species: Dict[int, Species],
    current_generation: int,
    species_stagnation: int = 20,
    min_size: int = 2,
    elites_path: Optional[str] = None,
    logger=None
) -> Tuple[Dict[int, Species], List[Dict], List[Dict], Dict[int, Species]]:
    
    if logger is None:
        logger = get_logger("Extinction")
    
    extinction_events = []
    archived_from_extinction_events = []
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
            small_species_ids.append(sid)
    
    for sid in small_species_ids:
        if sid not in species:
            continue
        
        sp = species[sid]
        original_size = sp.size
        moved_member_ids = []
        members_to_archive = list(sp.members)
        if sp.leader and sp.leader.id not in {m.id for m in sp.members}:
            members_to_archive.append(sp.leader)
        
        try:
            from .run_speciation import _get_state, _archive_individuals
            state = _get_state()
            genome_tracker = state.get("_genome_tracker")
            if genome_tracker and members_to_archive:
                updates = {str(m.id): -1 for m in members_to_archive}
                result = genome_tracker.batch_update(
                    updates, current_generation, f"extinct_to_archive_species_{sid}"
                )
                if result["failed"] > 0:
                    logger.warning(f"Genome tracker batch update had {result['failed']} failures during extinction")
                moved_member_ids = [m.id for m in members_to_archive]
            if members_to_archive:
                _archive_individuals(members_to_archive, current_generation, f"extinct_species_{sid}")
        except Exception as e:
            logger.debug(f"Could not archive genomes during extinction: {e}")
        
        sp.species_state = "extinct"
        sp.members = []
        incubator_species[sid] = sp
        
        archived_from_extinction_events.append({
            "generation": current_generation,
            "species_id": sid,
            "action": "archived",
            "new_state": "extinct",
            "size": original_size,
            "moved_count": len(moved_member_ids),
            "moved_member_ids": moved_member_ids
        })
        logger.info(f"Archived species {sid} ({len(moved_member_ids)} members) - state=extinct")
    
    for sid in incubator_species:
        species.pop(sid, None)
    
    return species, extinction_events, archived_from_extinction_events, incubator_species
