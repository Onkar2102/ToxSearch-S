"""
leader_follower.py

Leader-Follower clustering algorithm for speciation.
Reads temp.json and speciation_state.json directly, performs clustering,
and updates both files directly.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Set, TYPE_CHECKING
from pathlib import Path

from .species import Individual, Species, generate_species_id, SpeciesIdGenerator
from .distance import ensemble_distance, ensemble_distances_batch
from .reserves import CLUSTER_0_ID

if TYPE_CHECKING:
    from .reserves import Cluster0

from utils import get_custom_logging
from utils import get_system_utils

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _ = get_system_utils()


def _update_files_immediately(
    genome_id: int,
    species_id: int,
    species: Dict[int, Species],
    speciation_state_path_obj: Path,
    genome_tracker,
    events_tracker,
    current_generation: int,
    logger
):
    """Update all files immediately after variant assignment or leader update."""
    # 1. Update genome tracker
    if genome_tracker:
        genome_tracker.update_species_id(str(genome_id), species_id, current_generation, "variant_assignment")
    
    # 2. Update speciation_state.json
    if speciation_state_path_obj.exists():
        state_dict = {
            "species": {str(sid): sp.to_dict() for sid, sp in species.items()},
            "generation": current_generation
        }
        # Preserve other fields
        try:
            with open(speciation_state_path_obj, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            state_dict["cluster0"] = existing.get("cluster0", {})
            state_dict["global_best_id"] = existing.get("global_best_id")
            state_dict["metrics"] = existing.get("metrics", {})
        except Exception:
            pass  # If can't read, just save new structure
        with open(speciation_state_path_obj, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)
    
    # 3. Log to events tracker
    if events_tracker:
        events_tracker.log(genome_id, "assigned_to_species", {"species_id": species_id})


def _update_files_after_leader_update(
    species_id: int,
    species: Dict[int, Species],
    speciation_state_path_obj: Path,
    events_tracker,
    current_generation: int,
    logger
):
    """Update files immediately after leader update."""
    # Update speciation_state.json
    if speciation_state_path_obj.exists():
        state_dict = {
            "species": {str(sid): sp.to_dict() for sid, sp in species.items()},
            "generation": current_generation
        }
        # Preserve other fields
        try:
            with open(speciation_state_path_obj, 'r', encoding='utf-8') as f:
                existing = json.load(f)
            state_dict["cluster0"] = existing.get("cluster0", {})
            state_dict["global_best_id"] = existing.get("global_best_id")
            state_dict["metrics"] = existing.get("metrics", {})
        except Exception:
            pass
        with open(speciation_state_path_obj, 'w', encoding='utf-8') as f:
            json.dump(state_dict, f, indent=2, ensure_ascii=False)
    
    # Log to events tracker
    if events_tracker and species_id in species:
        sp = species[species_id]
        if sp.leader:
            events_tracker.log(sp.leader.id, "leader_updated", {"species_id": species_id, "fitness": sp.leader.fitness})


def _ensure_unique_leader(species: Dict[int, Species], new_leader: Individual, current_species_id: int, logger) -> None:
    """
    Ensure new_leader is not already leader of another species.
    
    If duplicate is found, reassigns the old species leader to next highest fitness member.
    If old species only has that leader, marks it for incubator (handled in Phase 5: Stagnation and Incubation).
    
    Note: This function only marks species as incubator but does NOT move members to cluster 0.
    Phase 5 (run_speciation) handles moving members to cluster 0 and removing from dict.
    
    Args:
        species: Dict of all species
        new_leader: Individual that will become leader
        current_species_id: ID of species getting the new leader
        logger: Logger instance
    """
    for sid, sp in species.items():
        if sid != current_species_id and sp.leader and sp.leader.id == new_leader.id:
            logger.warning(f"Genome {new_leader.id} is already leader of species {sid}, reassigning species {sid} leader")
            # Reassign old species leader to next highest fitness
            # First, remove the old leader from members (it's becoming leader of another species)
            if new_leader in sp.members:
                sp.members.remove(new_leader)
                new_leader.species_id = None  # Will be set when added to new species
            
            if len(sp.members) > 0:
                # Find next highest fitness member from remaining members
                sp.leader = max(sp.members, key=lambda x: x.fitness)
                # Ensure new leader is in members (should be, but verify)
                if sp.leader not in sp.members:
                    sp.members.insert(0, sp.leader)
                logger.info(f"Reassigned species {sid} leader to genome {sp.leader.id} (fitness={sp.leader.fitness:.4f})")
            else:
                # No other members - mark for incubator (Phase 5 will handle cleanup)
                sp.species_state = "incubator"
                sp.leader = None  # No leader if no members
                logger.info(f"Species {sid} has no other members, marking as incubator (will be processed in Phase 5)")


def leader_follower_clustering(
    temp_path: Optional[str] = None,
    speciation_state_path: Optional[str] = None,
    theta_sim: float = 0.2,
    current_generation: int = 0,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3,
    min_island_size: int = 2,
    skip_cluster0_outliers: bool = False,
    logger=None,
    genome_tracker=None,  # NEW: Pass tracker for immediate updates
    events_tracker=None  # NEW: Pass events tracker for immediate logging
) -> Tuple[Dict[int, Species], Set[int]]:
    """
    Leader-Follower clustering algorithm that reads and writes files directly.
    
    This function uses DEFERRED SPECIES ID ASSIGNMENT (adaptive to min_island_size):
    - Species IDs are only assigned when a leader gains enough followers to reach min_island_size
    - Individuals that don't fit existing species become "potential leaders"
    - Potential leaders become actual species only when they have (min_island_size - 1) followers
    - Potential leaders without enough followers stay in cluster 0 (reserves)
    
    This ensures:
    - No wasted species IDs
    - No species smaller than min_island_size
    - Species inherently have minimum size of min_island_size
    
    Pipeline:
    1. Reads genomes from temp.json (must have prompt_embedding field)
    2. Reads existing species from speciation_state.json (if exists)
    3. For Generation 0 (no species exist) – two-phase via Gen0Clustering:
       - Phase 1: Process entire population; build all (potential_leader, followers) groups.
         First = potential leader; each subsequent joins nearest potential leader within theta_sim
         or becomes a new potential leader. No species created during this pass.
       - Phase 2: For each group with |leader ∪ followers| >= min_island_size: set leader =
         argmax fitness, keep only valid_members with d(m, leader) < theta_sim; if
         |valid_members| >= min_island_size create Species, else all → reserves. Smaller
         groups → all to reserves.
    4. For Generation N:
       - Checks each genome against existing species leaders
       - If within theta_sim, assigns to that species
       - If not, and skip_cluster0_outliers=False, checks against cluster 0 outliers (from reserves.json) as potential leaders
       - If within theta_sim of an outlier, add as follower (tracked but no species yet)
       - When a potential leader has (min_island_size - 1) followers, species forms
       - If not within radius of any leader or outlier, adds to cluster 0
       - If skip_cluster0_outliers=True, all unassigned variants go directly to cluster 0 (no outlier checking)
       - check_speciation() handles additional species formation from remaining cluster 0 individuals
    5. Updates temp.json with species_id for each genome
    6. Updates speciation_state.json with new species structure
    
    Args:
        temp_path: Path to temp.json (defaults to outputs_path / "temp.json")
        speciation_state_path: Path to speciation_state.json (defaults to outputs_path / "speciation_state.json")
        theta_sim: Semantic distance threshold for species assignment
        current_generation: Current generation number
        w_genotype: Weight for genotype (embedding) distance in ensemble distance
        w_phenotype: Weight for phenotype distance in ensemble distance
        min_island_size: Minimum species size required before assigning species ID (default: 2)
        skip_cluster0_outliers: If True, skip loading and using cluster 0 outliers as potential leaders (default: False)
        logger: Optional logger instance
    
    Returns:
        Tuple of (Dict mapping species_id -> Species, Set of species_ids that received new members)
    """
    if logger is None:
        logger = get_logger("LeaderFollowerClustering")
    
    # Track which species received new members (for optimization)
    species_with_new_members = set()
    
    # Determine file paths
    if temp_path is None:
        outputs_path = get_outputs_path()
        temp_path = str(outputs_path / "temp.json")
    
    if speciation_state_path is None:
        outputs_path = get_outputs_path()
        speciation_state_path = str(outputs_path / "speciation_state.json")
    
    temp_path_obj = Path(temp_path)
    speciation_state_path_obj = Path(speciation_state_path)
    
    if not temp_path_obj.exists():
        logger.error(f"Temp file not found: {temp_path}")
        return {}, set()
    
    # Read genomes from temp.json
    with open(temp_path_obj, 'r', encoding='utf-8') as f:
        genomes = json.load(f)
    
    if not genomes:
        logger.warning("No genomes found in temp.json")
        return {}, set()
    
    # Convert genomes to Individual objects
    population = [Individual.from_genome(genome) for genome in genomes]
    
    # Filter out individuals without embeddings
    valid_population = [ind for ind in population if ind.embedding is not None]
    if not valid_population:
        logger.error("No individuals with embeddings")
        return {}, set()
    
    # Read existing species from speciation_state.json (if exists)
    existing_species: Dict[int, Species] = {}
    
    if speciation_state_path_obj.exists():
        try:
            with open(speciation_state_path_obj, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Restore species (excluding species 0)
            for sid_str, sp_dict in state.get("species", {}).items():
                sid = int(sid_str)
                if sid == CLUSTER_0_ID:
                    continue  # Skip species 0 (it's cluster0/reserves)
                
                # Reconstruct leader Individual
                leader_embedding = None
                if sp_dict.get("leader_embedding"):
                    leader_embedding = np.array(sp_dict["leader_embedding"])
                
                # Extract phenotype if available in genome_data
                leader_phenotype = None
                if sp_dict.get("leader_genome_data"):
                    from .phenotype_distance import extract_phenotype_vector
                    leader_phenotype = extract_phenotype_vector(sp_dict["leader_genome_data"], logger=logger)
                
                leader = Individual(
                    id=sp_dict["leader_id"],
                    prompt=sp_dict.get("leader_prompt", ""),
                    fitness=sp_dict.get("leader_fitness", 0.0),
                    embedding=leader_embedding,
                    phenotype=leader_phenotype,
                    species_id=sid
                )
                
                # Reconstruct species (members will be populated during clustering).
                # max_fitness = actual max over current members only, no merge with stored value.
                members_init = [leader]
                max_fit = max((m.fitness for m in members_init), default=0.0)
                sp = Species(
                    id=sid,
                    leader=leader,
                    members=members_init,
                    radius=sp_dict.get("radius", theta_sim),
                    stagnation=sp_dict.get("stagnation", 0),
                    max_fitness=max_fit,
                    species_state=sp_dict.get("species_state", "active"),
                    created_at=sp_dict.get("created_at", 0),
                    last_improvement=sp_dict.get("last_improvement", 0),
                    fitness_history=sp_dict.get("fitness_history", []),
                    labels=sp_dict.get("labels", []),
                    label_history=sp_dict.get("label_history", []),
                    cluster_origin=sp_dict.get("cluster_origin", "natural"),  # Default to "natural" if None
                    parent_ids=sp_dict.get("parent_ids"),
                    leader_distance=sp_dict.get("leader_distance", 0.0)
                )
                existing_species[sid] = sp
            
            # Update SpeciesIdGenerator to avoid ID conflicts
            if existing_species:
                max_species_id = max(existing_species.keys())
                SpeciesIdGenerator.set_min_id(max_species_id + 1)
            
            logger.info(f"Loaded {len(existing_species)} existing species from speciation_state.json")
            
        except Exception as e:
            logger.warning(f"Failed to load speciation_state.json: {e}, starting fresh")
            existing_species = {}
    
    # Determine if this is Generation 0 (no species exist, or only species 0)
    is_generation_0 = len(existing_species) == 0
    
    # Load cluster 0 outliers for Generation N (if not already loaded above and not skipping)
    cluster0_outliers: List[Tuple[np.ndarray, Optional[np.ndarray], Individual]] = []
    if not is_generation_0 and not skip_cluster0_outliers:
        outputs_path = get_outputs_path()
        reserves_path = outputs_path / "reserves.json"
        if reserves_path.exists():
            try:
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
                for genome in reserves_genomes:
                    if genome.get("prompt_embedding"):
                        outlier_emb = np.array(genome["prompt_embedding"])
                        # Extract phenotype if available
                        outlier_pheno = None
                        if genome.get("moderation_result"):
                            from .phenotype_distance import extract_phenotype_vector
                            outlier_pheno = extract_phenotype_vector(genome.get("moderation_result"), logger=logger)
                        outlier_ind = Individual.from_genome(genome)
                        if outlier_ind.embedding is not None:
                            cluster0_outliers.append((outlier_emb, outlier_pheno, outlier_ind))
                logger.debug(f"Loaded {len(cluster0_outliers)} cluster 0 outliers from reserves.json")
            except Exception as e:
                logger.warning(f"Failed to load cluster 0 outliers from reserves.json: {e}")
    elif not is_generation_0 and skip_cluster0_outliers:
        logger.debug("Skipping cluster 0 outliers loading (skip_cluster0_outliers=True)")
    
    # Sort population by fitness (descending) - highest fitness processed first
    sorted_pop = sorted(valid_population, key=lambda x: x.fitness, reverse=True)
    
    # Initialize species dict
    species: Dict[int, Species] = existing_species.copy() if existing_species else {}
    leaders: List[Tuple[int, np.ndarray, Optional[np.ndarray]]] = [
        (sid, sp.leader.embedding, sp.leader.phenotype) 
        for sid, sp in species.items() 
        if sp.leader.embedding is not None
    ]
    
    # DEFERRED SPECIES ID ASSIGNMENT (adaptive to min_island_size)
    # For Generation 0: Use potential leaders - species only form when min_island_size members are reached
    # For Generation N: Use existing species leaders
    
    # Potential leaders: Dict mapping leader_id -> (species_id_or_None, embedding, phenotype, Individual, followers_list)
    # species_id is None until (min_island_size - 1) followers are found, then it's assigned
    # followers_list: List of Individual objects that are followers of this potential leader
    # Note: leader_id is an integer (ind.id), not a string
    potential_leaders: Dict[int, Tuple[Optional[int], np.ndarray, Optional[np.ndarray], Individual, List[Individual]]] = {}
    
    # For Generation N: Track cluster 0 outliers as potential leaders (only if not skipping)
    # Format: Dict mapping leader_id -> (species_id_or_None, embedding, phenotype, Individual, followers_list)
    # Note: leader_id is an integer (ind.id), not a string
    cluster0_potential_leaders: Dict[int, Tuple[Optional[int], np.ndarray, Optional[np.ndarray], Individual, List[Individual]]] = {}
    if not is_generation_0 and cluster0_outliers and not skip_cluster0_outliers:
        # Convert cluster 0 outliers to potential leaders
        for outlier_emb, outlier_pheno, outlier_ind in cluster0_outliers:
            cluster0_potential_leaders[outlier_ind.id] = (None, outlier_emb, outlier_pheno, outlier_ind, [])
        logger.debug(f"Generation N: Loaded {len(cluster0_potential_leaders)} cluster 0 outliers as potential leaders")
    
    if is_generation_0:
        # Two-phase Gen 0: (1) build all leader–follower groups over entire population;
        # (2) create species only for groups with |leader ∪ followers| >= min_island_size.
        from .gen0_clustering import Gen0Clustering
        species, species_with_new_members = Gen0Clustering.run(
            valid_population, theta_sim, min_island_size, w_genotype, w_phenotype, current_generation, logger
        )
        n_reserves = sum(1 for ind in valid_population if getattr(ind, "species_id", None) == CLUSTER_0_ID)
        logger.info("Gen 0: formed %s species, %s in reserves", len(species), n_reserves)
    else:
        remaining_pop = sorted_pop

    # Process remaining individuals (Gen N only; Gen 0 uses Gen0Clustering and skips via empty iterable)
    for ind in ([] if is_generation_0 else remaining_pop):
        assigned = False
        min_dist = float('inf')
        nearest_leader_id = None
        
        # 1. First check against existing species leaders (Gen N, or Gen 0 after species form)
        if leaders:
            if len(leaders) > 1:
                leader_embeddings = np.array([emb for _, emb, _ in leaders])
                leader_phenotypes = [
                    pheno for _, _, pheno in leaders
                ]  # Keep as list to handle None values
                # ensemble_distances_batch handles None phenotypes by falling back to genotype-only
                distances = ensemble_distances_batch(
                    ind.embedding, leader_embeddings,
                    ind.phenotype, leader_phenotypes,
                    w_genotype, w_phenotype
                )
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                nearest_leader_id = leaders[min_idx][0]
            elif len(leaders) == 1:
                leader_emb = leaders[0][1]
                leader_pheno = leaders[0][2]
                min_dist = ensemble_distance(
                    ind.embedding, leader_emb,
                    ind.phenotype, leader_pheno,
                    w_genotype, w_phenotype
                )
                nearest_leader_id = leaders[0][0]
        
            # Assign to existing species if within threshold
            # IMPORTANT: If multiple leaders are within theta_sim, we assign to the CLOSEST one
            # (nearest_leader_id is determined by min_dist from ensemble_distances_batch above)
            # This ensures deterministic and principled assignment when genomes could fit multiple species
            if nearest_leader_id is not None and min_dist < theta_sim:
                sp = species[nearest_leader_id]
                sp.add_member(ind)
                ind.species_id = nearest_leader_id
                species_with_new_members.add(nearest_leader_id)
                assigned = True
                
                # Update files immediately after assignment
                _update_files_immediately(
                    ind.id, nearest_leader_id, species, speciation_state_path_obj,
                    genome_tracker, events_tracker, current_generation, logger
                )
                
                # Update leader if this new member has higher fitness
                if ind.fitness > sp.leader.fitness:
                    if ind.fitness > sp.max_fitness:
                        sp.max_fitness = ind.fitness
                        sp.stagnation = 0
                    # Ensure new leader is not already leader of another species
                    _ensure_unique_leader(species, ind, nearest_leader_id, logger)
                    sp.leader = ind
                    sp.leader_distance = min_dist
                    # Reactivate frozen species when leader is updated (without merging)
                    if sp.species_state == "frozen":
                        sp.species_state = "active"
                        logger.info(f"Frozen species {nearest_leader_id} reactivated: new leader {ind.id} with fitness {ind.fitness:.4f} (stagnation reset to 0)")
                    for i, (sid, _, _) in enumerate(leaders):
                        if sid == nearest_leader_id:
                            leaders[i] = (sid, sp.leader.embedding, sp.leader.phenotype)
                            break
                    
                    # Update files immediately after leader update
                    _update_files_after_leader_update(
                        nearest_leader_id, species, speciation_state_path_obj,
                        events_tracker, current_generation, logger
                    )
        
        # 2. For Generation 0: Check against potential leaders
        if not assigned and is_generation_0 and potential_leaders:
            # Use list() to create snapshot - prevents RuntimeError when modifying dict during iteration
            for pl_id, (pl_species_id, pl_emb, pl_pheno, pl_ind, followers) in list(potential_leaders.items()):
                dist = ensemble_distance(
                    ind.embedding, pl_emb,
                    ind.phenotype, pl_pheno,
                    w_genotype, w_phenotype
                )
                if dist < theta_sim:
                    if pl_species_id is None:
                        # Add as follower (tracked but no species yet)
                        followers.append(ind)
                        ind.species_id = CLUSTER_0_ID  # Stay in cluster 0 until species forms
                        # Check if we've reached minimum size
                        total_size = 1 + len(followers)  # leader + followers
                        if total_size >= min_island_size:
                            # Minimum size reached! Create the species now
                            new_species_id = generate_species_id()
                            # Determine leader (highest fitness among leader + followers)
                            all_members = [pl_ind] + followers
                            leader = max(all_members, key=lambda x: x.fitness)
                            
                            # After choosing the new leader, verify all members are within the leader's radius
                            # This ensures species are created only with members within the radius threshold,
                            # preventing members from being removed immediately in radius cleanup phase.
                            # This matches the logic in check_speciation() for consistency.
                            valid_members = [leader]  # Leader always included
                            for member in all_members:
                                if member.id == leader.id:
                                    continue  # Skip leader (already added)
                                
                                if member.embedding is None:
                                    continue  # Skip members without embeddings
                                
                                # Check distance to new leader
                                dist = ensemble_distance(
                                    member.embedding, leader.embedding,
                                    member.phenotype, leader.phenotype,
                                    w_genotype, w_phenotype
                                )
                                
                                if dist < theta_sim:
                                    valid_members.append(member)
                            
                            # Only create species if it has at least min_island_size valid members
                            if len(valid_members) >= min_island_size:
                                # Ensure leader is not already leader of another species
                                _ensure_unique_leader(species, leader, new_species_id, logger)
                                other_members = [m for m in valid_members if m.id != leader.id]
                                
                                new_species = Species(
                                    id=new_species_id,
                                    leader=leader,
                                    members=valid_members,
                                    radius=theta_sim,
                                    created_at=current_generation,
                                    last_improvement=current_generation,
                                    cluster_origin="natural",
                                    parent_ids=None,
                                    leader_distance=0.0
                                )
                                species[new_species_id] = new_species
                                # Assign species_id to all valid members
                                for member in valid_members:
                                    member.species_id = new_species_id
                                    # Update files immediately for each member
                                    if genome_tracker:
                                        genome_tracker.update_species_id(str(member.id), new_species_id, current_generation, "species_formed")
                                    if events_tracker:
                                        events_tracker.log(member.id, "species_formed", {"species_id": new_species_id})
                                species_with_new_members.add(new_species_id)
                                # Update potential_leaders entry with new species ID (use new leader's embedding/phenotype)
                                potential_leaders[pl_id] = (new_species_id, leader.embedding, leader.phenotype, leader, followers)
                                # Add to leaders list for future distance checks (use new leader's embedding/phenotype)
                                leaders.append((new_species_id, leader.embedding, leader.phenotype))
                                
                                # Update speciation_state.json immediately after species formation
                                _update_files_after_leader_update(
                                    new_species_id, species, speciation_state_path_obj,
                                    events_tracker, current_generation, logger
                                )
                                
                                logger.info(
                                    f"Species {new_species_id} formed: leader {leader.id} + {len(other_members)} followers "
                                    f"(total={len(valid_members)}, min={min_island_size}, filtered from {len(all_members)} candidates)"
                                )
                            else:
                                # Not enough valid members after filtering - remove invalid members from followers
                                invalid_members = [m for m in all_members if m not in valid_members]
                                for invalid in invalid_members:
                                    if invalid in followers:
                                        followers.remove(invalid)
                                logger.debug(
                                    f"Potential leader {pl_ind.id}: {len(all_members)} candidates but only {len(valid_members)} "
                                    f"within new leader's radius (need {min_island_size}), not creating species"
                                )
                        else:
                            # Not enough followers yet, keep tracking
                            logger.debug(f"Potential leader {pl_ind.id} now has {len(followers)} followers (need {min_island_size - 1} total)")
                    else:
                        # Species already exists (formed earlier), just add as member
                        sp = species[pl_species_id]
                        sp.add_member(ind)
                        ind.species_id = pl_species_id
                        species_with_new_members.add(pl_species_id)
                        
                        # Update files immediately after assignment
                        _update_files_immediately(
                            ind.id, pl_species_id, species, speciation_state_path_obj,
                            genome_tracker, events_tracker, current_generation, logger
                        )
                        
                        # Update leader if higher fitness
                        if ind.fitness > sp.leader.fitness:
                            if ind.fitness > sp.max_fitness:
                                sp.max_fitness = ind.fitness
                                sp.stagnation = 0
                            sp.leader = ind
                            sp.leader_distance = dist
                            # Reactivate frozen species when leader is updated (without merging)
                            if sp.species_state == "frozen":
                                sp.species_state = "active"
                                logger.info(f"Frozen species {pl_species_id} reactivated: new leader {ind.id} with fitness {ind.fitness:.4f} (stagnation reset to 0)")
                            # Update leaders list
                            for i, (sid, _, _) in enumerate(leaders):
                                if sid == pl_species_id:
                                    leaders[i] = (sid, sp.leader.embedding, sp.leader.phenotype)
                                    break
                            
                            # Update files immediately after leader update
                            _update_files_after_leader_update(
                                pl_species_id, species, speciation_state_path_obj,
                                events_tracker, current_generation, logger
                            )
                    assigned = True
                    break
        
        # 3. For Generation N: Check against cluster 0 outliers (potential leaders)
        if not assigned and not is_generation_0 and cluster0_potential_leaders:
            # Use list() to create snapshot - prevents RuntimeError when modifying dict during iteration
            for pl_id, (pl_species_id, pl_emb, pl_pheno, pl_ind, followers) in list(cluster0_potential_leaders.items()):
                dist = ensemble_distance(
                    ind.embedding, pl_emb,
                    ind.phenotype, pl_pheno,
                    w_genotype, w_phenotype
                )
                if dist < theta_sim:
                    if pl_species_id is None:
                        # Add as follower (tracked but no species yet)
                        followers.append(ind)
                        ind.species_id = CLUSTER_0_ID  # Stay in cluster 0 until species forms
                        # Check if we've reached minimum size
                        total_size = 1 + len(followers)  # leader + followers
                        if total_size >= min_island_size:
                            # Minimum size reached! Create the species now
                            new_species_id = generate_species_id()
                            # Determine leader (highest fitness among leader + followers)
                            all_members = [pl_ind] + followers
                            leader = max(all_members, key=lambda x: x.fitness)
                            
                            # After choosing the new leader, verify all members are within the leader's radius
                            # This prevents species from being created with members outside the new leader's radius
                            # which would cause them to be removed immediately in radius cleanup
                            # This matches the logic in Generation 0 and check_speciation() for consistency
                            valid_members = [leader]  # Leader always included
                            for member in all_members:
                                if member.id == leader.id:
                                    continue  # Skip leader (already added)
                                
                                if member.embedding is None:
                                    continue  # Skip members without embeddings
                                
                                # Check distance to new leader
                                dist = ensemble_distance(
                                    member.embedding, leader.embedding,
                                    member.phenotype, leader.phenotype,
                                    w_genotype, w_phenotype
                                )
                                
                                if dist < theta_sim:
                                    valid_members.append(member)
                            
                            # Only create species if it has at least min_island_size valid members
                            if len(valid_members) >= min_island_size:
                                # Ensure leader is not already leader of another species
                                _ensure_unique_leader(species, leader, new_species_id, logger)
                                other_members = [m for m in valid_members if m.id != leader.id]
                                
                                new_species = Species(
                                    id=new_species_id,
                                    leader=leader,
                                    members=valid_members,
                                    radius=theta_sim,
                                    created_at=current_generation,
                                    last_improvement=current_generation,
                                    cluster_origin="natural",
                                    parent_ids=None,
                                    leader_distance=0.0
                                )
                                species[new_species_id] = new_species
                                # Assign species_id to all valid members
                                for member in valid_members:
                                    member.species_id = new_species_id
                                    # Update files immediately for each member
                                    if genome_tracker:
                                        genome_tracker.update_species_id(str(member.id), new_species_id, current_generation, "species_formed_from_cluster0")
                                    if events_tracker:
                                        events_tracker.log(member.id, "species_formed_from_cluster0", {"species_id": new_species_id})
                                species_with_new_members.add(new_species_id)
                                
                                # Update speciation_state.json immediately after species formation
                                _update_files_after_leader_update(
                                    new_species_id, species, speciation_state_path_obj,
                                    events_tracker, current_generation, logger
                                )
                                # Update cluster0_potential_leaders entry with new species ID (use new leader's embedding/phenotype)
                                cluster0_potential_leaders[pl_id] = (new_species_id, leader.embedding, leader.phenotype, leader, followers)
                                # Add to leaders list for future distance checks (use new leader's embedding/phenotype)
                                leaders.append((new_species_id, leader.embedding, leader.phenotype))
                                logger.info(
                                    f"Species {new_species_id} formed from cluster 0: leader {leader.id} + {len(other_members)} followers "
                                    f"(total={len(valid_members)}, min={min_island_size}, filtered from {len(all_members)} candidates)"
                                )
                            else:
                                # Not enough valid members after filtering - remove invalid members from followers
                                invalid_members = [m for m in all_members if m not in valid_members]
                                for invalid in invalid_members:
                                    if invalid in followers:
                                        followers.remove(invalid)
                                logger.debug(
                                    f"Potential leader {pl_ind.id} (cluster 0): {len(all_members)} candidates but only {len(valid_members)} "
                                    f"within new leader's radius (need {min_island_size}), not creating species"
                                )
                        else:
                            # Not enough followers yet, keep tracking
                            logger.debug(f"Potential leader {pl_ind.id} (cluster 0) now has {len(followers)} followers (need {min_island_size - 1} total)")
                    else:
                        # Species already exists (formed earlier), just add as member
                        sp = species[pl_species_id]
                        sp.add_member(ind)
                        ind.species_id = pl_species_id
                        species_with_new_members.add(pl_species_id)
                        
                        # Update files immediately after assignment
                        _update_files_immediately(
                            ind.id, pl_species_id, species, speciation_state_path_obj,
                            genome_tracker, events_tracker, current_generation, logger
                        )
                        
                        # Update leader if higher fitness
                        if ind.fitness > sp.leader.fitness:
                            if ind.fitness > sp.max_fitness:
                                sp.max_fitness = ind.fitness
                                sp.stagnation = 0
                            # Ensure new leader is not already leader of another species
                            _ensure_unique_leader(species, ind, pl_species_id, logger)
                            sp.leader = ind
                            sp.leader_distance = dist
                            # Reactivate frozen species when leader is updated (without merging)
                            if sp.species_state == "frozen":
                                sp.species_state = "active"
                                logger.info(f"Frozen species {pl_species_id} reactivated: new leader {ind.id} with fitness {ind.fitness:.4f} (stagnation reset to 0)")
                            # Update leaders list
                            for i, (sid, _, _) in enumerate(leaders):
                                if sid == pl_species_id:
                                    leaders[i] = (sid, sp.leader.embedding, sp.leader.phenotype)
                                    break
                            
                            # Update files immediately after leader update
                            _update_files_after_leader_update(
                                pl_species_id, species, speciation_state_path_obj,
                                events_tracker, current_generation, logger
                            )
                    assigned = True
                    break
        
        # 4. If not assigned to any species, potential leader, or cluster 0 outlier
        if not assigned:
            if is_generation_0:
                # Become a new potential leader
                ind.species_id = CLUSTER_0_ID
                potential_leaders[ind.id] = (None, ind.embedding, ind.phenotype, ind, [])
                logger.debug(f"Individual {ind.id} becomes potential leader (cluster 0, min_size={min_island_size})")
            else:
                # Generation N: Not similar to species leaders or cluster 0 outliers -> add to cluster 0
                ind.species_id = CLUSTER_0_ID
                logger.debug(f"Individual {ind.id} outside all species and outliers -> added to cluster 0 (species_id=0)")
    
    # Log summary (Gen N only; Gen 0 logs in Gen0Clustering)
    if not is_generation_0:
        # Generation N: Log how many outliers formed species
        outliers_formed_species = sum(1 for pl_sid, _, _, _, _ in cluster0_potential_leaders.values() if pl_sid is not None)
        outliers_still_in_cluster0 = sum(1 for pl_sid, _, _, _, followers in cluster0_potential_leaders.values() if pl_sid is None)
        outliers_with_followers = sum(1 for pl_sid, _, _, _, followers in cluster0_potential_leaders.values() if pl_sid is None and len(followers) > 0)
        if outliers_formed_species > 0:
            logger.info(f"Generation N: {outliers_formed_species} cluster 0 outliers formed species (min_size={min_island_size})")
        if outliers_still_in_cluster0 > 0:
            logger.debug(f"Generation N: {outliers_still_in_cluster0} cluster 0 outliers remain (need {min_island_size - 1} followers)")
            if outliers_with_followers > 0:
                logger.debug(f"  ({outliers_with_followers} outliers have some followers but haven't reached min_size yet)")
    
    # Update temp.json with species_id for each genome
    # Also update genome tracker (master registry) if available
    genome_id_to_species = {ind.id: ind.species_id for ind in valid_population}
    
    # Try to get genome tracker from state
    try:
        from .run_speciation import _get_state
        state = _get_state()
        genome_tracker = state.get("_genome_tracker")
    except Exception:
        genome_tracker = None
    
    updates_for_tracker = {}
    for genome in genomes:
        genome_id = genome.get("id")
        if genome_id in genome_id_to_species:
            species_id = genome_id_to_species[genome_id]
            genome["species_id"] = species_id
            # Prepare batch update for tracker
            if genome_tracker and species_id is not None:
                updates_for_tracker[str(genome_id)] = species_id if species_id is not None else 0
        else:
            # Genome without embedding gets species_id=None
            genome["species_id"] = None
            if genome_tracker:
                updates_for_tracker[str(genome_id)] = 0
    
    # Batch update genome tracker
    if genome_tracker and updates_for_tracker:
        result = genome_tracker.batch_update(updates_for_tracker, current_generation, "leader_follower_clustering")
        if result["failed"] > 0:
            logger.warning(f"Genome tracker batch update had {result['failed']} failures")
        
        # Log reassignment events (if any): in current design archive genomes are not moved back; this supports possible future use
        reassigned_from_archive = result.get("reassigned_from_archive", [])
        if reassigned_from_archive:
            # Try to get events tracker from state
            try:
                from .run_speciation import _get_state
                state = _get_state()
                events_tracker = state.get("_events_tracker")
                if events_tracker:
                    for genome_id, old_species_id, new_species_id in reassigned_from_archive:
                        events_tracker.log(
                            genome_id, "reassigned_from_archive",
                            {
                                "from_species_id": old_species_id,
                                "to_species_id": new_species_id,
                                "reason": "leader_follower_clustering",
                                "note": "Genome was previously archived but is now active"
                            }
                        )
                    logger.info(f"Logged {len(reassigned_from_archive)} reassignment events for genomes reactivated from archive")
                else:
                    logger.warning(f"Events tracker not available - {len(reassigned_from_archive)} reassignment events not logged (genomes: {[gid for gid, _, _ in reassigned_from_archive[:5]]})")
            except Exception as e:
                logger.warning(f"Could not log reassignment events: {e} (this may indicate events_tracker is not initialized)")
    
    # Save updated temp.json
    with open(temp_path_obj, 'w', encoding='utf-8') as f:
        json.dump(genomes, f, indent=2, ensure_ascii=False)
    
    # NOTE: reserves.json is NOT updated here - Phase 7 (redistribution) handles all file distribution
    # from genome_tracker. This ensures single source of truth (tracker) for species membership.
    # Outliers that formed species are already tracked in genome_tracker via update_species_id calls above.
    
    # NOTE: speciation_state.json is NOT updated here - Phase 8 (save_state) handles the final save.
    # Immediate updates to genome_tracker and events_tracker are done above when trackers are provided.
    
    logger.info(f"Leader-Follower clustering: {len(valid_population)} individuals -> {len(species)} species")
    logger.debug(f"Species with new members: {sorted(species_with_new_members)}")
    return species, species_with_new_members
