

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
_, _, _, get_outputs_path, _, _, _ = get_system_utils()


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
    
    if genome_tracker:
        genome_tracker.update_species_id(str(genome_id), species_id, current_generation, "variant_assignment")
    
    if speciation_state_path_obj.exists():
        state_dict = {
            "species": {str(sid): sp.to_dict() for sid, sp in species.items()},
            "generation": current_generation
        }
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
    
    if speciation_state_path_obj.exists():
        state_dict = {
            "species": {str(sid): sp.to_dict() for sid, sp in species.items()},
            "generation": current_generation
        }
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
    
    if events_tracker and species_id in species:
        sp = species[species_id]
        if sp.leader:
            events_tracker.log(sp.leader.id, "leader_updated", {"species_id": species_id, "fitness": sp.leader.fitness})


def _ensure_unique_leader(species: Dict[int, Species], new_leader: Individual, current_species_id: int, logger) -> None:
    
    for sid, sp in species.items():
        if sid != current_species_id and sp.leader and sp.leader.id == new_leader.id:
            logger.warning(f"Genome {new_leader.id} is already leader of species {sid}, reassigning species {sid} leader")
            if new_leader in sp.members:
                sp.members.remove(new_leader)
                new_leader.species_id = None
            
            if len(sp.members) > 0:
                sp.leader = max(sp.members, key=lambda x: x.fitness)
                if sp.leader not in sp.members:
                    sp.members.insert(0, sp.leader)
                logger.info(f"Reassigned species {sid} leader to genome {sp.leader.id} (fitness={sp.leader.fitness:.4f})")
            else:
                sp.species_state = "incubator"
                sp.leader = None
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
    genome_tracker=None,
    events_tracker=None
) -> Tuple[Dict[int, Species], Set[int]]:
    
    if logger is None:
        logger = get_logger("LeaderFollowerClustering")
    
    species_with_new_members = set()
    
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
    
    with open(temp_path_obj, 'r', encoding='utf-8') as f:
        genomes = json.load(f)
    
    if not genomes:
        logger.warning("No genomes found in temp.json")
        return {}, set()
    
    population = [Individual.from_genome(genome) for genome in genomes]
    
    valid_population = [ind for ind in population if ind.embedding is not None]
    if not valid_population:
        logger.error("No individuals with embeddings")
        return {}, set()
    
    existing_species: Dict[int, Species] = {}
    
    if speciation_state_path_obj.exists():
        try:
            with open(speciation_state_path_obj, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            for sid_str, sp_dict in state.get("species", {}).items():
                sid = int(sid_str)
                if sid == CLUSTER_0_ID:
                    continue
                
                leader_embedding = None
                if sp_dict.get("leader_embedding"):
                    leader_embedding = np.array(sp_dict["leader_embedding"])
                
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
                    cluster_origin=sp_dict.get("cluster_origin", "natural"),
                    parent_ids=sp_dict.get("parent_ids"),
                    leader_distance=sp_dict.get("leader_distance", 0.0)
                )
                existing_species[sid] = sp
            
            if existing_species:
                max_species_id = max(existing_species.keys())
                SpeciesIdGenerator.set_min_id(max_species_id + 1)
            
            logger.info(f"Loaded {len(existing_species)} existing species from speciation_state.json")
            
        except Exception as e:
            logger.warning(f"Failed to load speciation_state.json: {e}, starting fresh")
            existing_species = {}
    
    is_generation_0 = len(existing_species) == 0
    
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
    
    sorted_pop = sorted(valid_population, key=lambda x: x.fitness, reverse=True)
    
    species: Dict[int, Species] = existing_species.copy() if existing_species else {}
    leaders: List[Tuple[int, np.ndarray, Optional[np.ndarray]]] = [
        (sid, sp.leader.embedding, sp.leader.phenotype) 
        for sid, sp in species.items() 
        if sp.leader.embedding is not None
    ]
    
    
    potential_leaders: Dict[int, Tuple[Optional[int], np.ndarray, Optional[np.ndarray], Individual, List[Individual]]] = {}
    
    cluster0_potential_leaders: Dict[int, Tuple[Optional[int], np.ndarray, Optional[np.ndarray], Individual, List[Individual]]] = {}
    if not is_generation_0 and cluster0_outliers and not skip_cluster0_outliers:
        for outlier_emb, outlier_pheno, outlier_ind in cluster0_outliers:
            cluster0_potential_leaders[outlier_ind.id] = (None, outlier_emb, outlier_pheno, outlier_ind, [])
        logger.debug(f"Generation N: Loaded {len(cluster0_potential_leaders)} cluster 0 outliers as potential leaders")
    
    if is_generation_0:
        from .gen0_clustering import Gen0Clustering
        species, species_with_new_members = Gen0Clustering.run(
            valid_population, theta_sim, min_island_size, w_genotype, w_phenotype, current_generation, logger
        )
        n_reserves = sum(1 for ind in valid_population if getattr(ind, "species_id", None) == CLUSTER_0_ID)
        logger.info("Gen 0: formed %s species, %s in reserves", len(species), n_reserves)
    else:
        remaining_pop = sorted_pop

    for ind in ([] if is_generation_0 else remaining_pop):
        assigned = False
        min_dist = float('inf')
        nearest_leader_id = None
        
        if leaders:
            if len(leaders) > 1:
                leader_embeddings = np.array([emb for _, emb, _ in leaders])
                leader_phenotypes = [
                    pheno for _, _, pheno in leaders
                ]
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
        
            if nearest_leader_id is not None and min_dist < theta_sim:
                sp = species[nearest_leader_id]
                sp.add_member(ind)
                ind.species_id = nearest_leader_id
                species_with_new_members.add(nearest_leader_id)
                assigned = True
                
                _update_files_immediately(
                    ind.id, nearest_leader_id, species, speciation_state_path_obj,
                    genome_tracker, events_tracker, current_generation, logger
                )
                
                if ind.fitness > sp.leader.fitness:
                    if ind.fitness > sp.max_fitness:
                        sp.max_fitness = ind.fitness
                        sp.stagnation = 0
                    _ensure_unique_leader(species, ind, nearest_leader_id, logger)
                    sp.leader = ind
                    sp.leader_distance = min_dist
                    if sp.species_state == "frozen":
                        sp.species_state = "active"
                        logger.info(f"Frozen species {nearest_leader_id} reactivated: new leader {ind.id} with fitness {ind.fitness:.4f} (stagnation reset to 0)")
                    for i, (sid, _, _) in enumerate(leaders):
                        if sid == nearest_leader_id:
                            leaders[i] = (sid, sp.leader.embedding, sp.leader.phenotype)
                            break
                    
                    _update_files_after_leader_update(
                        nearest_leader_id, species, speciation_state_path_obj,
                        events_tracker, current_generation, logger
                    )
        
        if not assigned and is_generation_0 and potential_leaders:
            for pl_id, (pl_species_id, pl_emb, pl_pheno, pl_ind, followers) in list(potential_leaders.items()):
                dist = ensemble_distance(
                    ind.embedding, pl_emb,
                    ind.phenotype, pl_pheno,
                    w_genotype, w_phenotype
                )
                if dist < theta_sim:
                    if pl_species_id is None:
                        followers.append(ind)
                        ind.species_id = CLUSTER_0_ID
                        total_size = 1 + len(followers)
                        if total_size >= min_island_size:
                            new_species_id = generate_species_id()
                            all_members = [pl_ind] + followers
                            leader = max(all_members, key=lambda x: x.fitness)
                            
                            valid_members = [leader]
                            for member in all_members:
                                if member.id == leader.id:
                                    continue
                                
                                if member.embedding is None:
                                    continue
                                
                                dist = ensemble_distance(
                                    member.embedding, leader.embedding,
                                    member.phenotype, leader.phenotype,
                                    w_genotype, w_phenotype
                                )
                                
                                if dist < theta_sim:
                                    valid_members.append(member)
                            
                            if len(valid_members) >= min_island_size:
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
                                for member in valid_members:
                                    member.species_id = new_species_id
                                    if genome_tracker:
                                        genome_tracker.update_species_id(str(member.id), new_species_id, current_generation, "species_formed")
                                    if events_tracker:
                                        events_tracker.log(member.id, "species_formed", {"species_id": new_species_id})
                                species_with_new_members.add(new_species_id)
                                potential_leaders[pl_id] = (new_species_id, leader.embedding, leader.phenotype, leader, followers)
                                leaders.append((new_species_id, leader.embedding, leader.phenotype))
                                
                                _update_files_after_leader_update(
                                    new_species_id, species, speciation_state_path_obj,
                                    events_tracker, current_generation, logger
                                )
                                
                                logger.info(
                                    f"Species {new_species_id} formed: leader {leader.id} + {len(other_members)} followers "
                                    f"(total={len(valid_members)}, min={min_island_size}, filtered from {len(all_members)} candidates)"
                                )
                            else:
                                invalid_members = [m for m in all_members if m not in valid_members]
                                for invalid in invalid_members:
                                    if invalid in followers:
                                        followers.remove(invalid)
                                logger.debug(
                                    f"Potential leader {pl_ind.id}: {len(all_members)} candidates but only {len(valid_members)} "
                                    f"within new leader's radius (need {min_island_size}), not creating species"
                                )
                        else:
                            logger.debug(f"Potential leader {pl_ind.id} now has {len(followers)} followers (need {min_island_size - 1} total)")
                    else:
                        sp = species[pl_species_id]
                        sp.add_member(ind)
                        ind.species_id = pl_species_id
                        species_with_new_members.add(pl_species_id)
                        
                        _update_files_immediately(
                            ind.id, pl_species_id, species, speciation_state_path_obj,
                            genome_tracker, events_tracker, current_generation, logger
                        )
                        
                        if ind.fitness > sp.leader.fitness:
                            if ind.fitness > sp.max_fitness:
                                sp.max_fitness = ind.fitness
                                sp.stagnation = 0
                            sp.leader = ind
                            sp.leader_distance = dist
                            if sp.species_state == "frozen":
                                sp.species_state = "active"
                                logger.info(f"Frozen species {pl_species_id} reactivated: new leader {ind.id} with fitness {ind.fitness:.4f} (stagnation reset to 0)")
                            for i, (sid, _, _) in enumerate(leaders):
                                if sid == pl_species_id:
                                    leaders[i] = (sid, sp.leader.embedding, sp.leader.phenotype)
                                    break
                            
                            _update_files_after_leader_update(
                                pl_species_id, species, speciation_state_path_obj,
                                events_tracker, current_generation, logger
                            )
                    assigned = True
                    break
        
        if not assigned and not is_generation_0 and cluster0_potential_leaders:
            for pl_id, (pl_species_id, pl_emb, pl_pheno, pl_ind, followers) in list(cluster0_potential_leaders.items()):
                dist = ensemble_distance(
                    ind.embedding, pl_emb,
                    ind.phenotype, pl_pheno,
                    w_genotype, w_phenotype
                )
                if dist < theta_sim:
                    if pl_species_id is None:
                        followers.append(ind)
                        ind.species_id = CLUSTER_0_ID
                        total_size = 1 + len(followers)
                        if total_size >= min_island_size:
                            new_species_id = generate_species_id()
                            all_members = [pl_ind] + followers
                            leader = max(all_members, key=lambda x: x.fitness)
                            
                            valid_members = [leader]
                            for member in all_members:
                                if member.id == leader.id:
                                    continue
                                
                                if member.embedding is None:
                                    continue
                                
                                dist = ensemble_distance(
                                    member.embedding, leader.embedding,
                                    member.phenotype, leader.phenotype,
                                    w_genotype, w_phenotype
                                )
                                
                                if dist < theta_sim:
                                    valid_members.append(member)
                            
                            if len(valid_members) >= min_island_size:
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
                                for member in valid_members:
                                    member.species_id = new_species_id
                                    if genome_tracker:
                                        genome_tracker.update_species_id(str(member.id), new_species_id, current_generation, "species_formed_from_cluster0")
                                    if events_tracker:
                                        events_tracker.log(member.id, "species_formed_from_cluster0", {"species_id": new_species_id})
                                species_with_new_members.add(new_species_id)
                                
                                _update_files_after_leader_update(
                                    new_species_id, species, speciation_state_path_obj,
                                    events_tracker, current_generation, logger
                                )
                                cluster0_potential_leaders[pl_id] = (new_species_id, leader.embedding, leader.phenotype, leader, followers)
                                leaders.append((new_species_id, leader.embedding, leader.phenotype))
                                logger.info(
                                    f"Species {new_species_id} formed from cluster 0: leader {leader.id} + {len(other_members)} followers "
                                    f"(total={len(valid_members)}, min={min_island_size}, filtered from {len(all_members)} candidates)"
                                )
                            else:
                                invalid_members = [m for m in all_members if m not in valid_members]
                                for invalid in invalid_members:
                                    if invalid in followers:
                                        followers.remove(invalid)
                                logger.debug(
                                    f"Potential leader {pl_ind.id} (cluster 0): {len(all_members)} candidates but only {len(valid_members)} "
                                    f"within new leader's radius (need {min_island_size}), not creating species"
                                )
                        else:
                            logger.debug(f"Potential leader {pl_ind.id} (cluster 0) now has {len(followers)} followers (need {min_island_size - 1} total)")
                    else:
                        sp = species[pl_species_id]
                        sp.add_member(ind)
                        ind.species_id = pl_species_id
                        species_with_new_members.add(pl_species_id)
                        
                        _update_files_immediately(
                            ind.id, pl_species_id, species, speciation_state_path_obj,
                            genome_tracker, events_tracker, current_generation, logger
                        )
                        
                        if ind.fitness > sp.leader.fitness:
                            if ind.fitness > sp.max_fitness:
                                sp.max_fitness = ind.fitness
                                sp.stagnation = 0
                            _ensure_unique_leader(species, ind, pl_species_id, logger)
                            sp.leader = ind
                            sp.leader_distance = dist
                            if sp.species_state == "frozen":
                                sp.species_state = "active"
                                logger.info(f"Frozen species {pl_species_id} reactivated: new leader {ind.id} with fitness {ind.fitness:.4f} (stagnation reset to 0)")
                            for i, (sid, _, _) in enumerate(leaders):
                                if sid == pl_species_id:
                                    leaders[i] = (sid, sp.leader.embedding, sp.leader.phenotype)
                                    break
                            
                            _update_files_after_leader_update(
                                pl_species_id, species, speciation_state_path_obj,
                                events_tracker, current_generation, logger
                            )
                    assigned = True
                    break
        
        if not assigned:
            if is_generation_0:
                ind.species_id = CLUSTER_0_ID
                potential_leaders[ind.id] = (None, ind.embedding, ind.phenotype, ind, [])
                logger.debug(f"Individual {ind.id} becomes potential leader (cluster 0, min_size={min_island_size})")
            else:
                ind.species_id = CLUSTER_0_ID
                logger.debug(f"Individual {ind.id} outside all species and outliers -> added to cluster 0 (species_id=0)")
    
    if not is_generation_0:
        outliers_formed_species = sum(1 for pl_sid, _, _, _, _ in cluster0_potential_leaders.values() if pl_sid is not None)
        outliers_still_in_cluster0 = sum(1 for pl_sid, _, _, _, followers in cluster0_potential_leaders.values() if pl_sid is None)
        outliers_with_followers = sum(1 for pl_sid, _, _, _, followers in cluster0_potential_leaders.values() if pl_sid is None and len(followers) > 0)
        if outliers_formed_species > 0:
            logger.info(f"Generation N: {outliers_formed_species} cluster 0 outliers formed species (min_size={min_island_size})")
        if outliers_still_in_cluster0 > 0:
            logger.debug(f"Generation N: {outliers_still_in_cluster0} cluster 0 outliers remain (need {min_island_size - 1} followers)")
            if outliers_with_followers > 0:
                logger.debug(f"  ({outliers_with_followers} outliers have some followers but haven't reached min_size yet)")
    
    genome_id_to_species = {ind.id: ind.species_id for ind in valid_population}
    
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
            if genome_tracker and species_id is not None:
                updates_for_tracker[str(genome_id)] = species_id if species_id is not None else 0
        else:
            genome["species_id"] = None
            if genome_tracker:
                updates_for_tracker[str(genome_id)] = 0
    
    if genome_tracker and updates_for_tracker:
        result = genome_tracker.batch_update(updates_for_tracker, current_generation, "leader_follower_clustering")
        if result["failed"] > 0:
            logger.warning(f"Genome tracker batch update had {result['failed']} failures")
        
        reassigned_from_archive = result.get("reassigned_from_archive", [])
        if reassigned_from_archive:
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
    
    with open(temp_path_obj, 'w', encoding='utf-8') as f:
        json.dump(genomes, f, indent=2, ensure_ascii=False)
    
    
    
    logger.info(f"Leader-Follower clustering: {len(valid_population)} individuals -> {len(species)} species")
    logger.debug(f"Species with new members: {sorted(species_with_new_members)}")
    return species, species_with_new_members
