

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

from .species import Individual, Species, generate_species_id
from .distance import ensemble_distance, ensemble_distances_batch

if TYPE_CHECKING:
    from .config import SpeciationConfig

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()

CLUSTER_0_ID = 0


@dataclass
class Cluster0Individual:
    """Wrapper for individuals in Cluster 0 (reserves). Cluster 0 individuals are outliers that don't fit existing species. They are preserved for potential speciation if enough similar individuals accumulate. Attributes: individual: The Individual instance in Cluster 0 (reserves) entered_at: Generation when individual entered Cluster 0 (reserves)"""
    individual: Individual
    entered_at: int
    
    def __hash__(self):
        return hash(self.individual.id)
    
    def __eq__(self, other):
        return isinstance(other, Cluster0Individual) and self.individual.id == other.individual.id




class Cluster0:
    """Cluster 0 (reserves) for individuals that don't fit existing species. Cluster 0 (ID=0) is a special holding area for: 1. High-fitness outliers that are semantically distant from all species leaders 2. Individuals removed from species when they exceed max capacity (100) 3. Individuals that may form new species if enough similar ones accumulate Key features: - Max capacity: Limited to cluster0_max_capacity (default 1000) individuals. When over capacity, selection is controlled by config.cluster0_selection: "nsga2" (default) uses NSGA-II with diversity first, then toxicity; "toxicity_only" uses legacy sort by fitness. - Speciation detection: When Cluster 0 individuals form cohesive clusters, they create new species - Leader-follower clustering: Uses leader-follower algorithm to find groups in Cluster 0 This preserves diversity by giving novel high-fitness solutions a chance to form new species rather than being discarded."""
    
    def __init__(
        self,
        min_cluster_size: int = 2,
        theta_sim: float = 0.2,
        max_capacity: int = 1000,
        min_island_size: int = 2,
        w_genotype: float = 0.7,
        w_phenotype: float = 0.3,
        logger=None
    ):
        
        self.members: List[Cluster0Individual] = []
        self.min_cluster_size = min_cluster_size
        self.theta_sim = theta_sim
        self.max_capacity = max_capacity
        self.min_island_size = min_island_size
        self.w_genotype = w_genotype
        self.w_phenotype = w_phenotype
        self.logger = logger or get_logger("Cluster0")
        self.speciation_events: List[Dict] = []
    
    @property
    def size(self) -> int:
        
        return len(self.members)
    
    @property
    def individuals(self) -> List[Individual]:
        
        return [lm.individual for lm in self.members]
    
    def add(self, individual: Individual, generation: int) -> None:
        
        for lm in self.members:
            if lm.individual.id == individual.id:
                return
        
        individual.species_id = CLUSTER_0_ID
        
        self.members.append(Cluster0Individual(
            individual=individual,
            entered_at=generation
        ))
    
    def add_batch(self, individuals: List[Individual], generation: int) -> None:
        
        for ind in individuals:
            self.add(ind, generation)
    
    def remove(self, individual: Individual) -> bool:
        
        for i, lm in enumerate(self.members):
            if lm.individual.id == individual.id:
                self.members.pop(i)
                return True
        return False
    
    def remove_batch(self, individuals: List[Individual]) -> int:
        
        ids = {ind.id for ind in individuals}
        original = len(self.members)
        self.members = [lm for lm in self.members if lm.individual.id not in ids]
        return original - len(self.members)
    
    def check_speciation(self, current_generation: int) -> List[Species]:
        
        if len(self.members) < self.min_cluster_size:
            return []
        
        individuals = [lm.individual for lm in self.members if lm.individual.embedding is not None]
        if len(individuals) < self.min_cluster_size:
            return []
        
        sorted_individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)
        
        potential_leaders: Dict[str, Tuple[Optional[int], np.ndarray, Optional[np.ndarray], Individual, List[Individual]]] = {}
        
        new_species_list: List[Species] = []
        individuals_to_remove: List[Individual] = []
        
        first = sorted_individuals[0]
        potential_leaders[first.id] = (None, first.embedding, first.phenotype, first, [])
        remaining_individuals = sorted_individuals[1:]
        
        for ind in remaining_individuals:
            assigned = False
            min_dist = float('inf')
            nearest_leader_id = None
            
            active_leaders = {pl_id: data for pl_id, data in potential_leaders.items() if data[0] is None}
            if active_leaders:
                leader_embeddings = []
                leader_phenotypes = []
                leader_ids = []
                for pl_id, (_, pl_emb, pl_pheno, _, _) in active_leaders.items():
                    leader_ids.append(pl_id)
                    leader_embeddings.append(pl_emb)
                    leader_phenotypes.append(pl_pheno)
                
                if len(leader_embeddings) > 1:
                    leader_embeddings_array = np.array(leader_embeddings)
                    distances = ensemble_distances_batch(
                        ind.embedding, leader_embeddings_array,
                        ind.phenotype, leader_phenotypes,
                        self.w_genotype, self.w_phenotype
                    )
                    min_idx = np.argmin(distances)
                    min_dist = distances[min_idx]
                    nearest_leader_id = leader_ids[min_idx]
                elif len(leader_embeddings) == 1:
                    min_dist = ensemble_distance(
                        ind.embedding, leader_embeddings[0],
                        ind.phenotype, leader_phenotypes[0],
                        self.w_genotype, self.w_phenotype
                    )
                    nearest_leader_id = leader_ids[0]
                
                if nearest_leader_id is not None and min_dist < self.theta_sim:
                    pl_species_id, pl_emb, pl_pheno, pl_ind, followers = potential_leaders[nearest_leader_id]
                    
                    if pl_species_id is None:
                        followers.append(ind)
                        total_size = 1 + len(followers)
                        if total_size >= self.min_island_size:
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
                                    self.w_genotype, self.w_phenotype
                                )
                                
                                if dist < self.theta_sim:
                                    valid_members.append(member)
                            
                            if len(valid_members) >= self.min_island_size:
                                other_members = [m for m in valid_members if m.id != leader.id]
                                
                                new_species = Species(
                                    id=new_species_id,
                                    leader=leader,
                                    members=valid_members,
                                    radius=self.theta_sim,
                                    created_at=current_generation,
                                    last_improvement=current_generation,
                                    cluster_origin="natural",
                                    parent_ids=None,
                                    leader_distance=0.0
                                )
                                
                                potential_leaders[nearest_leader_id] = (new_species_id, pl_emb, pl_pheno, pl_ind, followers)
                                
                                individuals_to_remove.extend(valid_members)
                                new_species_list.append(new_species)
                                
                                self.speciation_events.append({
                                    "generation": current_generation,
                                    "species_id": new_species.id,
                                    "size": len(valid_members),
                                    "leader_fitness": leader.fitness,
                                    "origin": "cluster_0_speciation"
                                })
                                self.logger.info(
                                    f"Speciation event! Created species {new_species.id} from {len(valid_members)} "
                                    f"Cluster 0 (reserves) individuals (filtered from {len(all_members)} candidates)"
                                )
                            else:
                                invalid_members = [m for m in all_members if m not in valid_members]
                                for invalid in invalid_members:
                                    if invalid in followers:
                                        followers.remove(invalid)
                                if pl_ind not in valid_members:
                                    self.logger.debug(
                                        f"Cluster 0 speciation: original potential leader {pl_ind.id} is outside "
                                        f"new leader's radius (new leader is {leader.id if 'leader' in locals() else 'unknown'})"
                                    )
                                self.logger.debug(
                                    f"Cluster 0 speciation: {len(all_members)} candidates but only {len(valid_members)} "
                                    f"within new leader's radius (need {self.min_island_size}), not creating species. "
                                    f"Invalid members ({len(invalid_members)}) remain in Cluster 0 for reassignment."
                                )
                    assigned = True
            
            if not assigned:
                potential_leaders[ind.id] = (None, ind.embedding, ind.phenotype, ind, [])
        
        if individuals_to_remove:
            self.remove_batch(individuals_to_remove)
            self.logger.debug(f"Removed {len(individuals_to_remove)} individuals from Cluster 0 (formed {len(new_species_list)} new species)")
        
        return new_species_list
    
    
    def get_best(self, n: int = 1) -> List[Individual]:
        
        sorted_members = sorted(self.members, key=lambda x: x.individual.fitness, reverse=True)
        return [lm.individual for lm in sorted_members[:n]]
    
    def pop_best(self) -> Optional[Individual]:
        
        if not self.members:
            return None
        best_idx = max(range(len(self.members)), key=lambda i: self.members[i].individual.fitness)
        return self.members.pop(best_idx).individual
    
    def clear(self) -> None:
        
        self.members = []
    
    def to_dict(self) -> Dict:
        
        return {
            "cluster_id": CLUSTER_0_ID,
            "size": self.size,
            "max_capacity": self.max_capacity,
            "speciation_events": self.speciation_events[-10:]
        }

