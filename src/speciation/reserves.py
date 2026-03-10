"""
reserves.py

Cluster 0 (reserves buffer) management for speciation.
Holding area for individuals that don't fit existing species (ID >= 1).
Cluster 0 is a special cluster with ID 0 that holds outliers and removed individuals.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING

from .species import Individual, Species, generate_species_id
from .distance import ensemble_distance, ensemble_distances_batch

if TYPE_CHECKING:
    from .config import SpeciationConfig

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


# Cluster 0 ID (reserved, species IDs start from 1)
CLUSTER_0_ID = 0


@dataclass
class Cluster0Individual:
    """
    Wrapper for individuals in Cluster 0 (reserves).
    
    Cluster 0 individuals are outliers that don't fit existing species.
    They are preserved for potential speciation if enough similar individuals accumulate.
    
    Attributes:
        individual: The Individual instance in Cluster 0 (reserves)
        entered_at: Generation when individual entered Cluster 0 (reserves)
    """
    individual: Individual
    entered_at: int
    
    def __hash__(self):
        return hash(self.individual.id)
    
    def __eq__(self, other):
        return isinstance(other, Cluster0Individual) and self.individual.id == other.individual.id




class Cluster0:
    """
    Cluster 0 (reserves) for individuals that don't fit existing species.
    
    Cluster 0 (ID=0) is a special holding area for:
    1. High-fitness outliers that are semantically distant from all species leaders
    2. Individuals removed from species when they exceed max capacity (100)
    3. Individuals that may form new species if enough similar ones accumulate
    
    Key features:
    - Max capacity: Limited to cluster0_max_capacity (default 1000) individuals.
      When over capacity, selection is controlled by config.cluster0_selection:
      "nsga2" (default) uses NSGA-II with diversity first, then toxicity;
      "toxicity_only" uses legacy sort by fitness.
    - Speciation detection: When Cluster 0 individuals form cohesive clusters, they create new species
    - Leader-follower clustering: Uses leader-follower algorithm to find groups in Cluster 0
    
    This preserves diversity by giving novel high-fitness solutions a chance to
    form new species rather than being discarded.
    """
    
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
        """
        Initialize Cluster 0 (reserves).
        
        Args:
            min_cluster_size: Minimum cluster size for speciation
            theta_sim: Semantic distance threshold for clustering (also used as species radius)
            max_capacity: Maximum number of individuals in Cluster 0 (reserves) (default: 1000)
            min_island_size: Minimum species size required (used to verify species after leader selection)
            w_genotype: Weight for genotype distance in ensemble distance
            w_phenotype: Weight for phenotype distance in ensemble distance
            logger: Optional logger instance
        """
        self.members: List[Cluster0Individual] = []
        self.min_cluster_size = min_cluster_size
        self.theta_sim = theta_sim
        self.max_capacity = max_capacity
        self.min_island_size = min_island_size
        self.w_genotype = w_genotype
        self.w_phenotype = w_phenotype
        self.logger = logger or get_logger("Cluster0")
        self.speciation_events: List[Dict] = []  # Track speciation events from Cluster 0 (reserves)
    
    @property
    def size(self) -> int:
        """Number of individuals in Cluster 0 (reserves)."""
        return len(self.members)
    
    @property
    def individuals(self) -> List[Individual]:
        """Get all individuals (without TTL wrapper)."""
        return [lm.individual for lm in self.members]
    
    def add(self, individual: Individual, generation: int) -> None:
        """
        Add individual to Cluster 0 (reserves).
        
        Sets the individual's species_id to 0 (Cluster 0).
        
        Args:
            individual: Individual to add
            generation: Current generation number
        """
        # Avoid duplicates
        for lm in self.members:
            if lm.individual.id == individual.id:
                return
        
        # Mark individual as belonging to Cluster 0 (reserves)
        individual.species_id = CLUSTER_0_ID
        
        self.members.append(Cluster0Individual(
            individual=individual,
            entered_at=generation
        ))
    
    def add_batch(self, individuals: List[Individual], generation: int) -> None:
        """Add multiple individuals to Cluster 0 (reserves)."""
        for ind in individuals:
            self.add(ind, generation)
    
    def remove(self, individual: Individual) -> bool:
        """Remove an individual from Cluster 0 (reserves)."""
        for i, lm in enumerate(self.members):
            if lm.individual.id == individual.id:
                self.members.pop(i)
                return True
        return False
    
    def remove_batch(self, individuals: List[Individual]) -> int:
        """Remove multiple individuals from Cluster 0 (reserves). Returns count removed."""
        ids = {ind.id for ind in individuals}
        original = len(self.members)
        self.members = [lm for lm in self.members if lm.individual.id not in ids]
        return original - len(self.members)
    
    def check_speciation(self, current_generation: int) -> List[Species]:
        """
        Check if Cluster 0 (reserves) individuals can form new species via leader-follower clustering.
        
        This is the key mechanism for creating new species from Cluster 0. When enough
        similar high-fitness individuals accumulate, they can form cohesive clusters
        and create new species.
        
        Algorithm (Leader-Follower - Optimized to form all species in one pass):
        1. Check if enough individuals in Cluster 0 (>= min_cluster_size)
        2. Sort individuals by fitness (descending)
        3. Process all individuals:
           - First individual becomes potential leader
           - For each subsequent individual:
             * Check distance to all potential leaders
             * If within theta_sim, add as follower
             * When leader has (min_cluster_size - 1) followers, form species immediately
           - Continue processing remaining individuals even after forming species
        4. Remove all formed species members from Cluster 0
        5. Return list of all new species formed in this pass
        
        Args:
            current_generation: Current generation number
        
        Returns:
            List of new Species formed (empty list if none formed)
        """
        # Check minimum size requirement
        if len(self.members) < self.min_cluster_size:
            return []
        
        # Filter to individuals with embeddings
        individuals = [lm.individual for lm in self.members if lm.individual.embedding is not None]
        if len(individuals) < self.min_cluster_size:
            return []
        
        # Sort by fitness (descending) - highest fitness processed first
        sorted_individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)
        
        # Potential leaders: Dict mapping leader_id -> (species_id_or_None, embedding, phenotype, Individual, followers_list)
        # species_id is None until (min_cluster_size - 1) followers are found, then it's assigned
        potential_leaders: Dict[str, Tuple[Optional[int], np.ndarray, Optional[np.ndarray], Individual, List[Individual]]] = {}
        
        # Track all new species formed in this pass
        new_species_list: List[Species] = []
        # Track all individuals to remove (from formed species)
        individuals_to_remove: List[Individual] = []
        
        # First individual becomes first potential leader
        first = sorted_individuals[0]
        potential_leaders[first.id] = (None, first.embedding, first.phenotype, first, [])
        remaining_individuals = sorted_individuals[1:]
        
        # Process remaining individuals
        for ind in remaining_individuals:
            assigned = False
            min_dist = float('inf')
            nearest_leader_id = None
            
            # Check against all potential leaders (excluding those that already formed species)
            active_leaders = {pl_id: data for pl_id, data in potential_leaders.items() if data[0] is None}
            if active_leaders:
                # Collect all leader embeddings and phenotypes
                leader_embeddings = []
                leader_phenotypes = []
                leader_ids = []
                for pl_id, (_, pl_emb, pl_pheno, _, _) in active_leaders.items():
                    leader_ids.append(pl_id)
                    leader_embeddings.append(pl_emb)
                    leader_phenotypes.append(pl_pheno)
                
                if len(leader_embeddings) > 1:
                    # Vectorized distance computation
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
                
                # If within threshold, add as follower
                if nearest_leader_id is not None and min_dist < self.theta_sim:
                    pl_species_id, pl_emb, pl_pheno, pl_ind, followers = potential_leaders[nearest_leader_id]
                    
                    if pl_species_id is None:
                        # Add as follower (tracked but no species yet)
                        followers.append(ind)
                        # Check if we've reached minimum size
                        total_size = 1 + len(followers)  # leader + followers
                        if total_size >= self.min_island_size:
                            # Minimum size reached! Create the species now
                            new_species_id = generate_species_id()
                            # Determine leader (highest fitness among leader + followers)
                            all_members = [pl_ind] + followers
                            leader = max(all_members, key=lambda x: x.fitness)
                            
                            # After choosing the new leader, verify all members are within the leader's radius
                            # So species are only created with members within the radius threshold;
                            # we don't remove them in the radius cleanup phase.
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
                                    self.w_genotype, self.w_phenotype
                                )
                                
                                if dist < self.theta_sim:
                                    valid_members.append(member)
                            
                            # Only create species if it has at least min_island_size valid members
                            if len(valid_members) >= self.min_island_size:
                                other_members = [m for m in valid_members if m.id != leader.id]
                                
                                new_species = Species(
                                    id=new_species_id,
                                    leader=leader,
                                    members=valid_members,
                                    radius=self.theta_sim,  # Constant radius
                                    created_at=current_generation,
                                    last_improvement=current_generation,
                                    cluster_origin="natural",  # Formed naturally from Cluster 0
                                    parent_ids=None,
                                    leader_distance=0.0  # New species leader is reference point
                                )
                                
                                # Mark this potential leader as having formed a species
                                potential_leaders[nearest_leader_id] = (new_species_id, pl_emb, pl_pheno, pl_ind, followers)
                                
                                # Track for removal and event logging (only valid members)
                                individuals_to_remove.extend(valid_members)
                                new_species_list.append(new_species)
                                
                                # Track speciation event
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
                                # Not enough valid members after filtering - don't create species
                                # Invalid members stay in Cluster 0 and can be reassigned to other potential leaders
                                # or become new potential leaders themselves
                                invalid_members = [m for m in all_members if m not in valid_members]
                                # Remove invalid members from followers list so they're not counted for future attempts
                                # They'll stay in Cluster 0 and be processed again in the loop
                                for invalid in invalid_members:
                                    if invalid in followers:
                                        followers.remove(invalid)
                                # Check if the original potential leader (pl_ind) is invalid
                                # Note: pl_ind is the original potential leader, not a follower, so it's never in followers
                                if pl_ind not in valid_members:
                                    # Original potential leader is outside new leader's radius
                                    # This can happen if a different member became the new leader and pl_ind is too far
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
            
            # If not assigned to any potential leader, become a new potential leader
            if not assigned:
                potential_leaders[ind.id] = (None, ind.embedding, ind.phenotype, ind, [])
        
        # Remove all formed species members from Cluster 0 in one batch
        if individuals_to_remove:
            self.remove_batch(individuals_to_remove)
            self.logger.debug(f"Removed {len(individuals_to_remove)} individuals from Cluster 0 (formed {len(new_species_list)} new species)")
        
        return new_species_list
    
    
    def get_best(self, n: int = 1) -> List[Individual]:
        """Get top N individuals by fitness."""
        sorted_members = sorted(self.members, key=lambda x: x.individual.fitness, reverse=True)
        return [lm.individual for lm in sorted_members[:n]]
    
    def pop_best(self) -> Optional[Individual]:
        """Remove and return the best individual."""
        if not self.members:
            return None
        best_idx = max(range(len(self.members)), key=lambda i: self.members[i].individual.fitness)
        return self.members.pop(best_idx).individual
    
    def clear(self) -> None:
        """Clear all members from Cluster 0 (reserves)."""
        self.members = []
    
    def to_dict(self) -> Dict:
        """Serialize Cluster 0 (reserves) metadata to dictionary for JSON storage.
        
        Note: Full member data (genomes) is NOT stored here since reserves.json 
        already contains the complete genome data for cluster 0 members.
        This only stores metadata: size, capacity, and speciation events.
        """
        return {
            "cluster_id": CLUSTER_0_ID,
            "size": self.size,
            "max_capacity": self.max_capacity,
            "speciation_events": self.speciation_events[-10:]
        }

