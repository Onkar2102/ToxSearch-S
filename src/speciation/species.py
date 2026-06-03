

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional



@dataclass
class Individual:
    """Represents an individual genome in the evolutionary population. An Individual is a wrapper around a genome (prompt) that includes: - Genotype: Semantic embedding for clustering (prompt embedding) - Phenotype: Response scores (8D toxicity scores) - Fitness score for selection - Species assignment for speciation This class bridges the gap between the raw genome format (dict) and the speciation framework's internal representation. Attributes: id: Unique identifier for the individual (matches genome ID) prompt: The text prompt (genome content) fitness: Fitness score (typically toxicity score, range [0, 1]) embedding: L2-normalized semantic embedding vector (384-dim for all-MiniLM-L6-v2) Used for genotype distance computation in clustering phenotype: Phenotype vector (8D response scores) for phenotype distance computation species_id: ID of the species this individual belongs to (None if unassigned, 0 for cluster 0) generation: Generation number when this individual was created genome_data: Original genome dictionary (for preserving metadata)"""
    id: int
    prompt: str
    fitness: float = 0.0
    embedding: Optional[np.ndarray] = None
    phenotype: Optional[np.ndarray] = None
    species_id: Optional[int] = None
    generation: int = 0
    genome_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)
        
        if self.phenotype is not None and not isinstance(self.phenotype, np.ndarray):
            self.phenotype = np.array(self.phenotype)
    
    @classmethod
    def from_genome(cls, genome: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> "Individual":
        
        from utils.population_io import _extract_north_star_score
        from utils.evaluator_profiles import get_active_north_star
        fitness = _extract_north_star_score(genome, get_active_north_star())
        
        final_embedding = embedding
        if "prompt_embedding" in genome:
            embedding_list = genome["prompt_embedding"]
            if isinstance(embedding_list, list):
                final_embedding = np.array(embedding_list)
            elif isinstance(embedding_list, np.ndarray):
                final_embedding = embedding_list
        elif embedding is not None:
            final_embedding = embedding
        
        from .phenotype_distance import extract_phenotype_vector
        phenotype = extract_phenotype_vector(genome, logger=None)
        
        return cls(
            id=genome.get("id", 0),
            prompt=genome.get("prompt", ""),
            fitness=float(fitness) if fitness else 0.0,
            embedding=final_embedding,
            phenotype=phenotype,
            species_id=genome.get("species_id"),
            generation=genome.get("generation", 0),
            genome_data=genome
        )
    
    def to_genome(self) -> Dict[str, Any]:
        
        if self.genome_data:
            genome = self.genome_data.copy()
        else:
            genome = {"id": self.id}
            if hasattr(self, 'prompt') and self.prompt:
                genome["prompt"] = self.prompt
        
        genome["species_id"] = self.species_id
        genome["fitness"] = self.fitness
        return genome
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return isinstance(other, Individual) and self.id == other.id
    
    def __repr__(self):
        return f"Individual(id={self.id}, fitness={self.fitness:.4f}, species_id={self.species_id})"


@dataclass
class Species:
    """Represents a species in the speciation framework. A Species is a cluster of semantically similar individuals that evolve together. Each species has: - A leader (highest fitness individual, defines species center) - Members (all individuals assigned to this species, max species_capacity) - A radius (constant, equal to theta_sim for all species) - Fitness tracking for stagnation detection - Origin tracking (how the species was created: merge/split/natural) Species evolve independently, can merge with similar species, and can become frozen (excluded from parent selection but still alive). Note: Species IDs start from 1. ID 0 is reserved for Cluster 0 (reserves). Species States: - "active": Normal operating state, participates in evolution - "frozen": Species frozen due to stagnation (species_stagnation exceeded), excluded from parent selection (but still alive; can be reactivated if assigned a new leader) - "incubator": Species moved to cluster 0 (reserves), awaiting potential new species formation - "extinct": Species that merged with another (parent species become extinct, merged species is new) Attributes: id: Unique species identifier (1+, 0 reserved for cluster 0) leader: Leader individual (highest fitness, defines species center) members: List of all individuals in this species (includes leader, max species_capacity) radius: Semantic distance threshold for species membership (constant = theta_sim) stagnation: Incremented when species was selected as parent and max_fitness did not increase; reset to 0 when max_fitness increased. Unchanged when not selected. max_fitness: Actual max over current members only (no merge with stored/previous values). species_state: "active", "frozen", "incubator", or "extinct" (only active species used for parent selection) created_at: Generation when this species was created last_improvement: Generation when fitness last improved fitness_history: List of best fitness values over time (for trend analysis) labels: Current c-TF-IDF labels (top 10 representative words) label_history: History of labels over generations (for tracking topic evolution) cluster_origin: How this species was created ("merge", "split", or "natural") - never None parent_ids: List of parent species IDs (None or [] for natural, [id1, id2] for merge, [id1] for split) leader_distance: Ensemble distance score of leader (0-1 normalized, for reference)"""
    id: int
    leader: Individual
    members: List[Individual] = field(default_factory=list)
    radius: float = 0.4
    stagnation: int = 0
    max_fitness: float = 0.0
    species_state: str = "active"
    created_at: int = 0
    last_improvement: int = 0
    fitness_history: List[float] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    label_history: List[Dict[str, Any]] = field(default_factory=list)
    cluster_origin: str = "natural"
    parent_ids: Optional[List[int]] = None
    leader_distance: float = 0.0
    
    def __post_init__(self):
        
        if self.leader not in self.members:
            self.members.insert(0, self.leader)
        if self.max_fitness == 0.0 and self.leader:
            self.max_fitness = self.leader.fitness
        for member in self.members:
            member.species_id = self.id
    
    @property
    def size(self) -> int:
        
        return len(self.members)
    
    @property
    def best_fitness(self) -> float:
        
        return max((m.fitness for m in self.members), default=0.0)
    
    @property
    def min_fitness(self) -> float:
        
        return min((m.fitness for m in self.members), default=0.0)
    
    @property
    def avg_fitness(self) -> float:
        
        return sum(m.fitness for m in self.members) / len(self.members) if self.members else 0.0
    
    @property
    def leader_embedding(self) -> Optional[np.ndarray]:
        
        return self.leader.embedding if self.leader else None
    
    def add_member(self, individual: Individual) -> None:
        
        individual.species_id = self.id
        if individual not in self.members:
            self.members.append(individual)
    
    def remove_member(self, individual: Individual) -> bool:
        
        if individual in self.members:
            self.members.remove(individual)
            individual.species_id = None
            return True
        return False
    
    def record_fitness(self, generation: int, was_selected_as_parent: bool = False, max_fitness_increased: bool = False) -> None:
        
        current_best = self.best_fitness
        if max_fitness_increased:
            self.last_improvement = generation
            self.stagnation = 0
        elif was_selected_as_parent:
            self.stagnation += 1
        if not self.fitness_history or self.fitness_history[-1] != current_best or len(self.fitness_history) < generation + 1:
            self.fitness_history.append(current_best)
    
    def to_dict(self) -> Dict[str, Any]:
        
        return {
            "id": self.id,
            "leader_id": self.leader.id,
            "leader_prompt": self.leader.prompt,
            "leader_embedding": self.leader.embedding.tolist() if self.leader.embedding is not None else None,
            "leader_fitness": round(float(self.leader.fitness), 4),
            "leader_distance": round(float(self.leader_distance), 4),
            "member_ids": [m.id for m in self.members],
            "radius": round(float(self.radius), 4),
            "stagnation": self.stagnation,
            "max_fitness": round(float(self.max_fitness), 4),
            "min_fitness": round(self.min_fitness, 4),
            "species_state": self.species_state,
            "created_at": self.created_at,
            "last_improvement": self.last_improvement,
            "fitness_history": self.fitness_history[-20:],
            "labels": self.labels,
            "label_history": self.label_history[-20:],
            "cluster_origin": self.cluster_origin,
            "parent_ids": self.parent_ids,
        }
    
    def __repr__(self):
        origin_str = f", origin={self.cluster_origin}" if self.cluster_origin else ""
        return f"Species(id={self.id}, size={self.size}, best={self.best_fitness:.4f}{origin_str})"


class SpeciesIdGenerator:
    """Thread-safe species ID generator (singleton pattern). Ensures unique species IDs across the entire evolution run. IDs are sequential integers starting from 1 (ID 0 is reserved for cluster 0)."""
    _current_id: int = 0
    
    @classmethod
    def next_id(cls) -> int:
        
        cls._current_id += 1
        return cls._current_id
    
    @classmethod
    def reset(cls, start: int = 0) -> None:
        
        cls._current_id = start
    
    @classmethod
    def set_min_id(cls, min_id: int) -> None:
        
        cls._current_id = max(cls._current_id, min_id)


def generate_species_id() -> int:
    
    return SpeciesIdGenerator.next_id()
