"""
species.py

Species data structures for speciation framework.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


# Mode switching is handled by parent selection, not speciation.


@dataclass
class Individual:
    """
    Represents an individual genome in the evolutionary population.
    
    An Individual is a wrapper around a genome (prompt) that includes:
    - Genotype: Semantic embedding for clustering (prompt embedding)
    - Phenotype: Response scores (8D toxicity scores)
    - Fitness score for selection
    - Species assignment for speciation
    
    This class bridges the gap between the raw genome format (dict)
    and the speciation framework's internal representation.
    
    Attributes:
        id: Unique identifier for the individual (matches genome ID)
        prompt: The text prompt (genome content)
        fitness: Fitness score (typically toxicity score, range [0, 1])
        embedding: L2-normalized semantic embedding vector (384-dim for all-MiniLM-L6-v2)
                   Used for genotype distance computation in clustering
        phenotype: Phenotype vector (8D response scores) for phenotype distance computation
        species_id: ID of the species this individual belongs to (None if unassigned, 0 for cluster 0)
        generation: Generation number when this individual was created
        genome_data: Original genome dictionary (for preserving metadata)
    """
    id: int
    prompt: str
    fitness: float = 0.0
    embedding: Optional[np.ndarray] = None
    phenotype: Optional[np.ndarray] = None
    species_id: Optional[int] = None
    generation: int = 0
    genome_data: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """
        Post-initialization: ensure embedding and phenotype are numpy arrays.
        
        Converts embedding and phenotype to numpy arrays if they're not already,
        which is required for distance computations.
        """
        if self.embedding is not None and not isinstance(self.embedding, np.ndarray):
            self.embedding = np.array(self.embedding)
        
        if self.phenotype is not None and not isinstance(self.phenotype, np.ndarray):
            self.phenotype = np.array(self.phenotype)
    
    @classmethod
    def from_genome(cls, genome: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> "Individual":
        """
        Create Individual instance from genome dictionary.
        
        Extracts fitness from various possible locations in the genome dict (priority order):
        - "north_star_score" (primary - pre-computed metric)
        - "moderation_result"["google"]["scores"]["toxicity"] (standard format)
        - "moderation_result"["scores"]["toxicity"]
        - "toxicity" (direct)
        - "scores"["toxicity"] (nested)
        
        Extracts embedding from "prompt_embedding" field if present (preferred),
        otherwise uses provided embedding parameter.
        
        Args:
            genome: Genome dictionary with prompt, id, fitness, and optionally prompt_embedding
            embedding: Optional pre-computed embedding (used only if prompt_embedding not in genome)
        
        Returns:
            Individual instance with extracted data
        
        Example:
            >>> genome = {"id": 1, "prompt": "test", "toxicity": 0.8, "prompt_embedding": [0.1, 0.2, ...]}
            >>> ind = Individual.from_genome(genome)
        """
        # Extract fitness using standardized method from utils.population_io
        from utils.population_io import _extract_north_star_score
        fitness = _extract_north_star_score(genome, "toxicity")
        
        # Extract embedding from genome if present (preferred over parameter)
        final_embedding = embedding
        if "prompt_embedding" in genome:
            # Convert list to numpy array
            embedding_list = genome["prompt_embedding"]
            if isinstance(embedding_list, list):
                final_embedding = np.array(embedding_list)
            elif isinstance(embedding_list, np.ndarray):
                final_embedding = embedding_list
        elif embedding is not None:
            final_embedding = embedding
        
        # Extract phenotype vector (response scores)
        from .phenotype_distance import extract_phenotype_vector
        # extract_phenotype_vector can handle None logger, so we pass None to avoid circular imports
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
        """
        Convert Individual back to genome dictionary format.
        
        Updates the original genome dict with speciation information:
        - species_id: Which species this individual belongs to (0 = cluster 0)
        - fitness: Current fitness value
        
        Returns:
            Genome dictionary with updated speciation metadata.
            This is used to update genomes after speciation processing.
        """
        # Start with original genome data or create minimal dict
        if self.genome_data:
            genome = self.genome_data.copy()
        else:
            # Create minimal dict with required fields
            genome = {"id": self.id}
            if hasattr(self, 'prompt') and self.prompt:
                genome["prompt"] = self.prompt
        
        # Add speciation metadata (overwrite if present)
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
    """
    Represents a species in the speciation framework.
    
    A Species is a cluster of semantically similar individuals that evolve together.
    Each species has:
    - A leader (highest fitness individual, defines species center)
    - Members (all individuals assigned to this species, max species_capacity)
    - A radius (constant, equal to theta_sim for all species)
    - Fitness tracking for stagnation detection
    - Origin tracking (how the species was created: merge/split/natural)
    
    Species evolve independently, can merge with similar species, can go frozen (extinct).
    
    Note: Species IDs start from 1. ID 0 is reserved for Cluster 0 (reserves).
    
    Species States:
    - "active": Normal operating state, participates in evolution
    - "frozen": Species frozen due to stagnation (species_stagnation exceeded), excluded from parent selection (but still alive)
    - "incubator": Species moved to cluster 0 (reserves), awaiting potential new species formation
    - "extinct": Species that merged with another (parent species become extinct, merged species is new)
    
    Attributes:
        id: Unique species identifier (1+, 0 reserved for cluster 0)
        leader: Leader individual (highest fitness, defines species center)
        members: List of all individuals in this species (includes leader, max species_capacity)
        radius: Semantic distance threshold for species membership (constant = theta_sim)
        stagnation: Incremented when species was selected as parent and max_fitness did not increase;
                    reset to 0 when max_fitness increased. Unchanged when not selected.
        max_fitness: Actual max over current members only (no merge with stored/previous values).
        species_state: "active", "frozen", "incubator", or "extinct" (only active species used for parent selection)
        created_at: Generation when this species was created
        last_improvement: Generation when fitness last improved
        fitness_history: List of best fitness values over time (for trend analysis)
        labels: Current c-TF-IDF labels (top 10 representative words)
        label_history: History of labels over generations (for tracking topic evolution)
        cluster_origin: How this species was created ("merge", "split", or "natural") - never None
        parent_ids: List of parent species IDs (None or [] for natural, [id1, id2] for merge, [id1] for split)
        leader_distance: Ensemble distance score of leader (0-1 normalized, for reference)
    """
    id: int
    leader: Individual
    members: List[Individual] = field(default_factory=list)
    radius: float = 0.4  # Constant radius (theta_sim), no dynamic adjustment
    stagnation: int = 0  # Generations without max_fitness improvement
    max_fitness: float = 0.0  # Current maximum fitness in this species
    species_state: str = "active"  # "active", "frozen", "incubator", or "extinct"
    created_at: int = 0
    last_improvement: int = 0
    fitness_history: List[float] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)  # Current c-TF-IDF labels (10 words)
    label_history: List[Dict[str, Any]] = field(default_factory=list)  # Label history per generation
    cluster_origin: str = "natural"  # "merge", "split", or "natural" - never None
    parent_ids: Optional[List[int]] = None  # Parent species IDs: None/[] for natural, [id1,id2] for merge, [id1] for split
    leader_distance: float = 0.0  # Ensemble distance score of leader (0-1 normalized)
    
    def __post_init__(self):
        """
        Post-initialization: ensure leader is in members and initialize tracking.
        
        - Ensures leader is always first in members list
        - Initializes fitness history with leader's fitness
        - Initializes max_fitness with leader's fitness
        - Assigns species_id to all members
        """
        # Leader must be in members list (at position 0)
        if self.leader not in self.members:
            self.members.insert(0, self.leader)
        # Initialize max_fitness if empty (but don't add to fitness_history here - record_fitness will do it)
        if self.max_fitness == 0.0 and self.leader:
            self.max_fitness = self.leader.fitness
        # Note: fitness_history is NOT initialized here to avoid duplicate entries
        # It will be populated by record_fitness() when called for each generation
        # Assign species_id to all members
        for member in self.members:
            member.species_id = self.id
    
    @property
    def size(self) -> int:
        """Number of individuals in this species."""
        return len(self.members)
    
    @property
    def best_fitness(self) -> float:
        """Highest fitness value in this species."""
        return max((m.fitness for m in self.members), default=0.0)
    
    @property
    def min_fitness(self) -> float:
        """Lowest fitness value in this species."""
        return min((m.fitness for m in self.members), default=0.0)
    
    @property
    def avg_fitness(self) -> float:
        """Average fitness across all members."""
        return sum(m.fitness for m in self.members) / len(self.members) if self.members else 0.0
    
    @property
    def leader_embedding(self) -> Optional[np.ndarray]:
        """Leader's embedding vector (for distance computations)."""
        return self.leader.embedding if self.leader else None
    
    def add_member(self, individual: Individual) -> None:
        """
        Add an individual to this species.
        
        Args:
            individual: Individual to add (species_id is automatically set)
        """
        individual.species_id = self.id
        if individual not in self.members:
            self.members.append(individual)
    
    def remove_member(self, individual: Individual) -> bool:
        """
        Remove an individual from this species.
        
        Args:
            individual: Individual to remove
        
        Returns:
            True if removed, False if not found
        """
        if individual in self.members:
            self.members.remove(individual)
            individual.species_id = None
            return True
        return False
    
    def record_fitness(self, generation: int, was_selected_as_parent: bool = False, max_fitness_increased: bool = False) -> None:
        """
        Record current best fitness and update fitness history.
        
        Stagnation (improvement is defined solely by max_fitness_increased, not fitness_history):
        - If max_fitness_increased is True -> stagnation = 0 (reset).
        - Else, if was_selected_as_parent is True -> stagnation += 1.
        - Else (not selected) -> stagnation unchanged.
        
        NOTE: This is called for ALL species, including frozen species. For frozen species,
        stagnation continues to increment if they were selected as parents.
        
        Args:
            generation: Current generation number
            was_selected_as_parent: Whether this species was selected as a parent in this generation
            max_fitness_increased: Whether this species' max_fitness increased this generation (vs snapshot before Phase 1)
        """
        current_best = self.best_fitness
        if max_fitness_increased:
            self.last_improvement = generation
            self.stagnation = 0
        elif was_selected_as_parent:
            self.stagnation += 1
        # else: stagnation unchanged
        # Append to history for trend analysis
        if not self.fitness_history or self.fitness_history[-1] != current_best or len(self.fitness_history) < generation + 1:
            self.fitness_history.append(current_best)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize species to dictionary for JSON storage.
        
        Includes leader embedding for state restoration across generations.
        Includes cluster_origin and parent_ids for origin tracking.
        Includes labels and label_history for semantic characterization.
        
        Note: Computed fields (size, best_fitness, avg_fitness) are not included
        as they can be derived from members and are not used by load_state().
        
        IMPORTANT: Members Storage:
        - Only member_ids (list of genome IDs) are saved, not full Individual objects
        - This is a storage optimization - full member data is in elites.json
        - Members are reconstructed lazily from elites.json when load_state() is called
        - The members list in memory may be empty even if size > 0 - this is EXPECTED
        """
        return {
            "id": self.id,
            "leader_id": self.leader.id,
            "leader_prompt": self.leader.prompt,
            "leader_embedding": self.leader.embedding.tolist() if self.leader.embedding is not None else None,
            "leader_fitness": round(float(self.leader.fitness), 4),
            "leader_distance": round(float(self.leader_distance), 4),  # Ensemble distance score (0-1) rounded for storage
            "member_ids": [m.id for m in self.members],
            "radius": round(float(self.radius), 4),
            "stagnation": self.stagnation,
            "max_fitness": round(float(self.max_fitness), 4),
            "min_fitness": round(self.min_fitness, 4),  # Lowest fitness in species
            "species_state": self.species_state,
            "created_at": self.created_at,
            "last_improvement": self.last_improvement,
            "fitness_history": self.fitness_history[-20:],
            "labels": self.labels,
            "label_history": self.label_history[-20:],  # Keep last 20 generations
            "cluster_origin": self.cluster_origin,
            "parent_ids": self.parent_ids,
        }
    
    def __repr__(self):
        origin_str = f", origin={self.cluster_origin}" if self.cluster_origin else ""
        return f"Species(id={self.id}, size={self.size}, best={self.best_fitness:.4f}{origin_str})"


class SpeciesIdGenerator:
    """
    Thread-safe species ID generator (singleton pattern).
    
    Ensures unique species IDs across the entire evolution run.
    IDs are sequential integers starting from 1 (ID 0 is reserved for cluster 0).
    """
    _current_id: int = 0  # Class variable (shared across all instances)
    
    @classmethod
    def next_id(cls) -> int:
        """
        Generate next unique species ID.
        
        Returns:
            Next sequential species ID (starts from 1, 0 reserved for cluster 0)
        """
        cls._current_id += 1
        return cls._current_id
    
    @classmethod
    def reset(cls, start: int = 0) -> None:
        """
        Reset ID counter (useful for testing or fresh runs).
        
        Args:
            start: Starting ID value (default: 0, so first ID will be 1)
        """
        cls._current_id = start
    
    @classmethod
    def set_min_id(cls, min_id: int) -> None:
        """
        Ensure ID counter is at least min_id (for state restoration).
        
        Used when loading species from saved state to avoid ID conflicts.
        
        Args:
            min_id: Minimum ID value to set (counter will be max of current and min_id)
        """
        cls._current_id = max(cls._current_id, min_id)


def generate_species_id() -> int:
    """
    Convenience function to generate a new species ID.
    
    Returns:
        New unique species ID (1+, 0 is reserved for cluster 0)
    """
    return SpeciesIdGenerator.next_id()
