

import json
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

from .species import Individual, Species
from .distance import ensemble_distance

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


@dataclass
class GenerationMetrics:
    """Metrics for a single generation."""
    generation: int
    species_count: int
    total_population: int
    reserves_size: int
    best_fitness: float
    avg_fitness: float
    fitness_std: float
    speciation_events: int = 0
    merge_events: int = 0
    extinction_events: int = 0
    inter_species_diversity: float = 0.0
    intra_species_diversity: float = 0.0
    cluster_quality: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        result = {
            "generation": self.generation, "species_count": self.species_count,
            "total_population": self.total_population, "reserves_size": self.reserves_size,
            "best_fitness": round(self.best_fitness, 4), "avg_fitness": round(self.avg_fitness, 4),
            "fitness_std": round(self.fitness_std, 4),
            "speciation_events": self.speciation_events,
            "merge_events": self.merge_events, "extinction_events": self.extinction_events,
            "inter_species_diversity": round(self.inter_species_diversity, 4),
            "intra_species_diversity": round(self.intra_species_diversity, 4)
        }
        if self.cluster_quality:
            result["cluster_quality"] = self.cluster_quality
        return result


class SpeciationMetricsTracker:
    """Tracks metrics over evolution."""
    
    def __init__(self, logger=None):
        self.logger = logger or get_logger("SpeciationMetrics")
        self.history: List[GenerationMetrics] = []
        self.total_speciation = 0
        self.total_merges = 0
        self.total_extinctions = 0
    
    def record_generation(self, generation: int, species: Dict[int, Species], reserves_size: int = 0,
                          speciation_events: int = 0, merge_events: int = 0,
                          extinction_events: int = 0, cluster0=None, 
                          elites_path: Optional[str] = None, reserves_path: Optional[str] = None) -> GenerationMetrics:
        
        from pathlib import Path
        import json
        
        species_count = len(species)
        total_pop = 0
        all_fitness = []
        elites_genomes = []
        
        if elites_path and Path(elites_path).exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_genomes = json.load(f)
            
            unique_species_ids = set()
            for genome in elites_genomes:
                species_id = genome.get("species_id")
                if species_id is not None and species_id > 0:
                    unique_species_ids.add(species_id)
            species_count = len(unique_species_ids)
            self.logger.debug(f"Calculated species_count from elites.json: {species_count} species (in-memory had {len(species)})")
            
            total_pop += len(elites_genomes)
            from utils.population_io import _extract_north_star_score
            for genome in elites_genomes:
                fitness = _extract_north_star_score(genome, "toxicity")
                if fitness > 0:
                    all_fitness.append(float(fitness))
        else:
            if elites_path:
                self.logger.error(f"elites.json not found at {elites_path} - metrics will be inaccurate")
            total_pop = sum(sp.size for sp in species.values())
            all_fitness = [m.fitness for sp in species.values() for m in sp.members]
        
        actual_reserves_size = reserves_size
        if reserves_path and Path(reserves_path).exists():
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves_genomes = json.load(f)
            actual_reserves_size = len(reserves_genomes)
            total_pop += actual_reserves_size
            from utils.population_io import _extract_north_star_score
            for genome in reserves_genomes:
                fitness = _extract_north_star_score(genome, "toxicity")
                if fitness > 0:
                    all_fitness.append(float(fitness))
        else:
            if reserves_path:
                self.logger.warning(f"reserves.json not found at {reserves_path}, using parameter value (metrics may be inaccurate)")
            total_pop += reserves_size
            if cluster0 is not None and hasattr(cluster0, 'individuals'):
                all_fitness.extend([ind.fitness for ind in cluster0.individuals])
        
        best = max(all_fitness) if all_fitness else 0.0
        avg = np.mean(all_fitness) if all_fitness else 0.0
        std = np.std(all_fitness) if all_fitness else 0.0
        
        inter_div, intra_div = compute_diversity_metrics(species, w_genotype=0.7, w_phenotype=0.3, elites_path=elites_path)
        
        cluster_quality = None
        if species_count > 1 and total_pop >= 4:
            try:
                from utils.cluster_quality import calculate_cluster_quality_metrics
                if elites_path:
                    from pathlib import Path
                    outputs_path = str(Path(elites_path).parent)
                    temp_path = str(Path(outputs_path) / "temp.json")
                    temp_path_obj = Path(temp_path)
                    use_temp = temp_path_obj.exists()
                    if use_temp:
                        try:
                            with open(temp_path_obj, 'r', encoding='utf-8') as f:
                                temp_genomes = json.load(f)
                            has_embeddings = any(g.get("prompt_embedding") is not None for g in temp_genomes)
                            if not has_embeddings:
                                use_temp = False
                        except Exception:
                            use_temp = False
                    cluster_quality = calculate_cluster_quality_metrics(
                        outputs_path=outputs_path,
                        temp_path=temp_path if use_temp else None,
                        num_species_total=len(species),
                        logger=self.logger
                    )
            except Exception as e:
                self.logger.debug(f"Failed to calculate cluster quality metrics: {e}")
        
        self.total_speciation += speciation_events
        self.total_merges += merge_events
        self.total_extinctions += extinction_events
        
        metrics = GenerationMetrics(
            generation=generation, species_count=species_count, total_population=total_pop,
            reserves_size=actual_reserves_size, 
            best_fitness=round(float(best), 4), 
            avg_fitness=round(float(avg), 4),
            fitness_std=round(float(std), 4), 
            speciation_events=speciation_events,
            merge_events=merge_events, extinction_events=extinction_events,
            inter_species_diversity=round(float(inter_div), 4), 
            intra_species_diversity=round(float(intra_div), 4),
            cluster_quality=cluster_quality
        )
        
        self.history.append(metrics)
        return metrics
    
    def get_summary(self) -> Dict:
        if not self.history:
            return {}
        return {
            "total_generations": len(self.history),
            "final_species_count": self.history[-1].species_count,
            "best_fitness_ever": max(m.best_fitness for m in self.history),
            "total_speciation_events": self.total_speciation,
            "total_merge_events": self.total_merges,
            "total_extinction_events": self.total_extinctions
        }
    
    def to_dict(self) -> Dict:
        return {"history": [m.to_dict() for m in self.history], "summary": self.get_summary()}
    
    @classmethod
    def from_dict(cls, metrics_dict: Dict, logger=None) -> "SpeciationMetricsTracker":
        
        tracker = cls(logger=logger)
        
        if "history" in metrics_dict:
            for gen_dict in metrics_dict["history"]:
                metrics = GenerationMetrics(
                    generation=gen_dict.get("generation", 0),
                    species_count=gen_dict.get("species_count", 0),
                    total_population=gen_dict.get("total_population", 0),
                    reserves_size=gen_dict.get("reserves_size", 0),
                    best_fitness=gen_dict.get("best_fitness", 0.0),
                    avg_fitness=gen_dict.get("avg_fitness", 0.0),
                    fitness_std=gen_dict.get("fitness_std", 0.0),
                    speciation_events=gen_dict.get("speciation_events", 0),
                    merge_events=gen_dict.get("merge_events", 0),
                    extinction_events=gen_dict.get("extinction_events", 0),
                    inter_species_diversity=gen_dict.get("inter_species_diversity", 0.0),
                    intra_species_diversity=gen_dict.get("intra_species_diversity", 0.0),
                    cluster_quality=gen_dict.get("cluster_quality")
                )
                tracker.history.append(metrics)
        
        if "summary" in metrics_dict:
            summary = metrics_dict["summary"]
            tracker.total_speciation = summary.get("total_speciation_events", 0)
            tracker.total_merges = summary.get("total_merge_events", 0)
            tracker.total_extinctions = summary.get("total_extinction_events", 0)
        else:
            tracker.total_speciation = sum(m.speciation_events for m in tracker.history)
            tracker.total_merges = sum(m.merge_events for m in tracker.history)
            tracker.total_extinctions = sum(m.extinction_events for m in tracker.history)
        
        return tracker


def compute_diversity_metrics(species: Dict[int, Species], w_genotype: float = 0.7, w_phenotype: float = 0.3,
                               elites_path: Optional[str] = None) -> tuple:
    
    from pathlib import Path
    import json
    
    if not species:
        return 0.0, 0.0
    
    species_list = [sp for sp in species.values() if sp.leader is not None]
    
    if not species_list:
        return 0.0, 0.0
    
    elites_genomes_by_species = {}
    logger = get_logger("DiversityMetrics")
    if elites_path and Path(elites_path).exists():
        try:
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_genomes = json.load(f)
            
            for genome in elites_genomes:
                species_id = genome.get("species_id")
                if species_id is not None and species_id > 0:
                    if species_id not in elites_genomes_by_species:
                        elites_genomes_by_species[species_id] = []
                    elites_genomes_by_species[species_id].append(genome)
            
            logger.debug(f"Loaded {len(elites_genomes_by_species)} species from elites.json for diversity calculation")
        except Exception as e:
            logger.warning(f"Failed to load elites.json for diversity metrics: {e}")
            pass
    else:
        logger.debug("No elites_path provided, using in-memory species members for diversity calculation")
    
    inter = []
    for i, sp1 in enumerate(species_list):
        for sp2 in species_list[i + 1:]:
            if sp1.leader.embedding is not None and sp2.leader.embedding is not None:
                try:
                    e1 = sp1.leader.embedding.copy()
                    e2 = sp2.leader.embedding.copy()
                    norm1 = np.linalg.norm(e1)
                    norm2 = np.linalg.norm(e2)
                    if not np.isclose(norm1, 1.0, atol=1e-5):
                        e1 = e1 / norm1
                    if not np.isclose(norm2, 1.0, atol=1e-5):
                        e2 = e2 / norm2
                    
                    dist = ensemble_distance(
                        e1, e2,
                        sp1.leader.phenotype, sp2.leader.phenotype,
                        w_genotype, w_phenotype
                    )
                    inter.append(dist)
                except Exception as e:
                    get_logger("DiversityMetrics").debug(f"Failed to compute inter-species distance: {e}")
                    pass
    
    inter_div = np.mean(inter) if inter else 0.0
    
    intra_divs = []
    for sp in species_list:
        members = []
        
        if sp.id in elites_genomes_by_species:
            from .species import Individual
            from .phenotype_distance import extract_phenotype_vector
            
            for genome in elites_genomes_by_species[sp.id]:
                embedding = None
                if "prompt_embedding" in genome:
                    emb_list = genome["prompt_embedding"]
                    if isinstance(emb_list, list):
                        embedding = np.array(emb_list, dtype=np.float32)
                    elif isinstance(emb_list, np.ndarray):
                        embedding = emb_list
                
                phenotype = extract_phenotype_vector(genome, logger=None)
                
                if embedding is not None:
                    norm = np.linalg.norm(embedding)
                    if not np.isclose(norm, 1.0, atol=1e-5):
                        embedding = embedding / norm
                    
                    from utils.population_io import _extract_north_star_score
                    fitness = _extract_north_star_score(genome, "toxicity")
                    
                    member = Individual(
                        id=genome.get("id", 0),
                        prompt=genome.get("prompt", ""),
                        fitness=fitness,
                        embedding=embedding,
                        phenotype=phenotype,
                        species_id=sp.id
                    )
                    members.append(member)
            
            logger.debug(f"Species {sp.id}: Loaded {len(members)} members from elites.json (had {len(sp.members)} in-memory)")
        else:
            members = [m for m in sp.members if m is not None and m.embedding is not None]
            logger.debug(f"Species {sp.id}: Using {len(members)} in-memory members (not found in elites.json)")
        
        if len(members) < 2:
            continue
        
        dists = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                try:
                    dist = ensemble_distance(
                        members[i].embedding, members[j].embedding,
                        members[i].phenotype, members[j].phenotype,
                        w_genotype, w_phenotype
                    )
                    dists.append(dist)
                except Exception as e:
                    get_logger("DiversityMetrics").debug(f"Failed to compute intra-species distance: {e}")
                    pass
        
        if dists:
            intra_divs.append(np.mean(dists))
    
    intra_div = np.mean(intra_divs) if intra_divs else 0.0
    
    return round(float(inter_div), 4), round(float(intra_div), 4)


def get_species_statistics(species: Dict[int, Species], elites_path: Optional[str] = None) -> Dict:
    
    from pathlib import Path
    import json
    
    if not species:
        return {
            "count": 0, 
            "total_population": 0,
            "sizes": [],
            "avg_size": 0.0,
            "fitness": {"global_best": 0.0, "global_avg": 0.0},
            "modes": {"DEFAULT": 0, "EXPLORE": 0, "EXPLOIT": 0}
        }
    
    sizes = [sp.size for sp in species.values()]
    if elites_path and Path(elites_path).exists():
        try:
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_genomes = json.load(f)
            species_sizes_from_file = {}
            for genome in elites_genomes:
                species_id = genome.get("species_id")
                if species_id is not None and species_id > 0:
                    species_sizes_from_file[species_id] = species_sizes_from_file.get(species_id, 0) + 1
            sizes = [species_sizes_from_file.get(sid, sp.size) for sid, sp in species.items()]
        except Exception:
            pass
    
    best_fitness_values = [sp.best_fitness for sp in species.values() if sp.size > 0]
    avg_fitness_values = [sp.avg_fitness for sp in species.values() if sp.size > 0]
    
    return {
        "count": len(species), "sizes": sizes, "avg_size": np.mean(sizes) if sizes else 0.0,
        "total_population": sum(sizes),
        "fitness": {
            "global_best": max(best_fitness_values) if best_fitness_values else 0.0,
            "global_avg": np.mean(avg_fitness_values) if avg_fitness_values else 0.0
        }
    }


def log_generation_summary(generation: int, species: Dict[int, Species], reserves_size: int = 0,
                           events: Dict[str, int] = None, logger=None, elites_path: Optional[str] = None) -> None:
    
    if logger is None:
        logger = get_logger("SpeciationMetrics")
    
    stats = get_species_statistics(species, elites_path=elites_path)
    events = events or {}
    event_str = ", ".join(f"{k}={v}" for k, v in events.items() if v > 0)
    
    logger.info(f"Gen {generation}: {stats['count']} species, {stats['total_population']} pop, "
                f"reserves={reserves_size}, best={stats['fitness']['global_best']:.4f}, "
                f"avg={stats['fitness']['global_avg']:.4f}" + (f", events: {event_str}" if event_str else ""))
