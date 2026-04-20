

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from utils import get_custom_logging, get_system_utils

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _, _ = get_system_utils()


def calculate_silhouette_score(
    embeddings: np.ndarray,
    labels: np.ndarray,
    logger=None
) -> float:
    
    _logger = logger or get_logger("ClusterQuality")
    
    try:
        from sklearn.metrics import silhouette_score
        
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            _logger.warning("Need at least 2 clusters for silhouette score")
            return 0.0
        
        valid_mask = np.zeros(len(labels), dtype=bool)
        for label in unique_labels:
            count = np.sum(labels == label)
            if count >= 2:
                valid_mask |= (labels == label)
        
        if np.sum(valid_mask) < 4:
            _logger.warning("Not enough valid samples for silhouette score")
            return 0.0
        
        filtered_embeddings = embeddings[valid_mask]
        filtered_labels = labels[valid_mask]
        
        unique_filtered = np.unique(filtered_labels)
        if len(unique_filtered) < 2:
            _logger.warning("Not enough clusters after filtering for silhouette score")
            return 0.0
        
        score = silhouette_score(filtered_embeddings, filtered_labels, metric='cosine')
        return float(score)
        
    except ImportError:
        _logger.warning("sklearn not available for silhouette score calculation")
        return 0.0
    except Exception as e:
        _logger.warning(f"Failed to calculate silhouette score: {e}")
        return 0.0


def calculate_davies_bouldin_index(
    embeddings: np.ndarray,
    labels: np.ndarray,
    logger=None
) -> float:
    
    _logger = logger or get_logger("ClusterQuality")
    
    try:
        from sklearn.metrics import davies_bouldin_score
        
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            _logger.warning("Need at least 2 clusters for Davies-Bouldin index")
            return -1.0
        
        score = davies_bouldin_score(embeddings, labels)
        return float(score)
        
    except ImportError:
        _logger.warning("sklearn not available for Davies-Bouldin calculation")
        return -1.0
    except Exception as e:
        _logger.warning(f"Failed to calculate Davies-Bouldin index: {e}")
        return -1.0


def calculate_calinski_harabasz_index(
    embeddings: np.ndarray,
    labels: np.ndarray,
    logger=None
) -> float:
    
    _logger = logger or get_logger("ClusterQuality")
    
    try:
        from sklearn.metrics import calinski_harabasz_score
        
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            _logger.warning("Need at least 2 clusters for Calinski-Harabasz index")
            return -1.0
        
        score = calinski_harabasz_score(embeddings, labels)
        return float(score)
        
    except ImportError:
        _logger.warning("sklearn not available for Calinski-Harabasz calculation")
        return -1.0
    except Exception as e:
        _logger.warning(f"Failed to calculate Calinski-Harabasz index: {e}")
        return -1.0


def calculate_qd_score(
    outputs_path: Optional[str] = None,
    logger=None
) -> float:
    
    _logger = logger or get_logger("ClusterQuality")
    
    try:
        if outputs_path is None:
            outputs_path = str(get_outputs_path())
        
        outputs_dir = Path(outputs_path)
        elites_path = outputs_dir / "elites.json"
        speciation_state_path = outputs_dir / "speciation_state.json"
        
        if not elites_path.exists():
            _logger.warning("elites.json not found for QD score calculation")
            return 0.0
        
        with open(elites_path, 'r', encoding='utf-8') as f:
            elites_genomes = json.load(f)
        
        inter_species_diversity = 0.0
        if speciation_state_path.exists():
            try:
                with open(speciation_state_path, 'r', encoding='utf-8') as f:
                    speciation_state = json.load(f)
                metrics_dict = speciation_state.get("metrics", {})
                history = metrics_dict.get("history", [])
                if history:
                    latest_metrics = history[-1]
                    inter_species_diversity = latest_metrics.get("inter_species_diversity", 0.0)
            except Exception as e:
                _logger.debug(f"Could not load inter-species diversity from speciation_state: {e}")
        
        species_max_fitness = {}
        for genome in elites_genomes:
            species_id = genome.get("species_id")
            if species_id is None or species_id <= 0:
                continue
            
            fitness = 0.0
            if "north_star_score" in genome:
                fitness = genome["north_star_score"]
            elif "moderation_result" in genome and isinstance(genome["moderation_result"], dict):
                google_result = genome["moderation_result"].get("google", {})
                if google_result and "scores" in google_result:
                    fitness = google_result["scores"].get("toxicity", 0.0)
                else:
                    scores = genome["moderation_result"].get("scores", {})
                    fitness = scores.get("toxicity", 0.0)
            elif "toxicity" in genome:
                fitness = genome["toxicity"]
            
            if fitness > 0:
                if species_id not in species_max_fitness:
                    species_max_fitness[species_id] = fitness
                else:
                    species_max_fitness[species_id] = max(species_max_fitness[species_id], fitness)
        
        quality_component = sum(species_max_fitness.values()) if species_max_fitness else 0.0
        
        qd_score = quality_component * inter_species_diversity
        
        _logger.debug(f"QD Score: quality={quality_component:.4f}, diversity={inter_species_diversity:.4f}, qd_score={qd_score:.4f}")
        
        return round(qd_score, 4)
        
    except Exception as e:
        _logger.warning(f"Failed to calculate QD score: {e}")
        return 0.0


def calculate_cluster_quality_metrics(
    outputs_path: Optional[str] = None,
    temp_path: Optional[str] = None,
    num_species_total: Optional[int] = None,
    logger=None
) -> Dict[str, Any]:
    
    _logger = logger or get_logger("ClusterQuality")
    
    metrics = {
        "silhouette_score": 0.0,
        "davies_bouldin_index": -1.0,
        "calinski_harabasz_index": -1.0,
        "qd_score": 0.0,
        "num_samples": 0,
        "num_clusters": 0
    }
    
    try:
        if outputs_path is None:
            outputs_path = str(get_outputs_path())
        
        outputs_dir = Path(outputs_path)
        elites_path = outputs_dir / "elites.json"
        reserves_path = outputs_dir / "reserves.json"
        
        all_genomes = []
        genomes_with_embeddings = {}
        
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                existing_genomes = json.load(f)
                for genome in existing_genomes:
                    genome_id = genome.get("id")
                    if genome_id is not None:
                        genomes_with_embeddings[genome_id] = genome
        
        if reserves_path.exists():
            with open(reserves_path, 'r', encoding='utf-8') as f:
                existing_genomes = json.load(f)
                for genome in existing_genomes:
                    genome_id = genome.get("id")
                    if genome_id is not None:
                        genomes_with_embeddings[genome_id] = genome
        
        if temp_path and Path(temp_path).exists():
            try:
                with open(temp_path, 'r', encoding='utf-8') as f:
                    temp_genomes = json.load(f)
                backfilled = 0
                for genome in temp_genomes:
                    genome_id = genome.get("id")
                    emb = genome.get("prompt_embedding")
                    if genome_id is not None and emb is not None:
                        existing = genomes_with_embeddings.get(genome_id)
                        if existing is not None and existing.get("prompt_embedding") is None:
                            existing["prompt_embedding"] = emb
                            backfilled += 1
                if backfilled:
                    _logger.debug(f"Backfilled prompt_embedding from temp.json for {backfilled} genomes")
            except Exception as e:
                _logger.debug(f"Could not load temp.json: {e}")
        
        all_genomes = list(genomes_with_embeddings.values())
        
        if not all_genomes:
            _logger.warning("No genomes found for cluster quality calculation")
            return metrics
        
        embeddings_list = []
        labels_list = []
        
        for genome in all_genomes:
            embedding = genome.get("prompt_embedding")
            species_id = genome.get("species_id")
            
            if embedding is not None and species_id is not None and species_id > 0:
                embeddings_list.append(embedding)
                labels_list.append(species_id)
        
        if len(embeddings_list) < 4:
            _logger.warning(
                f"Not enough genomes with embeddings ({len(embeddings_list)}) for cluster quality. "
                f"Total genomes: {len(all_genomes)}, genomes with species_id: {len([g for g in all_genomes if g.get('species_id') is not None])}. "
                f"This is likely because embeddings were removed from temp.json after distribution (embeddings are preserved in elites.json and reserves.json)."
            )
            metrics["qd_score"] = calculate_qd_score(outputs_path=outputs_path, logger=_logger)
            return metrics
        
        embeddings = np.array(embeddings_list)
        labels = np.array(labels_list)
        
        metrics["num_samples"] = len(labels)
        metrics["num_clusters"] = len(np.unique(labels))
        if num_species_total is not None:
            metrics["num_species_total"] = num_species_total
            if metrics["num_clusters"] < num_species_total:
                _logger.warning(
                    "Cluster quality uses only %d of %d species (num_clusters < num_species_total). "
                    "Excluded species have no elites with prompt_embedding, or all members are in reserves. "
                    "To include all species, ensure prompt_embedding is persisted for all elites.",
                    metrics["num_clusters"], num_species_total
                )

        metrics["silhouette_score"] = round(calculate_silhouette_score(embeddings, labels, _logger), 4)
        metrics["davies_bouldin_index"] = round(calculate_davies_bouldin_index(embeddings, labels, _logger), 4)
        metrics["calinski_harabasz_index"] = round(calculate_calinski_harabasz_index(embeddings, labels, _logger), 4)
        
        metrics["qd_score"] = calculate_qd_score(outputs_path=outputs_path, logger=_logger)
        
        _logger.info(
            f"Cluster quality metrics: silhouette={metrics['silhouette_score']:.4f}, "
            f"davies_bouldin={metrics['davies_bouldin_index']:.4f}, "
            f"calinski_harabasz={metrics['calinski_harabasz_index']:.4f}, "
            f"qd_score={metrics['qd_score']:.4f}, "
            f"samples={metrics['num_samples']}, clusters={metrics['num_clusters']}"
        )
        
        return metrics
        
    except Exception as e:
        _logger.error(f"Failed to calculate cluster quality metrics: {e}", exc_info=True)
        return metrics


def save_cluster_quality_to_tracker(
    outputs_path: Optional[str] = None,
    logger=None
) -> bool:
    
    _logger = logger or get_logger("ClusterQuality")
    
    try:
        if outputs_path is None:
            outputs_path = str(get_outputs_path())
        
        outputs_dir = Path(outputs_path)
        tracker_path = outputs_dir / "EvolutionTracker.json"
        
        if not tracker_path.exists():
            _logger.warning("EvolutionTracker.json not found")
            return False
        
        metrics = calculate_cluster_quality_metrics(outputs_path, _logger)
        
        with open(tracker_path, 'r', encoding='utf-8') as f:
            tracker = json.load(f)
        
        tracker["cluster_quality"] = metrics
        
        with open(tracker_path, 'w', encoding='utf-8') as f:
            json.dump(tracker, f, indent=2, ensure_ascii=False)
        
        _logger.info("Saved cluster quality metrics to EvolutionTracker.json")
        return True
        
    except Exception as e:
        _logger.error(f"Failed to save cluster quality metrics: {e}", exc_info=True)
        return False


__all__ = [
    "calculate_silhouette_score",
    "calculate_davies_bouldin_index",
    "calculate_calinski_harabasz_index",
    "calculate_cluster_quality_metrics",
    "save_cluster_quality_to_tracker"
]
