

import json
import warnings
import numpy as np
from typing import List, Union, Optional
from pathlib import Path

from utils import get_custom_logging
from utils.device_utils import get_optimal_device
from utils import get_system_utils
warnings.filterwarnings('ignore', category=FutureWarning, message='.*np.object.*')

get_logger, _, _, PerformanceLogger = get_custom_logging()
_, _, _, get_outputs_path, _, _, _ = get_system_utils()


class EmbeddingModel:
    """Embedding model wrapper using sentence-transformers library. This class manages the semantic embedding model used for converting text prompts into high-dimensional vectors. The default model is all-MiniLM-L6-v2, which: - Produces 384-dimensional embeddings - Is fast and efficient (good for large batches) - Provides high-quality semantic representations - Supports L2-normalization for cosine distance computation Embeddings are L2-normalized by default, which ensures: - Cosine similarity = dot product (for normalized vectors) - Semantic distance = 1 - cosine_similarity - All vectors lie on the unit hypersphere The model is loaded on the optimal available device (CUDA > MPS > CPU)."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True
    ):
        
        self.logger = get_logger("EmbeddingModel")
        self.model_name = model_name
        self.normalize = normalize
        self.embedding_dim = 384
        
        if device is None:
            device = get_optimal_device()
        self.device = device
        
        self._model = None
        self._load_model()
        
    def _load_model(self) -> None:
        
        try:
            from sentence_transformers import SentenceTransformer
            
            self.logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'")
            self._model = SentenceTransformer(self.model_name, device=self.device)
            
            test_embedding = self._model.encode("test", normalize_embeddings=self.normalize)
            self.embedding_dim = len(test_embedding)
            
            self.logger.info(f"Embedding model loaded: dim={self.embedding_dim}, device={self.device}")
            
        except ImportError as e:
            self.logger.error(f"sentence-transformers not installed: {e}")
            raise
    
    def encode(
        self,
        text: Union[str, List[str]],
        batch_size: int = 64,
        show_progress: bool = False
        ) -> np.ndarray:
        
        if self._model is None:
            raise RuntimeError("Embedding model not loaded")
        
        return self._model.encode(
            text,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True
        )
    
    def __repr__(self) -> str:
        return f"EmbeddingModel(model_name='{self.model_name}', embedding_dim={self.embedding_dim})"


_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model(
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
    force_reload: bool = False
    ) -> EmbeddingModel:
    
    global _embedding_model
    
    if _embedding_model is None or force_reload:
        _embedding_model = EmbeddingModel(model_name=model_name, device=device, normalize=True)
    
    return _embedding_model


def compute_and_save_embeddings(
    temp_path: Optional[str] = None,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = False,
    logger=None) -> None:
    
    if logger is None:
        logger = get_logger("Embeddings")
    
    if temp_path is None:
        outputs_path = get_outputs_path()
        temp_path = str(outputs_path / "temp.json")
    
    temp_path_obj = Path(temp_path)
    if not temp_path_obj.exists():
        raise FileNotFoundError(f"Temp file not found: {temp_path}")
    
    logger.info(f"Computing embeddings for genomes in {temp_path}")
    
    with open(temp_path_obj, 'r', encoding='utf-8') as f:
        genomes = json.load(f)
    
    if not genomes:
        logger.warning("No genomes found in temp.json")
        return
    
    if all("prompt_embedding" in g for g in genomes):
        logger.info("Embeddings already exist in temp.json, skipping computation")
        return
    
    with PerformanceLogger(logger, "Embeddings: Compute and save", path=temp_path, num_genomes=len(genomes)):
        prompts = [g.get("prompt", "") for g in genomes]
        
        model = get_embedding_model(model_name=model_name)
        
        logger.info(f"Computing embeddings for {len(prompts)} prompts...")
        
        success_count = 0
        failure_count = 0
        embeddings = None
        
        try:
            embeddings = model.encode(
                prompts,
                batch_size=batch_size,
                show_progress=show_progress
            )
        except Exception as e:
            logger.error(f"Failed to compute embeddings batch: {e}", exc_info=True)
            raise
        
        for i, genome in enumerate(genomes):
            try:
                if embeddings is not None and i < len(embeddings):
                    embedding = embeddings[i]
                    if embedding is not None and len(embedding) > 0:
                        genome["prompt_embedding"] = embedding.tolist()
                        success_count += 1
                    else:
                        failure_count += 1
                        genome_id = genome.get("id", "unknown")
                        logger.warning(f"Embedding computation returned None/empty for genome {genome_id}")
                else:
                    failure_count += 1
                    genome_id = genome.get("id", "unknown")
                    logger.warning(f"Embedding missing for genome {genome_id} (index {i} out of range)")
            except Exception as e:
                failure_count += 1
                genome_id = genome.get("id", "unknown")
                logger.warning(f"Failed to add embedding for genome {genome_id}: {e}")
        
        logger.info(f"Embedding computation: {success_count} success, {failure_count} failures")
        if failure_count > 0:
            logger.warning(f"Population reduced by {failure_count} genomes due to embedding failures")
        
        with open(temp_path_obj, 'w', encoding='utf-8') as f:
            json.dump(genomes, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Successfully computed and saved embeddings for {len(genomes)} genomes")


def backfill_embeddings_for_genomes(
    genomes: List[dict],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = False,
    logger=None,
) -> int:
    
    if logger is None:
        logger = get_logger("Embeddings")
    need_emb = [
        (i, g) for i, g in enumerate(genomes)
        if isinstance(g, dict)
        and g.get("prompt") is not None
        and (not g.get("prompt_embedding") or (isinstance(g.get("prompt_embedding"), list) and len(g.get("prompt_embedding", [])) == 0))
    ]
    if not need_emb:
        return 0
    indices = [x[0] for x in need_emb]
    prompt_genomes = [x[1] for x in need_emb]
    prompts = [g.get("prompt", "") for g in prompt_genomes]
    model = get_embedding_model(model_name=model_name)
    try:
        embeddings = model.encode(prompts, batch_size=batch_size, show_progress=show_progress)
    except Exception as e:
        logger.error("Backfill embeddings failed: %s", e, exc_info=True)
        raise
    count = 0
    for j, (idx, genome) in enumerate(need_emb):
        if j < len(embeddings) and embeddings[j] is not None and len(embeddings[j]) > 0:
            genome["prompt_embedding"] = embeddings[j].tolist()
            count += 1
    logger.info("Backfilled prompt_embedding for %d/%d genomes", count, len(need_emb))
    return count


def remove_embeddings_from_temp(
    temp_path: Optional[str] = None,
    logger=None) -> None:
    
    if logger is None:
        logger = get_logger("Embeddings")
    
    if temp_path is None:
        outputs_path = get_outputs_path()
        temp_path = str(outputs_path / "temp.json")
    
    temp_path_obj = Path(temp_path)
    if not temp_path_obj.exists():
        raise FileNotFoundError(f"Temp file not found: {temp_path}")
    
    logger.info(f"Removing embeddings from genomes in {temp_path}")
    
    with open(temp_path_obj, 'r', encoding='utf-8') as f:
        genomes = json.load(f)
    
    if not genomes:
        logger.debug("temp.json is empty (already cleared by distribute_genomes) - skipping embedding removal")
        return
    
    removed_count = 0
    for genome in genomes:
        if "prompt_embedding" in genome:
            del genome["prompt_embedding"]
            removed_count += 1
    
    with open(temp_path_obj, 'w', encoding='utf-8') as f:
        json.dump(genomes, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Successfully removed embeddings from {removed_count} genomes")

