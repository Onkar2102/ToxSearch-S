"""
embeddings.py

Prompt embedding computation using sentence-transformers for speciation.
Reads from temp.json, computes embeddings, and adds "prompt_embedding" field to genomes.

Model: all-MiniLM-L6-v2 (384-dimensional, L2-normalized embeddings)
"""

import json
import warnings
import numpy as np
from typing import List, Union, Optional
from pathlib import Path

from utils import get_custom_logging
from utils.device_utils import get_optimal_device
from utils import get_system_utils

# Suppress FutureWarning from Keras/TensorFlow about np.object deprecation
# This warning comes from third-party libraries (keras/tf2onnx) and cannot be fixed in our code
warnings.filterwarnings('ignore', category=FutureWarning, message='.*np.object.*')

get_logger, _, _, PerformanceLogger = get_custom_logging()
_, _, _, get_outputs_path, _, _, _ = get_system_utils()


class EmbeddingModel:
    """
    Embedding model wrapper using sentence-transformers library.
    
    This class manages the semantic embedding model used for converting text prompts
    into high-dimensional vectors. The default model is all-MiniLM-L6-v2, which:
    - Produces 384-dimensional embeddings
    - Is fast and efficient (good for large batches)
    - Provides high-quality semantic representations
    - Supports L2-normalization for cosine distance computation
    
    Embeddings are L2-normalized by default, which ensures:
    - Cosine similarity = dot product (for normalized vectors)
    - Semantic distance = 1 - cosine_similarity
    - All vectors lie on the unit hypersphere
    
    The model is loaded on the optimal available device (CUDA > MPS > CPU).
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize: bool = True
    ):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of sentence-transformer model (must be compatible)
            device: Target device ("cuda", "mps", "cpu", or None for auto-detect)
            normalize: If True, L2-normalize embeddings (required for cosine distance)
        """
        self.logger = get_logger("EmbeddingModel")
        self.model_name = model_name
        self.normalize = normalize
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
        # Auto-detect optimal device if not specified
        if device is None:
            device = get_optimal_device()
        self.device = device
        
        self._model = None  # Lazy-loaded SentenceTransformer instance
        self._load_model()
        
    def _load_model(self) -> None:
        """
        Load the sentence-transformer model (private method).
        
        Loads the model onto the specified device and verifies embedding dimension.
        Raises ImportError if sentence-transformers is not installed.
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            self.logger.info(f"Loading embedding model '{self.model_name}' on device '{self.device}'")
            # Load model (will download if not cached)
            self._model = SentenceTransformer(self.model_name, device=self.device)
            
            # Verify embedding dimension with test encoding
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
        """
        Encode text(s) into L2-normalized embedding vector(s).
        
        Args:
            text: Single string or list of strings to encode
            batch_size: Number of texts to process in parallel (larger = faster but more memory)
            show_progress: If True, show progress bar for large batches
        
        Returns:
            numpy array of embeddings:
            - Single text: shape (embedding_dim,)
            - Multiple texts: shape (num_texts, embedding_dim)
            - All embeddings are L2-normalized if normalize=True
        
        Raises:
            RuntimeError: If model not loaded
        """
        if self._model is None:
            raise RuntimeError("Embedding model not loaded")
        
        # Encode with normalization and numpy conversion
        return self._model.encode(
            text,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,  # L2-normalize for cosine distance
            convert_to_numpy=True  # Return numpy array (not torch tensor)
        )
    
    def __repr__(self) -> str:
        return f"EmbeddingModel(model_name='{self.model_name}', embedding_dim={self.embedding_dim})"


_embedding_model: Optional[EmbeddingModel] = None


def get_embedding_model(
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
    force_reload: bool = False
    ) -> EmbeddingModel:
    """
    Get or create singleton embedding model instance.
    
    Uses singleton pattern to ensure only one model is loaded in memory,
    which is important because:
    - Model loading is expensive (time and memory)
    - Multiple instances would waste GPU/CPU memory
    - Model can be reused across all embedding computations
    
    Args:
        model_name: Name of sentence-transformer model
        device: Target device (None for auto-detect)
        force_reload: If True, reload model even if already loaded (useful for testing)
    
    Returns:
        Singleton EmbeddingModel instance
    """
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
    """
    Read temp.json, compute embeddings for all prompts, and save back to temp.json.
    
    This function:
    1. Reads genomes from temp.json
    2. Extracts prompts from genomes
    3. Computes L2-normalized embeddings in batch
    4. Adds "prompt_embedding" field to each genome (as list for JSON serialization)
    5. Saves updated genomes back to temp.json
    
    The "prompt_embedding" field is stored as a list of floats (JSON-compatible format).
    When reading embeddings later, convert back to numpy array.
    
    Args:
        temp_path: Path to temp.json file. If None, uses default outputs_path / "temp.json"
        model_name: Name of sentence-transformer model
        batch_size: Batch size for embedding computation
        show_progress: If True, show progress bar
        logger: Optional logger instance
    
    Raises:
        FileNotFoundError: If temp.json doesn't exist
        ValueError: If temp.json is empty or invalid
    """
    if logger is None:
        logger = get_logger("Embeddings")
    
    # Determine temp path
    if temp_path is None:
        outputs_path = get_outputs_path()
        temp_path = str(outputs_path / "temp.json")
    
    temp_path_obj = Path(temp_path)
    if not temp_path_obj.exists():
        raise FileNotFoundError(f"Temp file not found: {temp_path}")
    
    logger.info(f"Computing embeddings for genomes in {temp_path}")
    
    # Load genomes from temp.json
    with open(temp_path_obj, 'r', encoding='utf-8') as f:
        genomes = json.load(f)
    
    if not genomes:
        logger.warning("No genomes found in temp.json")
        return
    
    # Check if embeddings already exist (skip if already computed)
    if all("prompt_embedding" in g for g in genomes):
        logger.info("Embeddings already exist in temp.json, skipping computation")
        return
    
    with PerformanceLogger(logger, "Embeddings: Compute and save", path=temp_path, num_genomes=len(genomes)):
        # Extract prompts
        prompts = [g.get("prompt", "") for g in genomes]
        
        # Get embedding model
        model = get_embedding_model(model_name=model_name)
        
        # Compute embeddings in batch
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
        
        # Add embeddings to genomes (convert numpy array to list for JSON)
        for i, genome in enumerate(genomes):
            try:
                if embeddings is not None and i < len(embeddings):
                    embedding = embeddings[i]
                    if embedding is not None and len(embedding) > 0:
                        # Convert numpy array to list (JSON-compatible)
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
        
        # Log summary
        logger.info(f"Embedding computation: {success_count} success, {failure_count} failures")
        if failure_count > 0:
            logger.warning(f"Population reduced by {failure_count} genomes due to embedding failures")
        
        # Save updated genomes back to temp.json
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
    """
    Add prompt_embedding to any genome that has 'prompt' but is missing (or has empty) prompt_embedding.
    Modifies genomes in place. Uses the same embedding model as speciation (L2-normalized).
    Returns the number of genomes that were backfilled.
    """
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
    """
    Remove "prompt_embedding" field from all genomes in temp.json.
    
    This function is called after speciation is complete to reduce storage size in temp.json.
    Embeddings are preserved in elites.json, reserves.json, and archive.json (required for
    speciation, leader-follower clustering, cluster quality metrics, and GDP historic visualization).
    
    This function directly reads and updates temp.json.
    
    Args:
        temp_path: Path to temp.json file. If None, uses default outputs_path / "temp.json"
        logger: Optional logger instance
    
    Raises:
        FileNotFoundError: If temp.json doesn't exist
    """
    if logger is None:
        logger = get_logger("Embeddings")
    
    # Determine temp path
    if temp_path is None:
        outputs_path = get_outputs_path()
        temp_path = str(outputs_path / "temp.json")
    
    temp_path_obj = Path(temp_path)
    if not temp_path_obj.exists():
        raise FileNotFoundError(f"Temp file not found: {temp_path}")
    
    logger.info(f"Removing embeddings from genomes in {temp_path}")
    
    # Load genomes from temp.json
    with open(temp_path_obj, 'r', encoding='utf-8') as f:
        genomes = json.load(f)
    
    if not genomes:
        # This is expected after distribute_genomes() clears temp.json
        # No need to warn - it's normal behavior
        logger.debug("temp.json is empty (already cleared by distribute_genomes) - skipping embedding removal")
        return
    
    # Remove prompt_embedding field from all genomes
    removed_count = 0
    for genome in genomes:
        if "prompt_embedding" in genome:
            del genome["prompt_embedding"]
            removed_count += 1
    
    # Save updated genomes back to temp.json
    with open(temp_path_obj, 'w', encoding='utf-8') as f:
        json.dump(genomes, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Successfully removed embeddings from {removed_count} genomes")

