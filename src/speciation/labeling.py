"""
labeling.py

c-TF-IDF based species labeling for semantic characterization.

c-TF-IDF (class-based Term Frequency-Inverse Document Frequency) extracts
representative words for each species based on the prompts of its members.
Unlike regular TF-IDF which operates on individual documents, c-TF-IDF
treats each species as a single "document" by concatenating all member prompts,
then finds words that are distinctive to each species compared to others.

The labels help understand what topics/themes each species represents and
how they differ from other species.
"""

import re
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

if TYPE_CHECKING:
    from .species import Species, Individual

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


# Common English stopwords to exclude from labels
STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been', 'be', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may',
    'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'it', 'its',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we', 'they', 'me',
    'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their', 'mine',
    'yours', 'hers', 'ours', 'theirs', 'what', 'which', 'who', 'whom', 'whose',
    'when', 'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then',
    'once', 'if', 'because', 'until', 'while', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down',
    'out', 'off', 'over', 'under', 'again', 'further', 'any', 'being', 'having',
    'doing', 'am', 'get', 'gets', 'got', 'getting', 'make', 'makes', 'made',
    'making', 'way', 'ways', 'one', 'ones', 'someone', 'something', 'anyone',
    'anything', 'everyone', 'everything', 'nothing', 'things', 'thing', 'people',
    'person', 'using', 'use', 'used'
}


def preprocess_text(text: str) -> str:
    """
    Preprocess text for c-TF-IDF analysis.
    
    - Converts to lowercase
    - Removes punctuation except hyphens
    - Removes numbers
    - Collapses whitespace
    
    Args:
        text: Input text
        
    Returns:
        Preprocessed text
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation except hyphens (to keep compound words)
    text = re.sub(r'[^\w\s-]', ' ', text)
    # Remove standalone numbers
    text = re.sub(r'\b\d+\b', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_species_labels(
    species_dict: Dict[int, "Species"],
    n_words: int = 10,
    min_df: int = 1,
    max_df: float = 0.95,
    logger=None
) -> Dict[int, List[str]]:
    """
    Extract c-TF-IDF based labels for all species.
    
    c-TF-IDF treats each species as a class and finds words that are
    distinctive to each species compared to others. This helps identify
    the semantic themes that characterize each species.
    
    Algorithm:
    1. For each species, concatenate all member prompts into one "document"
    2. Build vocabulary using CountVectorizer (exclude stopwords)
    3. Compute TF for each class (word frequency in class / total words in class)
    4. Compute IDF across classes (log(n_classes / n_classes_with_word))
    5. Multiply TF * IDF to get c-TF-IDF scores
    6. Extract top n_words for each species
    
    Args:
        species_dict: Dictionary mapping species_id -> Species
        n_words: Number of label words to extract per species (default: 10)
        min_df: Minimum document frequency for vocabulary (default: 1)
        max_df: Maximum document frequency ratio for vocabulary (default: 0.95)
        logger: Optional logger instance
        
    Returns:
        Dictionary mapping species_id -> list of top n_words labels
    """
    if logger is None:
        logger = get_logger("SpeciesLabeling")
    
    if not species_dict:
        logger.debug("No species to label")
        return {}
    
    # Prepare documents: one concatenated document per species
    species_ids = list(species_dict.keys())
    documents = []
    
    # Load prompts from elites.json for frozen species (they may not have all members in memory)
    from utils import get_system_utils
    _, _, _, get_outputs_path, _, _, _ = get_system_utils()
    outputs_path = get_outputs_path()
    elites_path = outputs_path / "elites.json"
    elites_genomes = []
    if elites_path.exists():
        try:
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_genomes = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load elites.json for label extraction: {e}")
    
    for sid in species_ids:
        species = species_dict[sid]
        # For frozen species or species with few members, load prompts from elites.json
        # So we have all member prompts for accurate labeling
        prompts = []
        if species.species_state == "frozen" or len(species.members) < 3:
            # Load from elites.json for frozen species or small species
            species_genomes = [g for g in elites_genomes if g.get("species_id") == sid]
            prompts = [g.get("prompt", "") for g in species_genomes if g.get("prompt")]
        else:
            # Use in-memory members for active species
            prompts = [m.prompt for m in species.members if m.prompt]
        
        combined_text = ' '.join(prompts)
        preprocessed = preprocess_text(combined_text)
        documents.append(preprocessed)
    
    if not documents or all(not doc.strip() for doc in documents):
        logger.warning("No valid text content in species for labeling")
        return {sid: [] for sid in species_ids}
    
    try:
        # Adjust max_df and min_df based on number of documents to avoid ValueError
        n_docs = len(documents)
        
        # If max_df as ratio results in fewer documents than min_df, adjust
        if isinstance(max_df, float):
            max_df_abs = int(max_df * n_docs)
        else:
            max_df_abs = max_df
        
        # Ensure max_df_abs >= min_df and is at least 1
        if max_df_abs < min_df:
            # Adjust max_df to be at least min_df + 1, or n_docs if that's smaller
            max_df_abs = min(min_df + 1, n_docs)
            max_df = max_df_abs / n_docs if n_docs > 0 else 1.0
            logger.debug(f"Adjusted max_df to {max_df:.3f} (absolute: {max_df_abs}) for {n_docs} documents")
        
        # If we have very few documents, adjust min_df to be at most n_docs - 1
        if min_df >= n_docs:
            min_df = max(1, n_docs - 1) if n_docs > 1 else 1
            logger.debug(f"Adjusted min_df to {min_df} for {n_docs} documents")
        
        # Build vocabulary with CountVectorizer
        vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            stop_words=list(STOPWORDS),
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z-]{1,}\b',  # Words with 2+ chars
            ngram_range=(1, 1)  # Unigrams only for simplicity
        )
        
        # Get word counts per species
        count_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        if len(feature_names) == 0:
            logger.warning("No features extracted from species prompts")
            return {sid: [] for sid in species_ids}
        
        # Compute c-TF-IDF
        # TF: normalize by total words in each class
        tf_matrix = count_matrix.toarray().astype(float)
        row_sums = tf_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        tf_matrix = tf_matrix / row_sums
        
        # IDF: log(n_classes / n_classes_with_word)
        n_classes = len(documents)
        doc_freq = (count_matrix.toarray() > 0).sum(axis=0)
        doc_freq[doc_freq == 0] = 1  # Avoid division by zero
        idf = np.log(n_classes / doc_freq) + 1  # Add 1 to avoid log(1) = 0
        
        # c-TF-IDF = TF * IDF
        ctfidf_matrix = tf_matrix * idf
        
        # Extract top n_words for each species
        labels_dict = {}
        for i, sid in enumerate(species_ids):
            scores = ctfidf_matrix[i]
            # Get indices of top n_words scores
            if len(scores) <= n_words:
                top_indices = np.argsort(scores)[::-1]
            else:
                top_indices = np.argsort(scores)[-n_words:][::-1]
            
            # Extract words (filter out zero-score words)
            labels = [
                feature_names[idx] for idx in top_indices
                if scores[idx] > 0
            ][:n_words]
            
            labels_dict[sid] = labels
            logger.debug(f"Species {sid} labels: {labels}")
        
        logger.info(f"Generated c-TF-IDF labels for {len(labels_dict)} species")
        return labels_dict
        
    except Exception as e:
        logger.error(f"Failed to extract species labels: {e}", exc_info=True)
        return {sid: [] for sid in species_ids}


def update_species_labels(
    species_dict: Dict[int, "Species"],
    current_generation: int,
    n_words: int = 10,
    logger=None
) -> None:
    """
    Update labels for all species and record in label_history.
    
    This function:
    1. Extracts c-TF-IDF labels for all species
    2. Updates each species' `labels` attribute with current labels
    3. Appends current labels and fitness metrics to each species' `label_history`
    
    Each label_history entry contains:
    - generation: The generation number
    - labels: List of top n_words representative words
    - best_fitness: Best fitness in the species at this generation
    - avg_fitness: Average fitness across all members
    - size: Number of members in the species
    
    Should be called after each generation's speciation processing.
    
    Args:
        species_dict: Dictionary mapping species_id -> Species
        current_generation: Current generation number
        n_words: Number of label words to extract (default: 10)
        logger: Optional logger instance
    """
    if logger is None:
        logger = get_logger("SpeciesLabeling")
    
    if not species_dict:
        return
    
    # Extract labels for all species
    labels_dict = extract_species_labels(species_dict, n_words=n_words, logger=logger)
    
    # Update each species
    for sid, species in species_dict.items():
        labels = labels_dict.get(sid, [])
        
        # Update current labels
        species.labels = labels
        
        # Append to label history (keep last 20 generations like fitness_history)
        if not hasattr(species, 'label_history') or species.label_history is None:
            species.label_history = []
        
        # Get fitness metrics for this generation
        best_fitness = species.best_fitness if hasattr(species, 'best_fitness') else 0.0
        avg_fitness = species.avg_fitness if hasattr(species, 'avg_fitness') else 0.0
        size = species.size if hasattr(species, 'size') else len(species.members)
        
        species.label_history.append({
            "generation": current_generation,
            "labels": labels,
            "best_fitness": round(best_fitness, 4),
            "avg_fitness": round(avg_fitness, 4),
            "size": size
        })
        
        # Keep only last 20 entries
        if len(species.label_history) > 20:
            species.label_history = species.label_history[-20:]
    
    logger.info(f"Updated labels for {len(species_dict)} species at generation {current_generation}")
