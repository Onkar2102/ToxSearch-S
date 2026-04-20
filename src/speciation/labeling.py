

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
    
    text = text.lower()
    text = re.sub(r'[^\w\s-]', ' ', text)
    text = re.sub(r'\b\d+\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_species_labels(
    species_dict: Dict[int, "Species"],
    n_words: int = 10,
    min_df: int = 1,
    max_df: float = 0.95,
    logger=None
) -> Dict[int, List[str]]:
    
    if logger is None:
        logger = get_logger("SpeciesLabeling")
    
    if not species_dict:
        logger.debug("No species to label")
        return {}
    
    species_ids = list(species_dict.keys())
    documents = []
    
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
        prompts = []
        if species.species_state == "frozen" or len(species.members) < 3:
            species_genomes = [g for g in elites_genomes if g.get("species_id") == sid]
            prompts = [g.get("prompt", "") for g in species_genomes if g.get("prompt")]
        else:
            prompts = [m.prompt for m in species.members if m.prompt]
        
        combined_text = ' '.join(prompts)
        preprocessed = preprocess_text(combined_text)
        documents.append(preprocessed)
    
    if not documents or all(not doc.strip() for doc in documents):
        logger.warning("No valid text content in species for labeling")
        return {sid: [] for sid in species_ids}
    
    try:
        n_docs = len(documents)
        
        if isinstance(max_df, float):
            max_df_abs = int(max_df * n_docs)
        else:
            max_df_abs = max_df
        
        if max_df_abs < min_df:
            max_df_abs = min(min_df + 1, n_docs)
            max_df = max_df_abs / n_docs if n_docs > 0 else 1.0
            logger.debug(f"Adjusted max_df to {max_df:.3f} (absolute: {max_df_abs}) for {n_docs} documents")
        
        if min_df >= n_docs:
            min_df = max(1, n_docs - 1) if n_docs > 1 else 1
            logger.debug(f"Adjusted min_df to {min_df} for {n_docs} documents")
        
        vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            stop_words=list(STOPWORDS),
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z-]{1,}\b',
            ngram_range=(1, 1)
        )
        
        count_matrix = vectorizer.fit_transform(documents)
        feature_names = vectorizer.get_feature_names_out()
        
        if len(feature_names) == 0:
            logger.warning("No features extracted from species prompts")
            return {sid: [] for sid in species_ids}
        
        tf_matrix = count_matrix.toarray().astype(float)
        row_sums = tf_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        tf_matrix = tf_matrix / row_sums
        
        n_classes = len(documents)
        doc_freq = (count_matrix.toarray() > 0).sum(axis=0)
        doc_freq[doc_freq == 0] = 1
        idf = np.log(n_classes / doc_freq) + 1
        
        ctfidf_matrix = tf_matrix * idf
        
        labels_dict = {}
        for i, sid in enumerate(species_ids):
            scores = ctfidf_matrix[i]
            if len(scores) <= n_words:
                top_indices = np.argsort(scores)[::-1]
            else:
                top_indices = np.argsort(scores)[-n_words:][::-1]
            
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
    
    if logger is None:
        logger = get_logger("SpeciesLabeling")
    
    if not species_dict:
        return
    
    labels_dict = extract_species_labels(species_dict, n_words=n_words, logger=logger)
    
    for sid, species in species_dict.items():
        labels = labels_dict.get(sid, [])
        
        species.labels = labels
        
        if not hasattr(species, 'label_history') or species.label_history is None:
            species.label_history = []
        
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
        
        if len(species.label_history) > 20:
            species.label_history = species.label_history[-20:]
    
    logger.info(f"Updated labels for {len(species_dict)} species at generation {current_generation}")
