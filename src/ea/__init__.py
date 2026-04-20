

def get_EvolutionEngine():
    
    from ea.evolution_engine import EvolutionEngine
    return EvolutionEngine

def get_run_evolution():
    
    from ea.run_evolution import run_evolution
    return run_evolution

def get_create_final_statistics_with_tracker():
    
    from ea.run_evolution import create_final_statistics_with_tracker
    return create_final_statistics_with_tracker

def get_update_evolution_tracker_with_generation_global():
    
    from ea.run_evolution import update_evolution_tracker_with_generation_global
    return update_evolution_tracker_with_generation_global

def get_LLM_POSAwareSynonymReplacement():
    
    from ea.synonym_replacement import LLM_POSAwareSynonymReplacement
    return LLM_POSAwareSynonymReplacement

def get_POSAwareAntonymReplacement():
    
    from ea.antonym_replacement import POSAwareAntonymReplacement
    return POSAwareAntonymReplacement

def get_MLMOperator():
    
    from ea.mlm_operator import MLMOperator
    return MLMOperator

def get_LLMBasedParaphrasingOperator():
    
    from ea.paraphrasing import LLMBasedParaphrasingOperator
    return LLMBasedParaphrasingOperator

def get_StylisticMutator():
    
    from ea.stylistic_mutator import StylisticMutator
    return StylisticMutator

def get_LLMBackTranslationHIOperator():
    
    from ea.back_translation import LLMBackTranslationHIOperator
    return LLMBackTranslationHIOperator

def get_LLMBackTranslationFROperator():
    
    from ea.back_translation import LLMBackTranslationFROperator
    return LLMBackTranslationFROperator

def get_LLMBackTranslationDEOperator():
    
    from ea.back_translation import LLMBackTranslationDEOperator
    return LLMBackTranslationDEOperator

def get_LLMBackTranslationJAOperator():
    
    from ea.back_translation import LLMBackTranslationJAOperator
    return LLMBackTranslationJAOperator

def get_LLMBackTranslationZHOperator():
    
    from ea.back_translation import LLMBackTranslationZHOperator
    return LLMBackTranslationZHOperator

def get_NegationOperator():
    
    from ea.negation_operator import NegationOperator
    return NegationOperator

def get_TypographicalErrorsOperator():
    
    from ea.typographical_errors import TypographicalErrorsOperator
    return TypographicalErrorsOperator

def get_ConceptAdditionOperator():
    
    from ea.concept_addition import ConceptAdditionOperator
    return ConceptAdditionOperator

def get_InformedEvolutionOperator():
    
    from ea.informed_evolution import InformedEvolutionOperator
    return InformedEvolutionOperator

def get_SemanticSimilarityCrossover():
    
    from ea.semantic_similarity_crossover import SemanticSimilarityCrossover
    return SemanticSimilarityCrossover

def get_SemanticFusionCrossover():
    
    from ea.fusion_crossover import SemanticFusionCrossover
    return SemanticFusionCrossover




import logging
logger = logging.getLogger(__name__)

__all__ = [
    "get_EvolutionEngine",
    "get_run_evolution",
    "get_create_final_statistics_with_tracker",
    "get_update_evolution_tracker_with_generation_global",
    
    "get_LLM_POSAwareSynonymReplacement",
    "get_LLM_POSAwareAntonymReplacement",
    "get_MLMOperator",
    "get_LLMBasedParaphrasingOperator",
    "get_StylisticMutator",
    
    "get_LLMBackTranslationHIOperator",
    "get_LLMBackTranslationFROperator",
    "get_LLMBackTranslationDEOperator",
    "get_LLMBackTranslationJAOperator",
    "get_LLMBackTranslationZHOperator",
    
    "get_NegationOperator",
    "get_TypographicalErrorsOperator",
    "get_ConceptAdditionOperator",
    "get_InformedEvolutionOperator",
    
    "get_SemanticSimilarityCrossover",
    "get_SemanticFusionCrossover",
]