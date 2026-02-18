"""
Generative Neural Engine (GNE) package for LLM integration and moderation.

This package provides:
- ResponseGenerator: Response generation using prompt_template from RGConfig.yaml
- PromptGenerator: Prompt generation using task templates from PGConfig.yaml
- evaluator: Content moderation using Google Perspective API
"""

def get_ResponseGenerator():
    """Lazy import of ResponseGenerator to avoid circular imports"""
    from gne.response_generator import ResponseGenerator
    return ResponseGenerator

def get_PromptGenerator():
    """Lazy import of PromptGenerator to avoid circular imports"""
    from gne.prompt_generator import PromptGenerator
    return PromptGenerator

def get_run_moderation_on_population():
    """Lazy import of run_moderation_on_population to avoid circular imports"""
    from gne.evaluator import run_moderation_on_population
    return run_moderation_on_population

__all__ = [
    "get_ResponseGenerator",
    "get_PromptGenerator",
    "get_run_moderation_on_population",
]
