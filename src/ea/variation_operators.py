"""
Abstract base class defining the interface for all variation operators in the evolutionary pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List
import logging
logger = logging.getLogger(__name__)

class VariationOperator(ABC):
    """
    Abstract base class for variation operators (e.g., mutation, crossover) used in prompt evolution.

    Args:
        name: Name of the operator (defaults to class name).
        operator_type: Operator category ('mutation', 'crossover', or 'hybrid').
        description: Short description of the operator's functionality.
    """
    def __init__(self, name=None, operator_type="mutation", description=""):
        self.name = name or self.__class__.__name__
        self.operator_type = operator_type
        self.description = description
        logger.debug(f"Initialized operator: {self.name} (type={self.operator_type})")

    @abstractmethod
    def apply(self, operator_input: Dict[str, Any]) -> List[str]:
        """
        Apply the operator to produce one or more text variants.

        Args:
            operator_input: Dict containing at least 'prompt' (parent text) and operator-specific fields
                (e.g. species_id, toxicity). May include full genome fields when provided by the engine.

        Returns:
            List of variant strings (prompts). May be empty if the operator produces no valid variant.
        """
        pass

    def __str__(self):
        logger.debug(f"__str__ called on operator: {self.name}")
        return f"{self.name} ({self.operator_type})"

    def get_metadata(self) -> dict:
        logger.debug(f"get_metadata called for operator: {self.name}")
        return {
            "name": self.name,
            "type": self.operator_type,
            "description": self.description
        }
