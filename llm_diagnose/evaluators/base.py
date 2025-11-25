"""
Base evaluator class for all evaluation strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """
    Base class for all evaluators in the framework.
    
    This class defines the interface that all evaluators must implement.
    Subclasses should implement the evaluate method to perform specific
    evaluation tasks.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            name: Name of the evaluator
            config: Configuration dictionary for the evaluator
        """
        self.name = name or self.__class__.__name__
        self.config = config or {}
        logger.info(f"Initialized evaluator: {self.name}")

    @abstractmethod
    def evaluate(
        self,
        model: Any,
        dataset: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Perform evaluation on a model with a dataset.
        
        Args:
            model: The model to evaluate
            dataset: The dataset to use for evaluation
            **kwargs: Additional arguments specific to the evaluator
            
        Returns:
            Dictionary containing evaluation results
        """
        pass

    def prepare(self, model: Any, dataset: Any) -> None:
        """
        Prepare the evaluator for evaluation (optional override).
        
        This method can be overridden to perform any setup or preprocessing
        before evaluation begins.
        
        Args:
            model: The model to evaluate
            dataset: The dataset to use for evaluation
        """
        pass

    def cleanup(self) -> None:
        """
        Cleanup after evaluation (optional override).
        
        This method can be overridden to perform any cleanup after evaluation.
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Get the evaluator's configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def update_config(self, **kwargs: Any) -> None:
        """
        Update the evaluator's configuration.
        
        Args:
            **kwargs: Configuration key-value pairs to update
        """
        self.config.update(kwargs)
        logger.debug(f"Updated config for {self.name}: {kwargs}")

    def __repr__(self) -> str:
        """String representation of the evaluator."""
        return f"{self.__class__.__name__}(name={self.name})"

