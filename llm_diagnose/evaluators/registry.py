"""
Registry for evaluators to enable extensible registration.
"""

from typing import Type, Optional
import logging

from llm_diagnose.registry.base_registry import BaseRegistry
from llm_diagnose.evaluators.base import BaseEvaluator

logger = logging.getLogger(__name__)


class EvaluatorRegistry(BaseRegistry[Type[BaseEvaluator]]):
    """
    Registry for managing evaluator classes.
    
    Allows registration and retrieval of evaluator classes by name.
    """

    def __init__(self):
        """Initialize the evaluator registry."""
        super().__init__(name="EvaluatorRegistry")

    def register_evaluator(
        self,
        name: Optional[str] = None,
    ) -> callable:
        """
        Register an evaluator class.
        
        Can be used as a decorator.
        
        Args:
            name: Optional name for the evaluator (defaults to class name)
            
        Returns:
            Decorator function
            
        Examples:
            @registry.register_evaluator("my_evaluator")
            class MyEvaluator(BaseEvaluator):
                ...
        """
        def decorator(evaluator_class: Type[BaseEvaluator]) -> Type[BaseEvaluator]:
            evaluator_name = name or evaluator_class.__name__
            self.register(evaluator_name, factory=lambda *args, **kwargs: evaluator_class(*args, **kwargs))
            logger.info(f"Registered evaluator: {evaluator_name}")
            return evaluator_class
        
        return decorator

    def create_evaluator(
        self,
        name: str,
        *args,
        **kwargs,
    ) -> BaseEvaluator:
        """
        Create an evaluator instance by name.
        
        Args:
            name: Name of the evaluator to create
            *args: Arguments to pass to evaluator constructor
            **kwargs: Keyword arguments to pass to evaluator constructor
            
        Returns:
            Evaluator instance
        """
        evaluator_class = self.get(name)
        return evaluator_class(*args, **kwargs)


# Global evaluator registry instance
_evaluator_registry = EvaluatorRegistry()


def get_evaluator_registry() -> EvaluatorRegistry:
    """Get the global evaluator registry instance."""
    return _evaluator_registry

