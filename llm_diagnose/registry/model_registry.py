"""
Model registry for managing LLM and MLLM models.
"""

from typing import Optional, Dict, Any, Callable
import logging

from llm_diagnose.registry.base_registry import BaseRegistry

logger = logging.getLogger(__name__)


class ModelRegistry(BaseRegistry):
    """
    Registry for managing model instances and factories.
    
    Supports registration of both model instances and factory functions
    that create models on demand.
    """

    def __init__(self):
        """Initialize the model registry."""
        super().__init__(name="ModelRegistry")

    def register_model(
        self,
        name: str,
        model: Optional[Any] = None,
        factory: Optional[Callable[..., Any]] = None,
        **metadata: Any,
    ) -> Callable:
        """
        Register a model or model factory.
        
        Args:
            name: Unique identifier for the model
            model: Model instance to register (optional)
            factory: Factory function that creates the model (optional)
            **metadata: Additional metadata about the model
            
        Returns:
            Decorator function if used as decorator
            
        Examples:
            # Register a model instance
            registry.register_model("gpt2", model=GPT2Model())
            
            # Register a factory function
            @registry.register_model("gpt2")
            def create_gpt2():
                return GPT2Model.from_pretrained("gpt2")
        """
        # Store metadata if provided
        if metadata:
            if not hasattr(self, "_metadata"):
                self._metadata: Dict[str, Dict[str, Any]] = {}
            self._metadata[name] = metadata

        return self.register(name, model, factory)

    def get_model(self, name: str, *args, **kwargs) -> Any:
        """
        Get a model by name.
        
        Args:
            name: Name of the model to retrieve
            *args: Arguments to pass to factory function
            **kwargs: Keyword arguments to pass to factory function
            
        Returns:
            The model instance
        """
        return self.get(name, *args, **kwargs)

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get metadata for a registered model.
        
        Args:
            name: Name of the model
            
        Returns:
            Dictionary of metadata, empty if not found
        """
        if hasattr(self, "_metadata") and name in self._metadata:
            return self._metadata[name]
        return {}


# Global model registry instance
_model_registry = ModelRegistry()


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    return _model_registry

