"""
Dataset registry for managing datasets.
"""

from typing import Optional, Dict, Any, Callable
import logging

from llm_diagnose.registry.base_registry import BaseRegistry

logger = logging.getLogger(__name__)


class DatasetRegistry(BaseRegistry):
    """
    Registry for managing dataset instances and factories.
    
    Supports registration of both dataset instances and factory functions
    that create datasets on demand.
    """

    def __init__(self):
        """Initialize the dataset registry."""
        super().__init__(name="DatasetRegistry")

    def register_dataset(
        self,
        name: str,
        dataset: Optional[Any] = None,
        factory: Optional[Callable[..., Any]] = None,
        **metadata: Any,
    ) -> Callable:
        """
        Register a dataset or dataset factory.
        
        Args:
            name: Unique identifier for the dataset
            dataset: Dataset instance to register (optional)
            factory: Factory function that creates the dataset (optional)
            **metadata: Additional metadata about the dataset
            
        Returns:
            Decorator function if used as decorator
            
        Examples:
            # Register a dataset instance
            registry.register_dataset("glue_sst2", dataset=GLUEDataset())
            
            # Register a factory function
            @registry.register_dataset("glue_sst2")
            def create_glue_sst2():
                return load_dataset("glue", "sst2")
        """
        # Store metadata if provided
        if metadata:
            if not hasattr(self, "_metadata"):
                self._metadata: Dict[str, Dict[str, Any]] = {}
            self._metadata[name] = metadata

        return self.register(name, dataset, factory)

    def get_dataset(self, name: str, *args, **kwargs) -> Any:
        """
        Get a dataset by name.
        
        Args:
            name: Name of the dataset to retrieve
            *args: Arguments to pass to factory function
            **kwargs: Keyword arguments to pass to factory function
            
        Returns:
            The dataset instance
        """
        return self.get(name, *args, **kwargs)

    def get_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get metadata for a registered dataset.
        
        Args:
            name: Name of the dataset
            
        Returns:
            Dictionary of metadata, empty if not found
        """
        if hasattr(self, "_metadata") and name in self._metadata:
            return self._metadata[name]
        return {}


# Global dataset registry instance
_dataset_registry = DatasetRegistry()


def get_dataset_registry() -> DatasetRegistry:
    """Get the global dataset registry instance."""
    return _dataset_registry

