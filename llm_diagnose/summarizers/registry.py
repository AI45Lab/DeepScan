"""
Registry for summarizers to enable extensible registration.
"""

from typing import Type, Optional
import logging

from llm_diagnose.registry.base_registry import BaseRegistry
from llm_diagnose.summarizers.base import BaseSummarizer

logger = logging.getLogger(__name__)


class SummarizerRegistry(BaseRegistry[Type[BaseSummarizer]]):
    """
    Registry for managing summarizer classes.
    
    Allows registration and retrieval of summarizer classes by name.
    """

    def __init__(self):
        """Initialize the summarizer registry."""
        super().__init__(name="SummarizerRegistry")

    def register_summarizer(
        self,
        name: Optional[str] = None,
    ) -> callable:
        """
        Register a summarizer class.
        
        Can be used as a decorator.
        
        Args:
            name: Optional name for the summarizer (defaults to class name)
            
        Returns:
            Decorator function
            
        Examples:
            @registry.register_summarizer("my_summarizer")
            class MySummarizer(BaseSummarizer):
                ...
        """
        def decorator(summarizer_class: Type[BaseSummarizer]) -> Type[BaseSummarizer]:
            summarizer_name = name or summarizer_class.__name__
            self.register(
                summarizer_name,
                factory=lambda *args, **kwargs: summarizer_class(*args, **kwargs),
            )
            logger.info(f"Registered summarizer: {summarizer_name}")
            return summarizer_class
        
        return decorator

    def create_summarizer(
        self,
        name: str,
        *args,
        **kwargs,
    ) -> BaseSummarizer:
        """
        Create a summarizer instance by name.
        
        Args:
            name: Name of the summarizer to create
            *args: Arguments to pass to summarizer constructor
            **kwargs: Keyword arguments to pass to summarizer constructor
            
        Returns:
            Summarizer instance
        """
        summarizer_class = self.get(name)
        return summarizer_class(*args, **kwargs)


# Global summarizer registry instance
_summarizer_registry = SummarizerRegistry()


def get_summarizer_registry() -> SummarizerRegistry:
    """Get the global summarizer registry instance."""
    return _summarizer_registry

