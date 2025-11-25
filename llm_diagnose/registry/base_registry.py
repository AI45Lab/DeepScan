"""
Base registry class for extensible registration system.
"""

from abc import ABC, abstractmethod
from typing import Dict, TypeVar, Generic, Optional, Callable, Any
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseRegistry(ABC, Generic[T]):
    """
    Base registry class that provides registration and retrieval functionality.
    
    This class implements a registry pattern that allows dynamic registration
    and retrieval of components by name.
    """

    def __init__(self, name: str = "registry"):
        """
        Initialize the registry.
        
        Args:
            name: Name of the registry for logging purposes
        """
        self.name = name
        self._registry: Dict[str, T] = {}
        self._factories: Dict[str, Callable[..., T]] = {}

    def register(
        self,
        name: str,
        component: Optional[T] = None,
        factory: Optional[Callable[..., T]] = None,
    ) -> Callable:
        """
        Register a component or factory function.
        
        Can be used as a decorator or called directly.
        
        Args:
            name: Unique identifier for the component
            component: The component instance to register (optional)
            factory: Factory function that creates the component (optional)
            
        Returns:
            Decorator function if used as decorator, otherwise None
            
        Examples:
            # As decorator
            @registry.register("my_component")
            def create_component():
                return MyComponent()
            
            # Direct registration
            registry.register("my_component", MyComponent())
            registry.register("my_component", factory=lambda: MyComponent())
        """
        if component is not None and factory is not None:
            raise ValueError("Cannot specify both component and factory")

        def decorator(func_or_class: Any) -> Any:
            if callable(func_or_class):
                if isinstance(func_or_class, type):
                    # It's a class, register as factory
                    self._factories[name] = func_or_class
                else:
                    # It's a function, register as factory
                    self._factories[name] = func_or_class
                logger.info(f"Registered {name} in {self.name}")
                return func_or_class
            else:
                # It's an instance
                self._registry[name] = func_or_class
                logger.info(f"Registered {name} in {self.name}")
                return func_or_class

        if component is not None:
            self._registry[name] = component
            logger.info(f"Registered {name} in {self.name}")
            return None
        elif factory is not None:
            self._factories[name] = factory
            logger.info(f"Registered {name} in {self.name}")
            return None
        else:
            # Used as decorator
            return decorator

    def get(self, name: str, *args, **kwargs) -> T:
        """
        Retrieve a component by name.
        
        Args:
            name: Name of the component to retrieve
            *args: Arguments to pass to factory function if needed
            **kwargs: Keyword arguments to pass to factory function if needed
            
        Returns:
            The registered component
            
        Raises:
            KeyError: If the component is not registered
        """
        if name in self._registry:
            return self._registry[name]
        elif name in self._factories:
            factory = self._factories[name]
            return factory(*args, **kwargs)
        else:
            raise KeyError(
                f"'{name}' not found in {self.name}. "
                f"Available: {list(self._registry.keys()) + list(self._factories.keys())}"
            )

    def list(self) -> list[str]:
        """
        List all registered component names.
        
        Returns:
            List of registered component names
        """
        return list(set(list(self._registry.keys()) + list(self._factories.keys())))

    def unregister(self, name: str) -> None:
        """
        Unregister a component.
        
        Args:
            name: Name of the component to unregister
        """
        if name in self._registry:
            del self._registry[name]
            logger.info(f"Unregistered {name} from {self.name}")
        elif name in self._factories:
            del self._factories[name]
            logger.info(f"Unregistered {name} from {self.name}")
        else:
            logger.warning(f"'{name}' not found in {self.name}, cannot unregister")

    def is_registered(self, name: str) -> bool:
        """
        Check if a component is registered.
        
        Args:
            name: Name of the component to check
            
        Returns:
            True if registered, False otherwise
        """
        return name in self._registry or name in self._factories

    def clear(self) -> None:
        """Clear all registered components."""
        self._registry.clear()
        self._factories.clear()
        logger.info(f"Cleared all entries from {self.name}")

