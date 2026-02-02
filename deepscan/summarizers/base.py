"""
Base summarizer class for all summarization strategies.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class BaseSummarizer(ABC):
    """
    Base class for all summarizers in the framework.
    
    Summarizers aggregate and present evaluation results in a format
    suitable for different benchmarks or reporting needs.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the summarizer.
        
        Args:
            name: Name of the summarizer
            config: Configuration dictionary for the summarizer
        """
        self.name = name or self.__class__.__name__
        self.config = config or {}
        logger.info(f"Initialized summarizer: {self.name}")

    @abstractmethod
    def summarize(
        self,
        results: Dict[str, Any],
        benchmark: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Summarize evaluation results.
        
        Args:
            results: Dictionary containing evaluation results
            benchmark: Name of the benchmark (optional)
            **kwargs: Additional arguments specific to the summarizer
            
        Returns:
            Dictionary containing summarized results
        """
        pass

    def format_report(
        self,
        summary: Dict[str, Any],
        format: str = "dict",
    ) -> Any:
        """
        Format the summary as a report.
        
        Args:
            summary: Summary dictionary from summarize()
            format: Output format ("dict", "json", "markdown", "text")
            
        Returns:
            Formatted report in the requested format
        """
        if format == "dict":
            return summary
        elif format == "json":
            import json
            return json.dumps(summary, indent=2)
        elif format == "markdown":
            return self._format_markdown(summary)
        elif format == "text":
            return self._format_text(summary)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _format_markdown(self, summary: Dict[str, Any]) -> str:
        """
        Format summary as Markdown.
        
        Args:
            summary: Summary dictionary
            
        Returns:
            Markdown formatted string
        """
        lines = [f"# {self.name} Summary\n"]
        
        for key, value in summary.items():
            if isinstance(value, dict):
                lines.append(f"## {key}\n")
                for sub_key, sub_value in value.items():
                    lines.append(f"- **{sub_key}**: {sub_value}\n")
            else:
                lines.append(f"- **{key}**: {value}\n")
        
        return "\n".join(lines)

    def _format_text(self, summary: Dict[str, Any]) -> str:
        """
        Format summary as plain text.
        
        Args:
            summary: Summary dictionary
            
        Returns:
            Plain text formatted string
        """
        lines = [f"{self.name} Summary", "=" * 50]
        
        for key, value in summary.items():
            if isinstance(value, dict):
                lines.append(f"\n{key}:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  {sub_key}: {sub_value}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)

    def get_config(self) -> Dict[str, Any]:
        """
        Get the summarizer's configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config.copy()

    def update_config(self, **kwargs: Any) -> None:
        """
        Update the summarizer's configuration.
        
        Args:
            **kwargs: Configuration key-value pairs to update
        """
        self.config.update(kwargs)
        logger.debug(f"Updated config for {self.name}: {kwargs}")

    def __repr__(self) -> str:
        """String representation of the summarizer."""
        return f"{self.__class__.__name__}(name={self.name})"

