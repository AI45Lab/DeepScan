"""
Example usage of the LLM-Diagnose Framework.

This example demonstrates how to:
1. Register models and datasets
2. Load configurations
3. Create and use evaluators
4. Summarize results
"""

from llm_diagnose import (
    ModelRegistry,
    DatasetRegistry,
    ConfigLoader,
    NeuronAttributionEvaluator,
    RepresentationEngineeringEvaluator,
    BaseSummarizer,
)
from llm_diagnose.evaluators.registry import get_evaluator_registry
from llm_diagnose.summarizers.registry import get_summarizer_registry


# Example 1: Register a model
def example_register_model():
    """Example of registering a model."""
    registry = ModelRegistry()
    
    # Register a model factory
    @registry.register_model("gpt2")
    def create_gpt2():
        # In practice, this would load the actual model
        return {"model_type": "gpt2", "model_name": "gpt2"}
    
    # Get the model
    model = registry.get_model("gpt2")
    print(f"Retrieved model: {model}")


# Example 2: Register a dataset
def example_register_dataset():
    """Example of registering a dataset."""
    registry = DatasetRegistry()
    
    # Register a dataset factory
    @registry.register_dataset("glue_sst2")
    def create_sst2():
        # In practice, this would load the actual dataset
        return {"dataset_name": "glue", "subset": "sst2", "split": "test"}
    
    # Get the dataset
    dataset = registry.get_dataset("glue_sst2")
    print(f"Retrieved dataset: {dataset}")


# Example 3: Load configuration
def example_load_config():
    """Example of loading configuration."""
    # Create a config from dictionary
    config = ConfigLoader.from_dict({
        "model": {
            "name": "gpt2",
            "device": "cuda",
        },
        "evaluator": {
            "attribution_method": "gradient",
            "target_layers": ["layer_0", "layer_1"],
        },
    })
    
    # Access config values
    model_name = config.get("model.name")
    print(f"Model name: {model_name}")
    
    # Update config
    config.set("evaluator.top_k", 20)
    print(f"Updated config: {config.to_dict()}")


# Example 4: Create a custom evaluator
def example_custom_evaluator():
    """Example of creating a custom evaluator."""
    from llm_diagnose.evaluators.base import BaseEvaluator
    
    class CustomNeuronAttributionEvaluator(NeuronAttributionEvaluator):
        """Custom neuron attribution evaluator."""
        
        def _compute_attributions(self, model, dataset, target_layers, target_neurons=None):
            """Implement custom attribution computation."""
            # Your custom implementation here
            return {
                layer: {"neuron_0": 0.5, "neuron_1": 0.3}
                for layer in target_layers
            }
    
    # Register the evaluator
    evaluator_registry = get_evaluator_registry()
    evaluator_registry.register_evaluator("custom_attribution")(CustomNeuronAttributionEvaluator)
    
    # Create and use the evaluator
    evaluator = evaluator_registry.create_evaluator(
        "custom_attribution",
        name="my_custom_evaluator",
        attribution_method="gradient",
    )
    
    # Mock model and dataset
    model = {"type": "gpt2"}
    dataset = {"name": "sst2"}
    
    results = evaluator.evaluate(
        model,
        dataset,
        target_layers=["layer_0"],
        top_k=5,
    )
    print(f"Evaluation results: {results}")


# Example 5: Create a custom summarizer
def example_custom_summarizer():
    """Example of creating a custom summarizer."""
    class BenchmarkSummarizer(BaseSummarizer):
        """Custom summarizer for benchmark results."""
        
        def summarize(self, results, benchmark=None, **kwargs):
            """Summarize results for a specific benchmark."""
            summary = {
                "benchmark": benchmark or "unknown",
                "evaluator": results.get("method", "unknown"),
                "num_layers": len(results.get("attributions", {})),
            }
            
            if "top_neurons" in results:
                summary["top_neurons_found"] = sum(
                    len(neurons) for neurons in results["top_neurons"].values()
                )
            
            return summary
    
    # Register the summarizer
    summarizer_registry = get_summarizer_registry()
    summarizer_registry.register_summarizer("benchmark")(BenchmarkSummarizer)
    
    # Create and use the summarizer
    summarizer = summarizer_registry.create_summarizer("benchmark")
    
    # Mock results
    results = {
        "method": "gradient",
        "attributions": {"layer_0": {}, "layer_1": {}},
        "top_neurons": {"layer_0": [("neuron_0", 0.5)], "layer_1": [("neuron_1", 0.3)]},
    }
    
    summary = summarizer.summarize(results, benchmark="glue_sst2")
    print(f"Summary: {summary}")
    
    # Format as markdown
    report = summarizer.format_report(summary, format="markdown")
    print(f"\nMarkdown report:\n{report}")


# Example 6: Complete workflow
def example_complete_workflow():
    """Example of a complete evaluation workflow."""
    # 1. Setup registries
    model_registry = ModelRegistry()
    dataset_registry = DatasetRegistry()
    evaluator_registry = get_evaluator_registry()
    summarizer_registry = get_summarizer_registry()
    
    # 2. Register components
    @model_registry.register_model("gpt2")
    def create_model():
        return {"type": "gpt2"}
    
    @dataset_registry.register_dataset("sst2")
    def create_dataset():
        return {"name": "sst2", "split": "test"}
    
    # 3. Load configuration
    config = ConfigLoader.from_dict({
        "model": "gpt2",
        "dataset": "sst2",
        "evaluator": {
            "type": "neuron_attribution",
            "attribution_method": "gradient",
        },
    })
    
    # 4. Get model and dataset
    model = model_registry.get_model(config.get("model"))
    dataset = dataset_registry.get_dataset(config.get("dataset"))
    
    # 5. Create evaluator (using base class for demo)
    evaluator = NeuronAttributionEvaluator(
        name="my_evaluator",
        config=config.get("evaluator", {}),
    )
    
    # 6. Run evaluation
    results = evaluator.evaluate(
        model,
        dataset,
        target_layers=["layer_0", "layer_1"],
        top_k=10,
    )
    
    # 7. Summarize results
    summarizer = BaseSummarizer(name="default_summarizer")
    summary = summarizer.summarize(results, benchmark="sst2")
    
    print("Complete workflow results:")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    print("=" * 60)
    print("Example 1: Register Model")
    print("=" * 60)
    example_register_model()
    
    print("\n" + "=" * 60)
    print("Example 2: Register Dataset")
    print("=" * 60)
    example_register_dataset()
    
    print("\n" + "=" * 60)
    print("Example 3: Load Configuration")
    print("=" * 60)
    example_load_config()
    
    print("\n" + "=" * 60)
    print("Example 4: Custom Evaluator")
    print("=" * 60)
    example_custom_evaluator()
    
    print("\n" + "=" * 60)
    print("Example 5: Custom Summarizer")
    print("=" * 60)
    example_custom_summarizer()
    
    print("\n" + "=" * 60)
    print("Example 6: Complete Workflow")
    print("=" * 60)
    example_complete_workflow()

