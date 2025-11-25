# LLM-Diagnose Framework

A flexible and extensible framework for diagnosing Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). This framework provides a modular architecture for evaluating models through neuron attribution and representation engineering techniques.

## Features

- **Model Registry**: Register and manage model instances and factories
- **Dataset Registry**: Register and manage dataset instances and factories
- **Configuration Management**: Load and manage configurations from YAML/JSON files
- **Extensible Evaluators**: Two main types of evaluators:
  - **Neuron Attribution**: Analyze which neurons contribute to model behavior
  - **Representation Engineering**: Analyze and manipulate model representations
- **Customizable Summarizers**: Aggregate and format evaluation results for different benchmarks
- **Plugin Architecture**: Easy to extend with custom evaluators and summarizers

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

Optional dependencies:

```bash
pip install datasets  # required for the built-in MMLU dataset loader
```

## Quick Start

### 1. Register Models and Datasets

#### Registering Individual Models

```python
from llm_diagnose import ModelRegistry, DatasetRegistry

# Register a model
model_registry = ModelRegistry()

@model_registry.register_model("gpt2")
def create_gpt2():
    from transformers import GPT2LMHeadModel
    return GPT2LMHeadModel.from_pretrained("gpt2")

# Register a dataset
dataset_registry = DatasetRegistry()

@dataset_registry.register_dataset("glue_sst2")
def create_sst2():
    from datasets import load_dataset
    return load_dataset("glue", "sst2", split="test")
```

#### Registering Model Families (e.g., Qwen)

For model families with multiple generations, organize by **generation** rather than size, since different generations may have different configurations even at the same size.

**Option 1: Register by generation (Recommended)**

```python
from llm_diagnose.registry.model_registry import get_model_registry

registry = get_model_registry()

@registry.register_model(
    "qwen3",
    model_family="qwen",
    model_generation="qwen3",
)
def create_qwen3(model_name: str = "Qwen3-7B", device: str = "cuda", **kwargs):
    """Create Qwen3 model of specified name."""
    from transformers import AutoModelForCausalLM
    
    model_paths = {
        "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
        "Qwen3-1.5B": "Qwen/Qwen3-1.5B",
        "Qwen3-2B": "Qwen/Qwen3-2B",
        "Qwen3-7B": "Qwen/Qwen3-7B",
        "Qwen3-14B": "Qwen/Qwen3-14B",
        "Qwen3-32B": "Qwen/Qwen3-32B",
    }
    
    if model_name not in model_paths:
        raise ValueError(f"Unsupported model: {model_name}")
    
    return AutoModelForCausalLM.from_pretrained(
        model_paths[model_name],
        device_map=device,
        **kwargs
    )

# Usage:
model = registry.get_model("qwen3", model_name="Qwen3-7B", device="cuda")
```

**Option 2: Register individual models with generation prefix**

```python
@registry.register_model(
    "qwen3/Qwen3-7B",
    model_family="qwen",
    model_generation="qwen3",
    model_name="Qwen3-7B",
)
def create_qwen3_7b(device: str = "cuda", **kwargs):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-7B",
        device_map=device,
        **kwargs
    )

# Usage:
model = registry.get_model("qwen3/Qwen3-7B", device="cuda")
```

**Why organize by generation?** Different Qwen generations (qwen, qwen2, qwen3) may have different architectures, tokenizers, and configurations even at the same parameter count.

**Option 3: Use pre-registered models (Recommended)**

Qwen models are automatically registered when you import the framework:

```python
from llm_diagnose.registry.model_registry import get_model_registry

# Models are already registered - just use them!
registry = get_model_registry()
model = registry.get_model("qwen3", model_name="Qwen3-7B", device="cuda")
```

Or simply import the framework and models are ready:

```python
import llm_diagnose  # Qwen models are auto-registered
from llm_diagnose.registry.model_registry import get_model_registry

registry = get_model_registry()
model = registry.get_model("qwen3", model_name="Qwen3-7B", device="cuda")
```

See `llm_diagnose/models/qwen.py` and `examples/qwen3_mmlu_evaluation.py`
for a full evaluation pipeline.

#### Pre-registered resources (models + datasets)

The framework ships with ready-to-use registrations that are loaded automatically
when you `import llm_diagnose`:

- **Models**: Qwen, Qwen2, Qwen3 (all subject sizes, e.g., `Qwen3-32B`)
- **Datasets**: MMLU (all 57 subjects)

```python
import llm_diagnose
from llm_diagnose.registry.model_registry import get_model_registry
from llm_diagnose.registry.dataset_registry import get_dataset_registry

model_registry = get_model_registry()
dataset_registry = get_dataset_registry()

model = model_registry.get_model("qwen3", model_name="Qwen3-32B", device="cuda")
dataset = dataset_registry.get_dataset("mmlu", subjects="astronomy", split="validation")
```

> ℹ️ The built-in MMLU loader relies on Hugging Face `datasets`.  
> Install it with `pip install datasets` if you plan to use the bundled datasets.

### 2. Load Configuration

```python
from llm_diagnose import ConfigLoader

# Load from file
config = ConfigLoader.from_file("config.yaml")

# Or create from dictionary
config = ConfigLoader.from_dict({
    "model": "gpt2",
    "dataset": "glue_sst2",
    "evaluator": {
        "attribution_method": "gradient",
        "target_layers": ["layer_0", "layer_1"],
    },
})
```

### 3. Create and Use Evaluators

```python
from llm_diagnose import NeuronAttributionEvaluator

# Create an evaluator
evaluator = NeuronAttributionEvaluator(
    name="gradient_attribution",
    attribution_method="gradient",
)

# Get model and dataset
model = model_registry.get_model("gpt2")
dataset = dataset_registry.get_dataset("glue_sst2")

# Run evaluation
results = evaluator.evaluate(
    model,
    dataset,
    target_layers=["layer_0", "layer_1"],
    top_k=10,
)
```

### 4. Create Custom Evaluators

```python
from llm_diagnose import NeuronAttributionEvaluator
from llm_diagnose.evaluators.registry import get_evaluator_registry

class CustomAttributionEvaluator(NeuronAttributionEvaluator):
    def _compute_attributions(self, model, dataset, target_layers, target_neurons=None):
        # Your custom attribution logic here
        attributions = {}
        # ... implementation ...
        return attributions

# Register the evaluator
registry = get_evaluator_registry()
registry.register_evaluator("custom_attribution")(CustomAttributionEvaluator)

# Use it
evaluator = registry.create_evaluator("custom_attribution", attribution_method="custom")
```

### 5. Summarize Results

```python
from llm_diagnose import BaseSummarizer

summarizer = BaseSummarizer(name="default")

summary = summarizer.summarize(results, benchmark="glue_sst2")

# Format as markdown
report = summarizer.format_report(summary, format="markdown")
print(report)
```

## Architecture

### Core Components

1. **Registry System** (`llm_diagnose/registry/`)
   - `BaseRegistry`: Generic registry pattern
   - `ModelRegistry`: Model registration and retrieval
   - `DatasetRegistry`: Dataset registration and retrieval

2. **Configuration** (`llm_diagnose/config/`)
   - `ConfigLoader`: Load and manage YAML/JSON configurations
   - Supports dot notation for nested access
   - Merge multiple configurations

3. **Evaluators** (`llm_diagnose/evaluators/`)
   - `BaseEvaluator`: Abstract base class for all evaluators
   - `NeuronAttributionEvaluator`: Base for neuron attribution methods
   - `RepresentationEngineeringEvaluator`: Base for representation engineering methods
   - `EvaluatorRegistry`: Registry for evaluator classes

4. **Summarizers** (`llm_diagnose/summarizers/`)
   - `BaseSummarizer`: Abstract base class for all summarizers
   - `SummarizerRegistry`: Registry for summarizer classes
   - Multiple output formats (dict, JSON, Markdown, text)

## Extending the Framework

### Adding a New Evaluator

1. Inherit from `NeuronAttributionEvaluator` or `RepresentationEngineeringEvaluator`
2. Implement the required abstract methods
3. Register it using the evaluator registry

```python
from llm_diagnose import NeuronAttributionEvaluator
from llm_diagnose.evaluators.registry import get_evaluator_registry

class MyEvaluator(NeuronAttributionEvaluator):
    def _compute_attributions(self, model, dataset, target_layers, target_neurons=None):
        # Your implementation
        pass

# Register
registry = get_evaluator_registry()
registry.register_evaluator("my_evaluator")(MyEvaluator)
```

### Adding a New Summarizer

1. Inherit from `BaseSummarizer`
2. Implement the `summarize` method
3. Register it using the summarizer registry

```python
from llm_diagnose import BaseSummarizer
from llm_diagnose.summarizers.registry import get_summarizer_registry

class MySummarizer(BaseSummarizer):
    def summarize(self, results, benchmark=None, **kwargs):
        # Your summarization logic
        return {"summary": "..."}

# Register
registry = get_summarizer_registry()
registry.register_summarizer("my_summarizer")(MySummarizer)
```

## Example Configuration File

```yaml
# config.yaml
model:
  name: gpt2
  device: cuda
  batch_size: 32

dataset:
  name: glue_sst2
  split: test
  max_samples: 1000

evaluator:
  type: neuron_attribution
  attribution_method: gradient
  target_layers:
    - layer_0
    - layer_1
  top_k: 10

summarizer:
  type: benchmark
  format: markdown
```

## Examples

See the `examples/` directory for complete usage examples:
- `example_usage.py`: Comprehensive examples of all framework features

## Development

```bash
# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black llm_diagnose/

# Type checking
mypy llm_diagnose/
```

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

