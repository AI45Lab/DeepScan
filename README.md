# LLM-Diagnose Framework

A flexible and extensible framework for diagnosing Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). This framework provides a modular architecture for evaluating models through neuron attribution and representation engineering techniques.

## Features

- **Model Registry**: Register and manage model instances and factories
- **Model Runners**: Consistent `generate`/`chat` abstraction across model families
- **Dataset Registry**: Register and manage dataset instances and factories
- **Configuration Management**: Load and manage configurations from YAML/JSON files
- **Extensible Evaluators**: Two main types of evaluators:
  - **Neuron Attribution**: Analyze which neurons contribute to model behavior
  - **Representation Engineering**: Analyze and manipulate model representations
- **Customizable Summarizers**: Aggregate and format evaluation results for different benchmarks
- **Plugin Architecture**: Easy to extend with custom evaluators and summarizers
- **TELLME Metrics**: Built-in evaluator for disentanglement metrics on filtered BeaverTails

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
# Hugging Face dataset loaders (BeaverTails + raw HF loading for TELLME/X-Boundary)
pip install datasets

# Qwen runner dependencies (shared by evaluators)
pip install -e ".[qwen]"

# TELLME evaluator + metrics stack
pip install -e ".[tellme]"

# X-Boundary evaluator + visualization stack
pip install -e ".[xboundary]"

# Everything (tellme + xboundary + runner deps)
pip install -e ".[all]"
```

### End-to-end from a config (any evaluator)
```python
from llm_diagnose import run_from_config

# YAML/JSON or dict with model/dataset/evaluator sections
results = run_from_config("examples/config.tellme.yaml")
```

CLI (no Python code needed):
```bash
python -m llm_diagnose.run --config examples/config.tellme.yaml --output-dir runs

# Optional: also write a single consolidated JSON to a specific location
python -m llm_diagnose.run --config examples/config.tellme.yaml --output results.json
```

## Quick Start

### 1. Register Models and Datasets (global registries)

#### Registering Individual Models

```python
from llm_diagnose.registry.model_registry import get_model_registry
from llm_diagnose.registry.dataset_registry import get_dataset_registry

# IMPORTANT: use the global registries so `run_from_config()` can find your entries
model_registry = get_model_registry()
dataset_registry = get_dataset_registry()

@model_registry.register_model("gpt2")
def create_gpt2():
    from transformers import GPT2LMHeadModel
    return GPT2LMHeadModel.from_pretrained("gpt2")

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
def create_qwen3(model_name: str = "Qwen3-8B", device: str = "cuda", **kwargs):
    """Create Qwen3 model of specified name."""
    from transformers import AutoModelForCausalLM
    
    model_paths = {
        "Qwen3-0.6B": "Qwen/Qwen3-0.6B",
        "Qwen3-1.5B": "Qwen/Qwen3-1.5B",
        "Qwen3-2B": "Qwen/Qwen3-2B",
        "Qwen3-8B": "Qwen/Qwen3-8B",
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
runner = registry.get_model("qwen3", model_name="Qwen3-8B", device="cuda")
```

**Option 2: Register individual models with generation prefix**

```python
@registry.register_model(
    "qwen3/Qwen3-8B",
    model_family="qwen",
    model_generation="qwen3",
    model_name="Qwen3-8B",
)
def create_qwen3_8b(device: str = "cuda", **kwargs):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        device_map=device,
        **kwargs
    )

# Usage:
runner = registry.get_model("qwen3/Qwen3-8B", device="cuda")
```

**Why organize by generation?** Different Qwen generations (qwen, qwen2, qwen3) may have different architectures, tokenizers, and configurations even at the same parameter count.

**Option 3: Use pre-registered models (Recommended)**

Qwen models are automatically registered when you import the framework:

```python
from llm_diagnose.registry.model_registry import get_model_registry

# Models are already registered - just use them!
registry = get_model_registry()
runner = registry.get_model("qwen3", model_name="Qwen3-8B", device="cuda")
```

Or simply import the framework and models are ready:

```python
import llm_diagnose  # Qwen models are auto-registered
from llm_diagnose.registry.model_registry import get_model_registry

registry = get_model_registry()
runner = registry.get_model("qwen3", model_name="Qwen3-8B", device="cuda")
```

See `llm_diagnose/models/qwen.py` and `examples/tellme_evaluation.py`
for an end-to-end evaluation pipeline.

#### Pre-registered resources (models + datasets)

The framework ships with ready-to-use registrations that are loaded automatically
when you `import llm_diagnose`:

- **Models**: Qwen (qwen / qwen2 / qwen2.5 / qwen3 variants; see `llm_diagnose/models/qwen.py`)
- **Datasets**: BeaverTails (HF dataset) and `tellme/beaver_tails_filtered` (CSV loader)

```python
import llm_diagnose
from llm_diagnose.registry.model_registry import get_model_registry
from llm_diagnose.registry.dataset_registry import get_dataset_registry

model_registry = get_model_registry()
dataset_registry = get_dataset_registry()

model = model_registry.get_model("qwen3", model_name="Qwen3-8B", device="cuda")
dataset = dataset_registry.get_dataset("beaver_tails", split="330k_train")
```

> ℹ️ The built-in dataset loaders rely on Hugging Face `datasets`.  
> Install it with `pip install datasets` if you plan to use the bundled datasets.

To use a copy saved via `datasets.save_to_disk`, pass a local path:

```python
dataset = dataset_registry.get_dataset(
    "beaver_tails",
    split="330k_train",
    path="/path/to/BeaverTails",
)
```

### Model Runners

Model registry lookups now return a **model runner**—an object that exposes a uniform
`generate()` interface and keeps the underlying Hugging Face model/tokenizer handy.

```python
from llm_diagnose.models.base_runner import GenerationRequest, PromptMessage, PromptContent
from llm_diagnose.registry.model_registry import get_model_registry

runner = get_model_registry().get_model(
    "qwen3",
    model_name="Qwen3-8B",
    device="cuda",
)

# Quick text generation
response = runner.generate("Explain what a registry pattern is in two sentences.")
print(response.text)

# Chat-style prompt with structured messages
chat_request = GenerationRequest.from_messages(
    [
        PromptMessage(role="system", content=[PromptContent(text="You are a math tutor.")]),
        PromptMessage(role="user", content=[PromptContent(text="Help me factor x^2 + 5x + 6.")]),
    ],
    temperature=0.1,
    max_new_tokens=128,
)
chat_response = runner.generate(chat_request)
print(chat_response.text)
```

Runners keep the raw model/tokenizer accessible via `runner.model` / `runner.tokenizer`
so existing diagnostic code can still reach low-level APIs when necessary.

### 2. Load Configuration

```python
from llm_diagnose import ConfigLoader

# Load from file
config = ConfigLoader.from_file("config.yaml")

# Or create from dictionary
config = ConfigLoader.from_dict({
    "model": {"generation": "qwen3", "model_name": "Qwen3-8B", "device": "cuda"},
    "dataset": {"name": "beaver_tails", "split": "330k_train"},
    "evaluator": {"type": "tellme", "batch_size": 4},
})
```

### 3. Create and Use Evaluators

```python
from llm_diagnose import NeuronAttributionEvaluator
from llm_diagnose.registry.model_registry import get_model_registry
from llm_diagnose.registry.dataset_registry import get_dataset_registry

# Create an evaluator
evaluator = NeuronAttributionEvaluator(
    name="gradient_attribution",
    attribution_method="gradient",
)

# Get model and dataset
model_registry = get_model_registry()
dataset_registry = get_dataset_registry()

model = model_registry.get_model("qwen3", model_name="Qwen3-8B", device="cuda")
dataset = dataset_registry.get_dataset("beaver_tails", split="330k_train")

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
from llm_diagnose.summarizers.base import BaseSummarizer

class SimpleSummarizer(BaseSummarizer):
    def summarize(self, results, benchmark=None, **kwargs):
        # Minimal example: keep a small subset of keys
        return {
            "benchmark": benchmark,
            "keys": sorted(results.keys()),
        }

summarizer = SimpleSummarizer(name="simple")

summary = summarizer.summarize(results, benchmark="beaver_tails")

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
# Minimal TELLME-style config (see `examples/config.tellme.yaml` for the full version)
model:
  generation: qwen3
  model_name: Qwen3-8B
  device: cuda
  dtype: float16
  # Optional: point to a local checkpoint dir to avoid downloads
  # path: /path/to/models--Qwen--Qwen3-8B

dataset:
  name: tellme/beaver_tails_filtered
  test_path: /path/to/test.csv
  # train_path: /path/to/train.csv
  # max_rows: 400

evaluator:
  type: tellme
  batch_size: 4
  layer_ratio: 0.6666
  token_position: -1
```

## Examples

See the `examples/` directory for complete usage examples:
- `tellme_evaluation.py`: Run TELLME disentanglement metrics on Qwen (config-driven)
- `config.tellme.yaml`: Ready-to-run config for TELLME
- `config.example.yaml`: Template config (requires you to register the referenced model/dataset)
- `QUICK_START_QWEN3.md`: Extra notes on Qwen registry usage

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

