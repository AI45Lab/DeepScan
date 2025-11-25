# Quick Start: Adding Qwen Models (Organized by Generation)

This guide shows you how to add Qwen model families to the LLM-Diagnose Framework.
Models are organized by **generation** (qwen, qwen2, qwen3) rather than size, since
different generations may have different configurations even at the same size.

## Method 1: Register by Generation (Recommended)

Register each generation separately with a factory that accepts model names:

```python
from llm_diagnose.registry.model_registry import get_model_registry

registry = get_model_registry()

@registry.register_model("qwen3", model_family="qwen", model_generation="qwen3")
def create_qwen3(model_name: str = "Qwen3-7B", device: str = "cuda", **kwargs):
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

## Method 2: Individual Models with Generation Prefix

Register individual models with generation prefix:

```python
from llm_diagnose.registry.model_registry import get_model_registry

registry = get_model_registry()

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

## Method 3: Using Pre-registered Models (Recommended)

Qwen models are **automatically registered** when you import the framework - no manual registration needed!

```python
# Just import the framework - models are already registered
import llm_diagnose
from llm_diagnose.registry.model_registry import get_model_registry

registry = get_model_registry()

# Individual model
model = registry.get_model("qwen3/Qwen3-7B", device="cuda")

# Or generation factory
model = registry.get_model("qwen3", model_name="Qwen3-7B", device="cuda")
```

The models are pre-registered, so you can use them immediately without any setup!

## Using Qwen Models in Evaluations

Qwen models are pre-registered, so you can use them directly:

```python
from llm_diagnose import ConfigLoader, NeuronAttributionEvaluator
from llm_diagnose.registry.model_registry import get_model_registry
from llm_diagnose.registry.dataset_registry import get_dataset_registry

# Models and datasets are already registered - no setup needed!

# Load configuration
config = ConfigLoader.from_dict({
    "model": {
        "generation": "qwen3",
        "model_name": "Qwen3-7B",
        "device": "cuda",
    },
    "evaluator": {
        "attribution_method": "gradient",
    },
})

# Get model
registry = get_model_registry()
model_config = config.get("model", {})
model = registry.get_model(
    model_config.get("generation", "qwen3"),
    model_name=model_config.get("model_name", "Qwen3-7B"),
    device=model_config.get("device", "cuda"),
)

# Get dataset (MMLU astronomy subject, validation split)
dataset_registry = get_dataset_registry()
dataset = dataset_registry.get_dataset(
    "mmlu",
    subjects="astronomy",
    split="validation",
)

# Create evaluator and run evaluation
evaluator = NeuronAttributionEvaluator(config=config.get("evaluator", {}))
# results = evaluator.evaluate(model, dataset, target_layers=["layer_0"])
```

## Configuration File Example

You can also specify Qwen models in YAML configuration:

```yaml
model:
  generation: qwen3  # or qwen, qwen2
  model_name: Qwen3-7B
  device: cuda
  load_in_8bit: false  # Optional: for quantization

evaluator:
  type: neuron_attribution
  attribution_method: gradient
  target_layers:
    - layer_0
    - layer_1
```

## Advanced Options

The Qwen registry supports advanced loading options:

```python
# With quantization
model = registry.get_model(
    "qwen3",
    model_name="Qwen3-7B",
    device="cuda",
    load_in_8bit=True,  # 8-bit quantization
)

# Or 4-bit
model = registry.get_model(
    "qwen3",
    model_name="Qwen3-7B",
    device="cuda",
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
)

# With custom torch dtype
model = registry.get_model(
    "qwen3",
    model_name="Qwen3-7B",
    device="cuda",
    torch_dtype="float16",
)

# Without tokenizer
model = registry.get_model(
    "qwen3",
    model_name="Qwen3-7B",
    device="cuda",
    load_tokenizer=False,
)
```

## Built-in MMLU dataset

The MMLU benchmark (all 57 subjects) is also pre-registered. You can load any
subject (or the entire benchmark) straight from the dataset registry:

```python
from llm_diagnose.registry.dataset_registry import get_dataset_registry

dataset_registry = get_dataset_registry()
astronomy = dataset_registry.get_dataset("mmlu", subjects="astronomy", split="validation")
all_subjects = dataset_registry.get_dataset("mmlu", split="test", return_dict=True)
```

> ℹ️ The bundled loader relies on the Hugging Face `datasets` library.  
> Install it with `pip install datasets` if you plan to use MMLU.

## Why Organize by Generation?

Different Qwen generations (qwen, qwen2, qwen3) may have:
- Different architectures
- Different tokenizers
- Different configuration formats
- Different loading requirements

Even models with the same parameter count (e.g., 7B) can have different configs across generations, so organizing by generation ensures proper handling of each model family.

## See Also

- `examples/qwen3_mmlu_evaluation.py` - End-to-end evaluation pipeline
- `llm_diagnose/models/qwen.py` - Production-ready Qwen registry module (all generations)

