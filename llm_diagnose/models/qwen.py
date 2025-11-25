"""
Qwen model registry organized by generation.

This module registers Qwen models by generation (qwen, qwen2, qwen3) since
different generations may have different configurations even at the same size.
"""

from llm_diagnose.registry.model_registry import get_model_registry
from typing import Optional, Dict, Any


# Qwen model configurations organized by generation
QWEN_MODELS = {
    "qwen": {
        "Qwen-0.5B": {
            "path": "Qwen/Qwen-0.5B",
            "params": "0.5B",
            "description": "Qwen 0.5B parameter model",
        },
        "Qwen-1.8B": {
            "path": "Qwen/Qwen-1.8B",
            "params": "1.8B",
            "description": "Qwen 1.8B parameter model",
        },
        "Qwen-7B": {
            "path": "Qwen/Qwen-7B",
            "params": "7B",
            "description": "Qwen 7B parameter model",
        },
        "Qwen-14B": {
            "path": "Qwen/Qwen-14B",
            "params": "14B",
            "description": "Qwen 14B parameter model",
        },
        "Qwen-72B": {
            "path": "Qwen/Qwen-72B",
            "params": "72B",
            "description": "Qwen 72B parameter model",
        },
    },
    "qwen2": {
        "Qwen2-0.5B": {
            "path": "Qwen/Qwen2-0.5B",
            "params": "0.5B",
            "description": "Qwen2 0.5B parameter model",
        },
        "Qwen2-1.5B": {
            "path": "Qwen/Qwen2-1.5B",
            "params": "1.5B",
            "description": "Qwen2 1.5B parameter model",
        },
        "Qwen2-7B": {
            "path": "Qwen/Qwen2-7B",
            "params": "7B",
            "description": "Qwen2 7B parameter model",
        },
        "Qwen2-72B": {
            "path": "Qwen/Qwen2-72B",
            "params": "72B",
            "description": "Qwen2 72B parameter model",
        },
    },
    "qwen3": {
        "Qwen3-0.6B": {
            "path": "Qwen/Qwen3-0.6B",
            "params": "0.6B",
            "description": "Qwen3 0.6B parameter model",
        },
        "Qwen3-1.5B": {
            "path": "Qwen/Qwen3-1.5B",
            "params": "1.5B",
            "description": "Qwen3 1.5B parameter model",
        },
        "Qwen3-2B": {
            "path": "Qwen/Qwen3-2B",
            "params": "2B",
            "description": "Qwen3 2B parameter model",
        },
        "Qwen3-7B": {
            "path": "Qwen/Qwen3-7B",
            "params": "7B",
            "description": "Qwen3 7B parameter model",
        },
        "Qwen3-14B": {
            "path": "Qwen/Qwen3-14B",
            "params": "14B",
            "description": "Qwen3 14B parameter model",
        },
        "Qwen3-32B": {
            "path": "Qwen/Qwen3-32B",
            "params": "32B",
            "description": "Qwen3 32B parameter model",
        },
    },
}


def register_qwen_models():
    """
    Register all Qwen models organized by generation.
    
    This function registers:
    1. Individual model variants (qwen/Qwen-7B, qwen2/Qwen2-7B, qwen3/Qwen3-7B, etc.)
    2. Generation factories (qwen, qwen2, qwen3) that can create any variant in that generation
    
    Usage:
        from llm_diagnose.models import register_qwen_models
        register_qwen_models()
        
        # Then use:
        from llm_diagnose.registry.model_registry import get_model_registry
        registry = get_model_registry()
        
        # Individual variant
        model = registry.get_model("qwen3/Qwen3-7B")
        
        # Or generation factory
        model = registry.get_model("qwen3", model_name="Qwen3-7B")
    """
    registry = get_model_registry()
    
    def _create_factory(generation: str, model_name: str, model_path: str, model_params: str, description: str):
        """Create a factory function for a specific Qwen model."""
        def factory(device: str = "cuda", **kwargs):
            """
            Create a Qwen model.
            
            Args:
                device: Device to load model on (default: "cuda")
                **kwargs: Additional arguments for model loading
                    - torch_dtype: Data type for model weights
                    - trust_remote_code: Whether to trust remote code
                    - load_in_8bit: Load in 8-bit mode
                    - load_in_4bit: Load in 4-bit mode
                    - load_tokenizer: Whether to load tokenizer (default: True)
            
            Returns:
                Loaded Qwen model (and tokenizer if load_tokenizer=True)
            """
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
            except ImportError:
                raise ImportError(
                    "transformers library is required. Install with: pip install transformers"
                )
            
            # Default arguments
            model_kwargs = {
                "device_map": device,
                "torch_dtype": kwargs.get("torch_dtype", "auto"),
                "trust_remote_code": kwargs.get("trust_remote_code", True),
            }
            
            # Add quantization options if specified
            if kwargs.get("load_in_8bit", False):
                model_kwargs["load_in_8bit"] = True
            elif kwargs.get("load_in_4bit", False):
                model_kwargs["load_in_4bit"] = True
                if "bitsandbytes" in kwargs:
                    model_kwargs["bnb_4bit_compute_dtype"] = kwargs.get("bnb_4bit_compute_dtype", "float16")
            
            # Add any other kwargs
            for key in ["max_memory", "offload_folder", "low_cpu_mem_usage"]:
                if key in kwargs:
                    model_kwargs[key] = kwargs[key]
            
            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
            
            # Optionally load tokenizer
            if kwargs.get("load_tokenizer", True):
                tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=kwargs.get("trust_remote_code", True),
                )
                return {"model": model, "tokenizer": tokenizer, "model_name": model_name}
            
            return model
        
        return factory
    
    # Register individual models and generation factories
    for generation, models in QWEN_MODELS.items():
        available_models = list(models.keys())
        
        # Register individual model variants
        for model_name, config in models.items():
            registry_name = f"{generation}/{model_name}"
            
            registry.register_model(
                registry_name,
                factory=_create_factory(
                    generation,
                    model_name,
                    config["path"],
                    config["params"],
                    config["description"],
                ),
                model_family="qwen",
                model_generation=generation,
                model_name=model_name,
                model_type="llm",
                parameters=config["params"],
                description=config["description"],
            )
        
        # Register generation factory (need to capture generation and models in closure)
        def _create_generation_factory(gen: str, models_dict: dict, available: list):
            @registry.register_model(
                gen,
                model_family="qwen",
                model_generation=gen,
                model_type="llm",
                available_models=available,
                description=f"Qwen {gen} model family factory",
            )
            def create_qwen_generation(
                model_name: str,
                device: str = "cuda",
                **kwargs
            ):
                """
                Create a Qwen model from the specified generation.
                
                Args:
                    model_name: Model name (e.g., "Qwen3-7B", "Qwen2-7B")
                    device: Device to load model on (default: "cuda")
                    **kwargs: Additional arguments for model loading
                        - torch_dtype: Data type for model weights
                        - trust_remote_code: Whether to trust remote code
                        - load_in_8bit: Load in 8-bit mode
                        - load_in_4bit: Load in 4-bit mode
                        - load_tokenizer: Whether to load tokenizer (default: True)
                
                Returns:
                    Loaded Qwen model (and tokenizer if load_tokenizer=True)
                    
                Raises:
                    ValueError: If model_name is not available in this generation
                """
                if model_name not in models_dict:
                    raise ValueError(
                        f"Model '{model_name}' not found in {gen}. "
                        f"Available models: {list(models_dict.keys())}"
                    )
                
                model_config = models_dict[model_name]
                model_path = model_config["path"]
                
                try:
                    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
                except ImportError:
                    raise ImportError(
                        "transformers library is required. Install with: pip install transformers"
                    )
                
                # Default arguments
                model_kwargs = {
                    "device_map": device,
                    "torch_dtype": kwargs.get("torch_dtype", "auto"),
                    "trust_remote_code": kwargs.get("trust_remote_code", True),
                }
                
                # Add quantization options if specified
                if kwargs.get("load_in_8bit", False):
                    model_kwargs["load_in_8bit"] = True
                elif kwargs.get("load_in_4bit", False):
                    model_kwargs["load_in_4bit"] = True
                    if "bitsandbytes" in kwargs:
                        model_kwargs["bnb_4bit_compute_dtype"] = kwargs.get("bnb_4bit_compute_dtype", "float16")
                
                # Add any other kwargs
                for key in ["max_memory", "offload_folder", "low_cpu_mem_usage"]:
                    if key in kwargs:
                        model_kwargs[key] = kwargs[key]
                
                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    **model_kwargs
                )
                
                # Optionally load tokenizer
                if kwargs.get("load_tokenizer", True):
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_path,
                        trust_remote_code=kwargs.get("trust_remote_code", True),
                    )
                    return {"model": model, "tokenizer": tokenizer, "model_name": model_name}
                
                return model
            
            return create_qwen_generation
        
        _create_generation_factory(generation, models, available_models)

