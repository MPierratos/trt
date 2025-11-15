"""Parser for Triton config.pbtxt files to extract configuration for LitServe."""

import re
from pathlib import Path
from typing import Dict, Optional


def parse_config_pbtxt(config_path: Path) -> Dict[str, any]:
    """Parse a Triton config.pbtxt file and extract relevant configuration.
    
    Args:
        config_path: Path to the config.pbtxt file
        
    Returns:
        Dictionary with parsed configuration including:
        - max_batch_size: int
        - accelerator: str ("cuda" or "cpu")
        - model_name: str
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    content = config_path.read_text()
    
    # Extract max_batch_size
    max_batch_match = re.search(r'max_batch_size\s*:\s*(\d+)', content)
    max_batch_size = int(max_batch_match.group(1)) if max_batch_match else 1
    
    # Extract model name
    name_match = re.search(r'name\s*:\s*"([^"]+)"', content)
    model_name = name_match.group(1) if name_match else None
    
    # Extract instance_group kind
    accelerator = "cpu"  # default
    instance_group_match = re.search(r'instance_group\s*\[\s*\{\s*kind\s*:\s*(KIND_\w+)', content)
    if instance_group_match:
        kind = instance_group_match.group(1)
        if kind == "KIND_GPU":
            accelerator = "cuda"
        elif kind == "KIND_CPU":
            accelerator = "cpu"
    
    return {
        "max_batch_size": max_batch_size,
        "accelerator": accelerator,
        "model_name": model_name,
    }


def get_model_config(model_name: str, model_repo_path: Path) -> Dict[str, any]:
    """Get configuration for a specific model from the Triton model repository.
    
    Args:
        model_name: Name of the model (e.g., "resnet50_libtorch")
        model_repo_path: Path to the Triton model repository
        
    Returns:
        Dictionary with parsed configuration
    """
    config_path = model_repo_path / model_name / "config.pbtxt"
    return parse_config_pbtxt(config_path)

