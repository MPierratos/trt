"""LitServe server implementations for ResNet50 models."""

import argparse
import json
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

import litserve as ls
import numpy as np
import torch
import openvino as ov


@dataclass
class ModelConfig:
    """Configuration for LitServe model deployment.
    
    Attributes:
        model_name: Name identifier for the model
        accelerator: Device type ("auto", "cpu", "cuda")
        max_batch_size: Maximum batch size for inference
        num_devices: Number of devices to use
        input_shape: Expected input tensor shape (excluding batch dimension)
        output_shape: Expected output tensor shape (excluding batch dimension)
        description: Optional model description
    """
    model_name: str
    accelerator: str = "auto"
    max_batch_size: int = 1
    num_devices: int = 1
    input_shape: List[int] = field(default_factory=lambda: [3, 224, 224])
    output_shape: List[int] = field(default_factory=lambda: [1000])
    description: Optional[str] = None
    
    @classmethod
    def from_json(cls, config_path: pathlib.Path) -> "ModelConfig":
        """Load configuration from a JSON file.
        
        Args:
            config_path: Path to the config.json file
            
        Returns:
            ModelConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is not valid JSON
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        return cls(**data)
    
    @classmethod
    def get_default(cls, model_type: str) -> "ModelConfig":
        """Get default configuration for a model type.
        
        Args:
            model_type: Either "libtorch" or "openvino"
            
        Returns:
            ModelConfig with default values
        """
        return cls(
            model_name=f"resnet50_{model_type}",
            accelerator="auto",
            max_batch_size=1,
            num_devices=1,
        )


def load_model_config(model_type: str, models_dir: pathlib.Path) -> ModelConfig:
    """Load configuration from the model's config.json file.
    
    Args:
        model_type: Either "libtorch" or "openvino"
        models_dir: Path to the litserve/models directory
        
    Returns:
        ModelConfig instance
    """
    config_path = models_dir / model_type / "config.json"
    return ModelConfig.from_json(config_path)


class ResnetLibtorchAPI(ls.LitAPI):
    """LitServe API for ResNet50 using PyTorch LibTorch backend."""
    
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = pathlib.Path(model_path)
        self.model = None
        self.device = None
    
    def setup(self, device):
        """Load the TorchScript model and move it to the appropriate device."""
        self.device = device
        self.model = torch.jit.load(str(self.model_path), map_location=device)
        self.model.eval()
    
    def decode_request(self, request):
        """Decode and validate the incoming request.
        
        Args:
            request: Dictionary with "input" key containing image data
            
        Returns:
            Tensor ready for inference [batch_size, 3, 224, 224]
        """
        input_data = request.get("input")
        if input_data is None:
            raise ValueError("Request must contain 'input' key")
        
        # Convert to tensor if needed
        if isinstance(input_data, list):
            input_tensor = torch.tensor(input_data, dtype=torch.float32, device=self.device)
        elif isinstance(input_data, np.ndarray):
            input_tensor = torch.from_numpy(input_data).float().to(self.device)
        else:
            input_tensor = input_data
        
        # Ensure correct shape [batch_size, 3, 224, 224]
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        return input_tensor
    
    def predict(self, x):
        """Run inference on the input tensor.
        
        Args:
            x: Input tensor [batch_size, 3, 224, 224]
            
        Returns:
            Output tensor [batch_size, 1000]
        """
        with torch.no_grad():
            return self.model(x)
    
    def encode_response(self, output):
        """Encode the model output for the response.
        
        Args:
            output: Model output tensor [batch_size, 1000]
            
        Returns:
            Dictionary with serialized output
        """
        return {"output": output.cpu().numpy().tolist()}


class ResnetOpenVINOAPI(ls.LitAPI):
    """LitServe API for ResNet50 using OpenVINO backend."""
    
    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = pathlib.Path(model_path)
        self.compiled_model = None
        self.output_tensor = None
    
    def setup(self, device):
        """Load the OpenVINO model and compile it.
        
        Note: OpenVINO always uses CPU regardless of the device parameter.
        """
        core = ov.Core()
        model = core.read_model(str(self.model_path))
        self.compiled_model = core.compile_model(model, "CPU")
        # Cache the output tensor reference
        self.output_tensor = self.compiled_model.outputs[0]
    
    def decode_request(self, request):
        """Decode and validate the incoming request.
        
        Args:
            request: Dictionary with "input" key containing image data
            
        Returns:
            NumPy array ready for inference [batch_size, 3, 224, 224]
        """
        input_data = request.get("input")
        if input_data is None:
            raise ValueError("Request must contain 'input' key")
        
        if isinstance(input_data, torch.Tensor):
            input_array = input_data.cpu().numpy().astype(np.float32)
        else:
            input_array = np.asarray(input_data, dtype=np.float32)
        
        # Ensure correct shape [batch_size, 3, 224, 224]
        if len(input_array.shape) == 3:
            input_array = np.expand_dims(input_array, axis=0)
        
        return input_array
    
    def predict(self, x):
        """Run inference on the input array.
        
        Args:
            x: Input array [batch_size, 3, 224, 224]
            
        Returns:
            Output array [batch_size, 1000]
        """
        return self.compiled_model([x])[self.output_tensor]
    
    def encode_response(self, output):
        """Encode the model output for the response.
        
        Args:
            output: Model output array [batch_size, 1000]
            
        Returns:
            Dictionary with serialized output
        """
        return {"output": output.tolist()}


def main():
    """Main entry point for running LitServe servers."""
    parser = argparse.ArgumentParser(
        description="LitServe server for ResNet50 models"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["libtorch", "openvino"],
        required=True,
        help="Model backend type: libtorch or openvino",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="inference_servers/litserve/models",
        help="Path to LitServe models directory (default: inference_servers/litserve/models)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on (default: 8000)",
    )
    parser.add_argument(
        "--max-batch-size-override",
        type=int,
        default=None,
        help="Override max_batch_size from config.json",
    )
    parser.add_argument(
        "--accelerator-override",
        type=str,
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Override accelerator from config.json",
    )
    
    args = parser.parse_args()
    
    # Determine model paths
    models_dir = pathlib.Path(args.models_dir)
    if args.model_type == "libtorch":
        model_path = models_dir / "libtorch" / "model.pt"
    else:
        model_path = models_dir / "openvino" / "model.xml"
    
    # Load configuration with defaults
    try:
        config = load_model_config(args.model_type, models_dir)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load config.json: {e}")
        print("Using defaults: max_batch_size=1, accelerator=auto, num_devices=1")
        config = ModelConfig.get_default(args.model_type)
    
    # Extract configuration with override precedence
    max_batch_size = args.max_batch_size_override or config.max_batch_size
    accelerator = args.accelerator_override or config.accelerator
    num_devices = config.num_devices
    model_name = config.model_name
    
    # Verify model file exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create the appropriate API
    if args.model_type == "libtorch":
        api = ResnetLibtorchAPI(str(model_path))
    else:
        api = ResnetOpenVINOAPI(str(model_path))
    
    # Create and run the server
    print(f"Starting LitServe server for {model_name}")
    print(f"  Model path: {model_path}")
    print(f"  Max batch size: {max_batch_size}")
    print(f"  Accelerator: {accelerator}")
    print(f"  Number of devices: {num_devices}")
    print(f"  Port: {args.port}")
    
    server = ls.LitServer(api, accelerator=accelerator, devices=num_devices, max_batch_size=max_batch_size)
    server.run(port=args.port)


if __name__ == "__main__":
    main()

