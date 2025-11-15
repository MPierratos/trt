"""LitServe server implementations for ResNet50 models."""

import argparse
import json
import pathlib
from typing import Dict, Any

import litserve as ls
import numpy as np
import torch
import openvino as ov


def load_model_config(model_type: str, models_dir: pathlib.Path) -> Dict[str, Any]:
    """Load configuration from the model's config.json file.
    
    Args:
        model_type: Either "libtorch" or "openvino"
        models_dir: Path to the litserve/models directory
        
    Returns:
        Dictionary with model configuration
    """
    config_path = models_dir / model_type / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


class ResnetLibtorchAPI(ls.LitAPI):
    """LitServe API for ResNet50 using PyTorch LibTorch backend."""
    
    def __init__(self, model_path: str, max_batch_size: int = 1):
        super().__init__(max_batch_size=max_batch_size)
        self.model_path = pathlib.Path(model_path)
        self.model = None
        self.device = None
    
    def setup(self, device):
        """Load the TorchScript model and move it to the appropriate device."""
        self.device = device
        self.model = torch.jit.load(str(self.model_path), map_location=device)
        self.model.eval()
        self.model.to(device)
    
    def predict(self, request):
        """Run inference on the input request.
        
        Args:
            request: Dictionary with "input" key containing image data
                    Shape: [batch_size, 3, 224, 224] or nested list
            
        Returns:
            Dictionary with "output" key containing logits [batch_size, 1000]
        """
        # Extract input from request
        input_data = request.get("input")
        
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
        
        # Run inference
        with torch.no_grad():
            output = self.model(input_tensor)
        
        # Convert to list for JSON serialization
        return {"output": output.cpu().numpy().tolist()}


class ResnetOpenVINOAPI(ls.LitAPI):
    """LitServe API for ResNet50 using OpenVINO backend."""
    
    def __init__(self, model_path: str, max_batch_size: int = 1):
        super().__init__(max_batch_size=max_batch_size)
        self.model_path = pathlib.Path(model_path)
        self.compiled_model = None
        self.device = None
    
    def setup(self, device):
        """Load the OpenVINO model and compile it."""
        self.device = device
        core = ov.Core()
        model = core.read_model(str(self.model_path))
        
        # Compile model for the device
        # OpenVINO uses "CPU" or "GPU" device names
        if device.type == "cuda":
            device_name = "GPU"
        else:
            device_name = "CPU"
        
        self.compiled_model = core.compile_model(model, device_name)
    
    def predict(self, request):
        """Run inference on the input request.
        
        Args:
            request: Dictionary with "input" key containing image data
                    Shape: [batch_size, 3, 224, 224] or nested list
            
        Returns:
            Dictionary with "output" key containing logits [batch_size, 1000]
        """
        # Extract input from request
        input_data = request.get("input")
        
        # Convert to numpy array if needed
        if isinstance(input_data, list):
            input_array = np.array(input_data, dtype=np.float32)
        elif isinstance(input_data, torch.Tensor):
            input_array = input_data.cpu().numpy().astype(np.float32)
        else:
            input_array = np.asarray(input_data, dtype=np.float32)
        
        # Ensure correct shape [batch_size, 3, 224, 224]
        if len(input_array.shape) == 3:
            input_array = np.expand_dims(input_array, axis=0)
        
        # Get input and output tensors
        input_tensor = self.compiled_model.inputs[0]
        output_tensor = self.compiled_model.outputs[0]
        
        # Run inference
        result = self.compiled_model([input_array])[output_tensor]
        
        # Convert to list for JSON serialization
        return {"output": result.tolist()}


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
    
    # Load configuration from config.json
    try:
        config = load_model_config(args.model_type, models_dir)
        max_batch_size = args.max_batch_size_override or config["max_batch_size"]
        accelerator = args.accelerator_override or config["accelerator"]
        model_name = config.get("model_name", f"resnet50_{args.model_type}")
    except Exception as e:
        print(f"Error: Could not load config.json: {e}")
        print("Using defaults: max_batch_size=1, accelerator=auto")
        max_batch_size = args.max_batch_size_override or 1
        accelerator = args.accelerator_override or "auto"
        model_name = f"resnet50_{args.model_type}"
    
    # Verify model file exists
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Create the appropriate API
    if args.model_type == "libtorch":
        api = ResnetLibtorchAPI(str(model_path), max_batch_size=max_batch_size)
    else:
        api = ResnetOpenVINOAPI(str(model_path), max_batch_size=max_batch_size)
    
    # Create and run the server
    print(f"Starting LitServe server for {model_name}")
    print(f"  Model path: {model_path}")
    print(f"  Max batch size: {max_batch_size}")
    print(f"  Accelerator: {accelerator}")
    print(f"  Port: {args.port}")
    
    server = ls.LitServer(api, accelerator=accelerator)
    server.run(port=args.port)


if __name__ == "__main__":
    main()

