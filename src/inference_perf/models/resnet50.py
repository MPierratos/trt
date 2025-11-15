import torchvision
import torch
import openvino as ov
from pathlib import Path

_model = None

def get_model():
    """Lazy load the ResNet50 model."""
    global _model
    if _model is None:
        _model = torchvision.models.resnet50(weights='DEFAULT').eval()
    return _model

def create_resnet_openvino(model_path='models/openvino/model.xml'):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = get_model()
    ov_model = ov.convert_model(model)
    # Set dynamic batch dimension (-1 means dynamic)
    # Get the input name from the model
    input_name = list(ov_model.inputs)[0].get_any_name()
    ov_model.reshape({input_name: ov.PartialShape([-1, 3, 224, 224])})
    ov.save_model(ov_model, str(model_path))

def create_resnet_libtorch(model_path='models/libtorch/model.pt'):
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model = get_model()
    # Use script to support dynamic batch sizes
    # If script fails, fall back to tracing with multiple batch sizes
    try:
        scripted_model = torch.jit.script(model)
        torch.jit.save(scripted_model, str(model_path))
    except Exception:
        # Fallback: trace with batch size 1, Triton will handle batching
        # by calling the model multiple times if needed
        traced_model = torch.jit.trace(model, torch.randn(1, 3, 224, 224))
        torch.jit.save(traced_model, str(model_path))

if __name__ == "__main__":
   create_resnet_openvino()
   create_resnet_libtorch()