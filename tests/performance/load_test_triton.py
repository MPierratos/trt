import time
import logging
import numpy as np
import pathlib
from PIL import Image
from torchvision import transforms
import tritonclient.http as httpclient
from locust import User, task, events, constant
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "data"
IMAGE_PATH = DATA_DIR / "img1.jpg"

def rn50_preprocess(img_path: str = IMAGE_PATH) -> np.ndarray:
    img = Image.open(img_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess(img).numpy()

# Preprocess once for all requests
TRANSFORMED_IMG = rn50_preprocess(IMAGE_PATH)

DEFAULT_MODEL_CONFIGS = {
    "resnet50_libtorch": {
        "input_name": "input__0",
        "output_name": "output__0",
    },
    "resnet50_openvino": {
        "input_name": "x",
        "output_name": "x.45",
    },
}

def _sanitize_host(host: str) -> str:
    """Locust provides host via -H; Triton client accepts "host:port" or "http(s)://host:port"""
    if host.startswith("http://"):
        host = host[len("http://") :]
    elif host.startswith("https://"):
        host = host[len("https://") :]
    return host.rstrip("/")

class TritonClient:
    def __init__(self, host: str, model_name: str, config: dict = None):
        self.host = _sanitize_host(host)
        self.model_name = model_name
        self.config = config or {}
        self.client = httpclient.InferenceServerClient(url=self.host, connection_timeout=30.0, network_timeout=60.0)
        self.input_name, self.output_name = self._resolve_io_names()

    def _resolve_io_names(self) -> tuple[str, str]:
        try:
            md = self.client.get_model_metadata(model_name=self.model_name)
            inputs = md.get("inputs", [])
            outputs = md.get("outputs", [])
            if inputs and outputs:
                return inputs[0]["name"], outputs[0]["name"]
        except Exception as e:
            logger.error(f"Falling back to defaults for IO names: {e}")
        # Fallback to defaults
        cfg = DEFAULT_MODEL_CONFIGS.get(self.model_name)
        if cfg:
            return cfg["input_name"], cfg["output_name"]
        # Last resort: common ResNet50 style names
        return "input__0", "output__0"
    
    def send(self):
        request_meta = {
            "request_type": "Infer",
            "name": "Triton",
            "start_time": time.time(),
            "response_length": 0,
            "context": {
                "model_name": self.model_name,
            },
            "exception": None
        }
        start_perf_counter = time.perf_counter()
        try:
            # Triton expects batch dimension; add it for single image -> [1, 3, 224, 224]
            batched = np.expand_dims(TRANSFORMED_IMG, axis=0).astype(np.float32)
            infer_input = httpclient.InferInput(self.input_name, batched.shape, datatype="FP32")
            infer_input.set_data_from_numpy(batched, binary_data=True)
            infer_output = httpclient.InferRequestedOutput(self.output_name, binary_data=True, class_count=1000)
            results = self.client.infer(model_name=self.model_name, inputs=[infer_input], outputs=[infer_output])
            inference_output = results.as_numpy(self.output_name)
            
            if inference_output is not None:
                request_meta["response_length"] = int(inference_output.nbytes)

        except Exception as e:
            request_meta["exception"] = str(e)
            error_msg = str(e)
            error_type = type(e).__name__
            request_meta["context"]["error_msg"] = error_msg
            request_meta["context"]["error_type"] = error_type
            if not hasattr(self, "_error_logged"):
                self._error_logged = set()
            error_key = f"{error_type}: {error_msg[:100]}"
            if len(self._error_logged) < 5 and error_key not in self._error_logged:
                logging.error(f"Triton inference error: {error_type} - {error_msg}")
                self._error_logged.add(error_key)
            raise
        finally:
            request_meta["response_time"] = (time.perf_counter() - start_perf_counter) * 1000.0 # ms
            events.request.fire(**request_meta)

class TritonUser(User):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.model_name = os.environ.get("MODEL_NAME")
        if not self.model_name:
            raise RuntimeError("MODEL_NAME environment variable is required")
        config = getattr(self, 'config', {})
        self.client = TritonClient(self.host, self.model_name, config)


class MyUser(TritonUser):
    # For maximum load testing, set wait_time to 0 or use constant_pacing
    # For simulating real-world traffic (e.g., 30fps), use: wait_time = constant(1/30.0)
    wait_time = constant(0)

    @task
    def send_request(self):
        self.client.send()
