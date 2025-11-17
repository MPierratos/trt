import os
import sys
import time
import logging
import json
import numpy as np
from PIL import Image
from torchvision import transforms
from locust import FastHttpUser, task, events, constant
from inference_perf import PROJECT_PATH


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_PATH / "data"
IMAGE_PATH = DATA_DIR / "img1.jpg"


def rn50_preprocess(img_path: str = IMAGE_PATH) -> np.ndarray:
    """
    Preprocess image for ResNet50 model.
    
    Args:
        img_path: Path to the image file
        
    Returns:
        np.ndarray: Preprocessed image tensor with shape [3, 224, 224]
    """
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
    """
    Sanitize host string for HTTP requests.

    - Strips any trailing slashes.
    - Adds 'http://' if missing scheme.
    """
    host = host.strip().rstrip("/")
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"http://{host}"
    return host

class InferenceClient:
    """
    HTTP client for sending inference requests to LitServe or Triton servers.
    
    This client ensures fair performance comparison between servers by:
    
    **Identical behaviors (for fair comparison):**
    - Same HTTP session (FastHttpSession) with automatic connection pooling
    - Same timeout values: 60.0 seconds for network timeout
    - Same tensor preprocessing: identical input data shape [1, 3, 224, 224] and dtype float32
    - Same connection behavior: automatic keep-alive (handled by FastHttpSession)
    - Same error handling: both check response status codes
    - Same timing: both use time.perf_counter() for accurate measurements
    - Same request rate: controlled by Locust user classes, not by client code
    
    **Intentionally different (required for server APIs):**
    - Request format:
      - LitServe: JSON payload (json=payload)
      - Triton: Binary format by default (data=request_body with JSON header + binary tensor)
        - Can be switched to JSON format via TRITON_USE_JSON=true env var or config["use_json"]=True
        - Binary format is optimal for performance; JSON format allows format parity testing
    - Headers:
      - LitServe: Uses session defaults + auto Content-Type from json=
      - Triton (binary): Explicitly sets Content-Type and Inference-Header-Content-Length
      - Triton (JSON): Uses session defaults + auto Content-Type from json=
    - Response parsing: Different formats (expected due to different response structures)
    
    These differences are protocol requirements and do not bias performance comparisons.
    Both servers receive semantically equivalent requests (same tensor data, same timing).
    """
    def __init__(self, session, host: str, model_name: str, server_type: str, config: dict = None):
        """
        Initialize InferenceClient.
        
        Args:
            session: FastHttpSession instance from FastHttpUser
            host: Server host URL
            model_name: Name of the model to use for inference
            server_type: Type of server ('litserve' or 'triton')
            config: Optional configuration dictionary
        """
        self.session = session
        self.host = _sanitize_host(host)
        self.model_name = model_name
        self.server_type = server_type.lower()
        self.config = config or {}
        self._error_logged = set()
        
        if self.server_type == "litserve":
            self.url = f"{self.host}/predict"
        elif self.server_type == "triton":
            self.url = f"{self.host}/v2/models/{self.model_name}/infer"
            self.input_name, self.output_name = self._resolve_io_names()
            # Check if JSON format is requested (default: binary for performance)
            # Can be set via config dict or TRITON_USE_JSON environment variable
            use_json = self.config.get("use_json", False)
            if not use_json:
                use_json = os.environ.get("TRITON_USE_JSON", "").lower() in ("true", "1", "yes")
            self.use_json_format = use_json
        else:
            raise ValueError(f"Unknown server type: {server_type}")
    
    def _resolve_io_names(self) -> tuple[str, str]:
        """
        Resolve input/output names for Triton model from metadata endpoint.
        
        Returns:
            tuple[str, str]: (input_name, output_name) tuple
        """
        try:
            metadata_url = f"{self.host}/v2/models/{self.model_name}"
            response = self.session.get(metadata_url, timeout=10.0)
            if response.status_code >= 400:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
            md = response.json()
            inputs = md.get("inputs", [])
            outputs = md.get("outputs", [])
            if inputs and outputs:
                return inputs[0]["name"], outputs[0]["name"]
        except Exception as e:
            logger.error(f"Failed to get model metadata, falling back to defaults: {e}")
        
        # Fallback to defaults
        cfg = DEFAULT_MODEL_CONFIGS.get(self.model_name)
        if cfg:
            return cfg["input_name"], cfg["output_name"]
        # Last resort: common ResNet50 style names
        return "input__0", "output__0"
    
    def _prepare_batched_tensor(self) -> np.ndarray:
        """
        Prepare batched tensor for inference requests.
        
        Returns a batched tensor with shape [1, 3, 224, 224] and dtype float32.
        This ensures both LitServe and Triton receive identical input data.
        
        Returns:
            np.ndarray: Batched tensor ready for inference
        """
        # Prepare input: single image [3, 224, 224] -> wrap in batch [1, 3, 224, 224]
        return np.expand_dims(TRANSFORMED_IMG, axis=0).astype(np.float32)
    
    def send(self):
        """
        Send inference request to the configured server.
        
        This method handles request metadata, timing, error handling, and event firing
        identically for both LitServe and Triton. The only differences are in the
        protocol-specific request format (JSON vs binary), which are handled by
        _send_litserve() and _send_triton() respectively.
        """
        request_meta = {
            "request_type": "infer",
            "name": self.server_type.capitalize(),
            "start_time": time.time(),
            "response_length": 0,
            "context": {
                "model_name": self.model_name,
            },
            "exception": None
        }
        start_perf_counter = time.perf_counter()
        
        try:
            if self.server_type == "litserve":
                self._send_litserve(request_meta)
            elif self.server_type == "triton":
                self._send_triton(request_meta)
        except Exception as e:
            request_meta["exception"] = str(e)
            error_msg = str(e)
            error_type = type(e).__name__
            request_meta["context"]["error_msg"] = error_msg
            request_meta["context"]["error_type"] = error_type
            error_key = f"{error_type}: {error_msg[:100]}"
            if len(self._error_logged) < 5 and error_key not in self._error_logged:
                logging.error(f"{self.server_type.capitalize()} inference error: {error_type} - {error_msg}")
                self._error_logged.add(error_key)
            raise
        finally:
            request_meta["response_time"] = (time.perf_counter() - start_perf_counter) * 1000.0
            events.request.fire(**request_meta)
    
    def _send_litserve(self, request_meta: dict):
        """
        Send inference request to LitServe server using JSON format.
        
        Uses the same session, timeout, and tensor preparation as Triton.
        Only differs in request format (JSON payload) and response parsing.
        """
        # Prepare batched tensor (identical to Triton)
        batched = self._prepare_batched_tensor()
        
        # Create JSON payload
        payload = {"input": batched.tolist()}
        
        # Send POST request with timeout
        # Using json=payload automatically sets Content-Type: application/json
        response = self.session.post(
            self.url,
            json=payload,
            timeout=60.0
        )
        if response.status_code >= 400:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        # Parse response
        result = response.json()
        output = result.get("output", [])
        
        # Calculate response length (approximate size for metrics)
        if output:
            request_meta["response_length"] = sys.getsizeof(str(output))
    
    def _send_triton(self, request_meta: dict):
        """
        Send inference request to Triton server.
        
        Supports both binary format (default, optimal performance) and JSON format.
        Format is controlled by self.use_json_format (from config or TRITON_USE_JSON env var).
        
        Uses the same session, timeout, and tensor preparation as LitServe.
        Only differs in request format and response parsing.
        """
        # Prepare batched tensor (identical to LitServe)
        batched = self._prepare_batched_tensor()
        
        if self.use_json_format:
            # JSON format: embed tensor data directly in JSON payload
            self._send_triton_json(request_meta, batched)
        else:
            # Binary format: JSON header + binary tensor data (optimal performance)
            self._send_triton_binary(request_meta, batched)
    
    def _send_triton_json(self, request_meta: dict, batched: np.ndarray):
        """Send Triton request using JSON format (tensor data embedded in JSON)."""
        # Create Triton v2 REST API JSON payload with tensor data embedded
        payload = {
            "inputs": [
                {
                    "name": self.input_name,
                    "shape": list(batched.shape),
                    "datatype": "FP32",
                    "data": batched.flatten().tolist()  # Flatten and convert to list
                }
            ],
            "outputs": [
                {
                    "name": self.output_name
                }
            ]
        }
        
        # Send POST request with JSON payload
        # Uses same session and timeout as LitServe for fair comparison
        response = self.session.post(
            self.url,
            json=payload,
            timeout=60.0
        )
        if response.status_code >= 400:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        # Parse JSON response
        result = response.json()
        outputs = result.get("outputs", [])
        if outputs and len(outputs) > 0:
            output_data = outputs[0].get("data", [])
            request_meta["response_length"] = sys.getsizeof(str(output_data))
    
    def _send_triton_binary(self, request_meta: dict, batched: np.ndarray):
        """Send Triton request using binary format (JSON header + binary tensor data)."""
        # Create Triton v2 REST API JSON header (without data field for binary format)
        json_header = {
            "inputs": [
                {
                    "name": self.input_name,
                    "shape": list(batched.shape),
                    "datatype": "FP32",
                    "parameters": {
                        "binary_data_size": int(batched.nbytes)
                    }
                }
            ],
            "outputs": [
                {
                    "name": self.output_name,
                    "parameters": {
                        "binary_data": True
                    }
                }
            ]
        }
        
        # Encode JSON header as UTF-8 bytes
        json_bytes = json.dumps(json_header).encode('utf-8')
        json_header_size = len(json_bytes)
        
        # Convert numpy array to binary (C-contiguous)
        tensor_bytes = batched.tobytes('C')
        
        # Construct request body: JSON header + binary tensor data
        request_body = json_bytes + tensor_bytes
        
        # Set headers for binary tensor format
        # Note: Triton requires Inference-Header-Content-Length header for binary format
        headers = {
            "Content-Type": "application/json",
            "Inference-Header-Content-Length": str(json_header_size)
        }
        
        # Send POST request with binary data
        # Uses same session and timeout as LitServe for fair comparison
        response = self.session.post(
            self.url,
            data=request_body,
            headers=headers,
            timeout=60.0
        )
        if response.status_code >= 400:
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        # Parse response (Triton returns binary format when requested)
        # Calculate response length (approximate size for metrics, comparable to LitServe)
        content_type = response.headers.get('Content-Type', '')
        if 'application/json' in content_type:
            # JSON response (fallback)
            result = response.json()
            outputs = result.get("outputs", [])
            if outputs and len(outputs) > 0:
                output_data = outputs[0].get("data", [])
                request_meta["response_length"] = sys.getsizeof(str(output_data))
        else:
            # Binary response format
            # Parse Inference-Header-Content-Length to find JSON boundary
            header_size = int(response.headers.get('Inference-Header-Content-Length', '0'))
            if header_size > 0:
                json_part = response.content[:header_size].decode('utf-8')
                result = json.loads(json_part)
                outputs = result.get("outputs", [])
                if outputs and len(outputs) > 0:
                    # Binary data follows JSON header
                    binary_data = response.content[header_size:]
                    # Calculate response length (bytes in binary data)
                    request_meta["response_length"] = len(binary_data)
            else:
                # Fallback: try JSON parsing
                try:
                    result = response.json()
                    outputs = result.get("outputs", [])
                    if outputs and len(outputs) > 0:
                        output_data = outputs[0].get("data", [])
                        request_meta["response_length"] = sys.getsizeof(str(output_data))
                except Exception:
                    # If JSON parsing fails, just log the error
                    logger.warning("Failed to parse Triton response as JSON or binary format")


def get_inference_server():
    """
    Get inference server type from environment variable.
    
    Returns:
        str: Server type ('litserve' or 'triton')
        
    Raises:
        RuntimeError: If INFERENCE_SERVER is not set or has invalid value
    """
    server = os.environ.get("INFERENCE_SERVER", "").lower()
    if server not in ["litserve", "triton"]:
        raise RuntimeError(
            f"INFERENCE_SERVER environment variable must be set to 'litserve' or 'triton'. "
            f"Got: {server}"
        )
    return server

class InferenceUser(FastHttpUser):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.model_name = os.environ.get("MODEL_NAME")
        if not self.model_name:
            raise RuntimeError("MODEL_NAME environment variable is required")
        
        # FastHttpUser creates self.client (FastHttpSession) - save it
        http_session = self.client
        
        server = get_inference_server()
        config = getattr(self, 'config', {})
        # Pass FastHttpSession to InferenceClient
        self.client = InferenceClient(http_session, self.host, self.model_name, server, config)


class LowUser(InferenceUser):
    """Low load user with constant pacing (30 requests per second per user)"""
    wait_time = constant(1/30.0)
    
    @task
    def send_request(self):
        self.client.send()


class HighUser(InferenceUser):
    """High load user with continuous sending (no wait between requests)"""
    wait_time = constant(0)
    
    @task
    def send_request(self):
        self.client.send()
