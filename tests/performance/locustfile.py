import os
import sys
import time
import logging
import numpy as np
import requests
from PIL import Image
from torchvision import transforms
from locust import User, task, events, constant
from inference_perf import PROJECT_PATH

# Try importing tritonclient, but don't fail if not available
try:
    import tritonclient.http as httpclient
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    httpclient = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_PATH / "data"
IMAGE_PATH = DATA_DIR / "img1.jpg"

# Profile configuration: maps profile name to wait time in seconds
PROFILE_WAIT_TIMES = {
    "max_load": 0,        # No wait, maximum throughput
    "30fps": 1/30.0,      # 30 requests per second per user
}

# Available test profiles
AVAILABLE_PROFILES = list(PROFILE_WAIT_TIMES.keys())

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

def _sanitize_host_litserve(host: str) -> str:
    """Sanitize host URL for requests library."""
    host = host.rstrip("/")
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"http://{host}"
    return host

def _sanitize_host_triton(host: str) -> str:
    """Locust provides host via -H; Triton client accepts "host:port" or "http(s)://host:port"""
    if host.startswith("http://"):
        host = host[len("http://") :]
    elif host.startswith("https://"):
        host = host[len("https://") :]
    return host.rstrip("/")

class LitServeClient:
    def __init__(self, host: str, model_name: str):
        self.host = _sanitize_host_litserve(host)
        self.model_name = model_name
        self.base_url = f"{self.host}/predict"
        self.session = requests.Session()
        self.session.timeout = 60.0
    
    def send(self):
        request_meta = {
            "request_type": "predict",
            "name": "LitServe",
            "start_time": time.time(),
            "response_length": 0,
            "context": {
                "model_name": self.model_name,
            },
            "exception": None
        }
        start_perf_counter = time.perf_counter()
        try:
            # Prepare input: single image [3, 224, 224] -> wrap in batch [1, 3, 224, 224]
            batched = np.expand_dims(TRANSFORMED_IMG, axis=0).astype(np.float32)
            
            # Create JSON payload
            payload = {
                "input": batched.tolist()
            }
            
            # Send POST request
            response = self.session.post(
                self.base_url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            output = result.get("output", [])
            
            if output:
                # Calculate response length (approximate)
                request_meta["response_length"] = sys.getsizeof(str(output))

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
                logging.error(f"LitServe inference error: {error_type} - {error_msg}")
                self._error_logged.add(error_key)
            raise
        finally:
            request_meta["response_time"] = (time.perf_counter() - start_perf_counter) * 1000.0 # ms
            events.request.fire(**request_meta)

class TritonClient:
    def __init__(self, host: str, model_name: str, config: dict = None):
        if not TRITON_AVAILABLE:
            raise RuntimeError("tritonclient is not available. Install it to use Triton inference server.")
        self.host = _sanitize_host_triton(host)
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

def get_profile_from_env(user_instance):
    """Get profile from Locust environment or environment variable"""
    profile = None
    
    # Try to get profile from Locust environment (set via UI or --profile flag)
    if hasattr(user_instance, 'environment'):
        env = user_instance.environment
        # Check parsed_options for profile
        if hasattr(env, 'parsed_options') and hasattr(env.parsed_options, 'profile'):
            profile = env.parsed_options.profile
        # Check if profile is stored in environment
        elif hasattr(env, 'profile'):
            profile = env.profile
    
    # Fall back to environment variable (set at startup)
    if not profile:
        profile = os.environ.get("LOCUST_PROFILE", "max_load")
    
    return profile

def get_inference_server():
    """Get inference server type from environment variable"""
    server = os.environ.get("INFERENCE_SERVER", "").lower()
    if server not in ["litserve", "triton"]:
        raise RuntimeError(
            f"INFERENCE_SERVER environment variable must be set to 'litserve' or 'triton'. "
            f"Got: {server}"
        )
    return server

class InferenceUser(User):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.model_name = os.environ.get("MODEL_NAME")
        if not self.model_name:
            raise RuntimeError("MODEL_NAME environment variable is required")
        
        server = get_inference_server()
        if server == "litserve":
            self.client = LitServeClient(self.host, self.model_name)
        elif server == "triton":
            config = getattr(self, 'config', {})
            self.client = TritonClient(self.host, self.model_name, config)


class MyUser(InferenceUser):
    # Dynamic wait time based on LOCUST_PROFILE environment variable
    # Use constant_pacing for rate-based profiles to maintain consistent request rate
    # Use constant(0) for max_load to achieve maximum throughput
    def __init__(self, env):
        super().__init__(env)
        profile = get_profile_from_env(self)
        wait_time_value = PROFILE_WAIT_TIMES.get(profile, PROFILE_WAIT_TIMES["max_load"])
        
        # Store wait_time_value and last task start time for pacing
        self._wait_time_value = wait_time_value
        self._last_task_start = None
    
    def wait_time(self):
        """Dynamic wait time based on profile - implements constant_pacing logic"""
        if self._wait_time_value == 0:
            # Maximum throughput - no wait between requests
            return 0
        else:
            # Rate-based profile - implement constant_pacing logic
            # constant_pacing ensures tasks execute at consistent intervals
            # accounting for task execution time
            now = time.time()
            if self._last_task_start is None:
                self._last_task_start = now
                return 0
            
            elapsed = now - self._last_task_start
            wait_needed = max(0, self._wait_time_value - elapsed)
            self._last_task_start = now + wait_needed
            return wait_needed
    
    @task
    def send_request(self):
        self.client.send()


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize environment with available profiles for Web UI"""
    if hasattr(environment, 'web_ui') and environment.web_ui:
        environment.web_ui.template_args["all_profiles"] = AVAILABLE_PROFILES

