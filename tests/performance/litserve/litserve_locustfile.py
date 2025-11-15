import os
import sys
import time
import logging
import numpy as np
import pathlib
import requests
from PIL import Image
from torchvision import transforms
from locust import User, task, events

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATA_DIR = pathlib.Path(__file__).parent.parent.parent / "data"
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

def _sanitize_host(host: str) -> str:
    """Sanitize host URL for requests library."""
    host = host.rstrip("/")
    if not host.startswith("http://") and not host.startswith("https://"):
        host = f"http://{host}"
    return host

class LitServeClient:
    def __init__(self, host: str, model_name: str):
        self.host = _sanitize_host(host)
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

def profile_wait_time(user_instance):
    """Wait time function that reads profile from Locust environment or environment variable"""
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
    
    return PROFILE_WAIT_TIMES.get(profile, PROFILE_WAIT_TIMES["max_load"])


class LitServeUser(User):
    abstract = True

    def __init__(self, env):
        super().__init__(env)
        self.model_name = os.environ.get("MODEL_NAME")
        if not self.model_name:
            raise RuntimeError("MODEL_NAME environment variable is required")
        
        self.client = LitServeClient(self.host, self.model_name)


class MyUser(LitServeUser):
    # Dynamic wait time based on LOCUST_PROFILE environment variable
    wait_time = profile_wait_time
    
    @task
    def send_request(self):
        self.client.send()


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    """Initialize environment with available profiles for Web UI"""
    if hasattr(environment, 'web_ui') and environment.web_ui:
        environment.web_ui.template_args["all_profiles"] = AVAILABLE_PROFILES

