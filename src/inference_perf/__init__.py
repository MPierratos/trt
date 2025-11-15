"""Inference performance testing library."""

from inference_perf.__module import MODULE_NAME, PROJECT_PATH, MODULE_PATH
from inference_perf import models
from inference_perf import client

__all__ = ["PROJECT_PATH",
            "MODULE_NAME",
            "MODULE_PATH",
            "models", 
            "client"]
