# Copyright 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import tritonclient.http as httpclient
from PIL import Image
from torchvision import transforms
from tritonclient.utils import triton_to_np_dtype
import pathlib
import time

DATA_DIR = pathlib.Path(__file__).parent / "data"

# Shared preprocessing function
def rn50_preprocess(img_path=DATA_DIR / "img1.jpg"):
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


# Model configurations
MODEL_CONFIGS = {
    "resnet50_libtorch": {
        "input_name": "input__0",
        "output_name": "output__0",
    },
    "resnet50_openvino": {
        "input_name": "x",
        "output_name": "x.45",
    },
}


def infer_model(client, model_name, transformed_img, config):
    """Make inference request to a single model."""
    # With max_batch_size > 0, Triton expects inputs without batch dimension
    # The input shape should be [3, 224, 224] and Triton will add batch dimension
    input_img = transformed_img  # Shape: [3, 224, 224]

    inputs = httpclient.InferInput(
        config["input_name"], input_img.shape, datatype="FP32"
    )
    inputs.set_data_from_numpy(input_img, binary_data=True)

    outputs = httpclient.InferRequestedOutput(
        config["output_name"], binary_data=True, class_count=1000
    )

    # Query the server
    results = client.infer(
        model_name=model_name, inputs=[inputs], outputs=[outputs]
    )
    inference_output = results.as_numpy(config["output_name"])
    
    # Handle output shape - might be [1000] or [1, 1000] depending on Triton's handling
    # Squeeze batch dimension if present and batch size is 1
    if len(inference_output.shape) == 2 and inference_output.shape[0] == 1:
        inference_output = inference_output.squeeze(0)

    return inference_output


def infer_model_batch(client, model_name, transformed_imgs, config):
    """Make batched inference request to a model.
    
    Args:
        client: Triton client instance
        model_name: Name of the model
        transformed_imgs: List or numpy array of preprocessed images, shape [batch_size, 3, 224, 224]
        config: Model configuration dict
    """
    # Stack images if they're a list
    if isinstance(transformed_imgs, list):
        batch_input = np.stack(transformed_imgs, axis=0)
    else:
        batch_input = transformed_imgs
    
    # With max_batch_size > 0, Triton expects inputs with batch dimension
    # Shape should be [batch_size, 3, 224, 224]
    inputs = httpclient.InferInput(
        config["input_name"], batch_input.shape, datatype="FP32"
    )
    inputs.set_data_from_numpy(batch_input, binary_data=True)

    outputs = httpclient.InferRequestedOutput(
        config["output_name"], binary_data=True, class_count=1000
    )

    # Query the server
    results = client.infer(
        model_name=model_name, inputs=[inputs], outputs=[outputs]
    )
    inference_output = results.as_numpy(config["output_name"])
    
    # Output shape will be [batch_size, 1000]
    return inference_output


# Preprocess image once
transformed_img = rn50_preprocess()

# Setting up client
client = httpclient.InferenceServerClient(url="localhost:8000")

# Make single-image requests to all models
print("Running single-image inference on all models...\n")
for model_name, config in MODEL_CONFIGS.items():
    print(f"Model: {model_name}")
    try:
        start_time = time.perf_counter()
        output = infer_model(client, model_name, transformed_img, config)
        elapsed_time = time.perf_counter() - start_time
        print(f"Output shape: {output.shape}")
        print(f"Output (first 5): {output[:5]}")
        print(f"Time: {elapsed_time:.4f} seconds")
        print()
    except Exception as e:
        print(f"Error: {e}\n")

# Test batch inference
print("Running batch inference (batch size 2)...\n")
batch_size = 2
batch_imgs = [transformed_img] * batch_size
for model_name, config in MODEL_CONFIGS.items():
    print(f"Model: {model_name}")
    try:
        start_time = time.perf_counter()
        output = infer_model_batch(client, model_name, batch_imgs, config)
        elapsed_time = time.perf_counter() - start_time
        print(f"Output shape: {output.shape}")
        print(f"Output (first 5 of first image): {output[0][:5]}")
        print(f"Time: {elapsed_time:.4f} seconds")
        print()
    except Exception as e:
        print(f"Error: {e}\n")

