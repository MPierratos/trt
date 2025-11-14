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

import argparse
import numpy as np
import pathlib
import time
from PIL import Image
from torchvision import transforms
import tritonclient.http as httpclient

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


def infer_model(client, model_name, transformed_imgs, config):
    """Make inference request to a model.
    
    With max_batch_size: 0, all inputs must have explicit batch dimension.
    
    Args:
        client: Triton client instance
        model_name: Name of the model
        transformed_imgs: Single image [3, 224, 224] or list/array of images
        config: Model configuration dict
        
    Returns:
        numpy array of shape [batch_size, 1000] with predictions
    """
    # Convert input to batched format [batch_size, 3, 224, 224]
    if isinstance(transformed_imgs, list):
        batch_input = np.stack(transformed_imgs, axis=0)
    elif len(transformed_imgs.shape) == 3:
        # Single image [3, 224, 224] -> add batch dimension [1, 3, 224, 224]
        batch_input = np.expand_dims(transformed_imgs, axis=0)
    else:
        # Already batched [batch_size, 3, 224, 224]
        batch_input = transformed_imgs
    
    # With max_batch_size: 0, Triton expects explicit batch dimension
    # Shape: [batch_size, 3, 224, 224]
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
    
    # Output shape: [batch_size, 1000]
    return inference_output


def main():
    parser = argparse.ArgumentParser(
        description="Triton Inference Client for ResNet50 models"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        choices=list(MODEL_CONFIGS.keys()),
        default="resnet50_libtorch",
        help="Model name to use for inference (default: resnet50_libtorch)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="localhost:8000",
        help="Triton server URL (default: localhost:8000)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (default: data/img1.jpg)",
    )
    
    args = parser.parse_args()
    
    # Preprocess image
    if args.image:
        transformed_img = rn50_preprocess(args.image)
    else:
        transformed_img = rn50_preprocess()
    
    # Create batch of images
    if args.batch_size == 1:
        batch_imgs = transformed_img
    else:
        batch_imgs = [transformed_img] * args.batch_size
    
    # Setting up client
    client = httpclient.InferenceServerClient(url=args.url)
    config = MODEL_CONFIGS[args.model_name]
    
    # Run inference
    print(f"Running inference on {args.model_name} with batch size {args.batch_size}...\n")
    try:
        start_time = time.perf_counter()
        output = infer_model(client, args.model_name, batch_imgs, config)
        elapsed_time = time.perf_counter() - start_time
        
        print(f"Model: {args.model_name}")
        print(f"Batch size: {args.batch_size}")
        print(f"Output shape: {output.shape}")
        print(f"Output (first 5 of first image): {output[0][:5]}")
        print(f"Total time: {elapsed_time:.4f} seconds")
        if args.batch_size > 1:
            print(f"Average time per image: {elapsed_time/args.batch_size:.4f} seconds")
        print()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

