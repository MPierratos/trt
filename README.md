# Inference Performance Benchmarking

This project compares inference performance between Triton Inference Server and LitServe for ResNet50 models using both LibTorch and OpenVINO backends.

## Project Structure

- `inference_servers/triton/` - Triton-specific assets (model repository, profiling configs)
- `inference_servers/litserve/` - LitServe server implementations
- `src/inference_perf/` - Shared client library and model utilities
- `tests/performance/` - Locust load testing configurations

## Running Triton Server

```sh
make docker-run-triton-gpu 
```

Sample request:

With larger batches, you'll notice a reduction in latency for libtorch as it's hosted on GPU.

```sh
# Sending a batch of 1000 to the libtorch resnet model
uv run client.py --model-name resnet50_libtorch -b 1000
# Sending a batch of 1000 to the openvino resnet model
uv run client.py --model-name resnet50_openvino -b 1000
```

## Running LitServe Server

LitServe servers use their own isolated model repository and configuration files. Each model has its own `config.json` file that specifies `max_batch_size`, `accelerator`, and other settings.

### LibTorch Backend

```sh
uv run python inference_servers/litserve/server.py --model-type libtorch
```

### OpenVINO Backend

```sh
uv run python inference_servers/litserve/server.py --model-type openvino
```

### Custom Configuration

You can override settings from `config.json`:

```sh
# Override max batch size
uv run python inference_servers/litserve/server.py --model-type libtorch --max-batch-size-override 32

# Override accelerator
uv run python inference_servers/litserve/server.py --model-type libtorch --accelerator-override cuda

# Change port
uv run python inference_servers/litserve/server.py --model-type libtorch --port 8001

# Use custom models directory
uv run python inference_servers/litserve/server.py --model-type libtorch --models-dir /path/to/models
```

### Model Configuration

Each model's configuration is stored in `inference_servers/litserve/models/{libtorch,openvino}/config.json`. You can edit these files directly to change default settings:

```json
{
  "model_name": "resnet50_libtorch",
  "accelerator": "cuda",
  "max_batch_size": 1,
  "input_shape": [3, 224, 224],
  "output_shape": [1000]
}
```

## Using the Client Library

The `inference_perf` library provides shared utilities for both Triton and LitServe clients:

```python
from inference_perf import rn50_preprocess, infer_model, MODEL_CONFIGS

# Preprocess an image
img = rn50_preprocess("path/to/image.jpg")

# Use with Triton client (see client.py for full example)
```

## Model Export Utilities

Export models for use with different backends:

```python
from inference_perf.models import export_to_openvino, export_to_libtorch

export_to_openvino()
export_to_libtorch()
```


# Measuring Performance

## Perf Analyzer (similar to locust)

The perf analyzer utility is provided by NVIDIA and can be accessed using the sdk image. Note, for now, this is only going to be available for GPU as part of this repo. This will run performance analysis directly against the Triton server.

```sh
make docker-run-triton-sdk
# run these inside the container
make perf-openvino
make perf-libtorch
```


## Model Analyzer (recommends config changes via objective) 

Profile both models with 50ms latency budget and automatic config sweep. 
The configuration for the models can be adjusted in `inference_servers/triton/profiling/model_analyzer_config.yaml`.

```sh
# We run in remote mode (Triton server already running)
make model-analyzer
```

Results will be saved to `inference_servers/triton/profiling/reports/` directory with:
- Summary reports (CSV, PDF)
- Detailed metrics
- Optimized model configurations

## Simulating traffic with Locust 

Locust config files are organized by server type in `tests/performance/triton/configs/` and `tests/performance/litserve/configs/`.

### Triton Load Tests

```sh
# 30fps tests
make performance-test CONFIG=triton/configs/30fps_libtorch.conf
make performance-test CONFIG=triton/configs/30fps_openvino.conf

# Maximum load tests
make performance-test CONFIG=triton/configs/max_load_libtorch.conf
make performance-test CONFIG=triton/configs/max_load_openvino.conf
```

### LitServe Load Tests

```sh
# 30fps tests
make performance-test CONFIG=litserve/configs/30fps_libtorch.conf
make performance-test CONFIG=litserve/configs/30fps_openvino.conf

# Maximum load tests
make performance-test CONFIG=litserve/configs/max_load_libtorch.conf
make performance-test CONFIG=litserve/configs/max_load_openvino.conf
```

Locust UI is found at http://localhost:8089/. Results are organized by model name and profile in `tests/performance/results/`.

## Comparing Triton vs LitServe

Both servers use the same underlying model files (copied to their respective model repositories) with similar configuration settings, ensuring a fair comparison. Run the same load test scenarios against both servers and compare:

- Throughput (requests/second)
- Latency (p50, p95, p99)
- Resource utilization
- Error rates

Results are saved separately for each configuration, making it easy to compare performance across different inference engines.