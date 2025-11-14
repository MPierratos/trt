# Running Triton Server

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


# Measuring Performance

## Perf Analyzer (similar to locust)

The perf analyzer utility is provided by NVIDIA and can be accessed using the sdk image. Note, for now, this is only going to be available for GPU as part of this repo. This wil

```sh
make docker-run-triton-sdk
# run these inside the container
make perf-openvino
make perf-libtorch
```


## Model Analyzer (recommends config changes via objective) 

Profile both models with 50ms latency budget and automatic config sweep. 
The configuration for the models can be adjusted in `profiling/model_analyzer_config.yaml`.

```sh
# We run in remote mode (Triton server already running)
make model-analyzer
```

Results will be saved to `reports/` directory with:
- Summary reports (CSV, PDF)
- Detailed metrics
- Optimized model configurations

## Simulating traffic with Locust 

Locust config files are found in `tests/performance/configs`.
```sh
make performance-test CONFIG=30fps_libtorch.conf
```

Locust UI is found at http://localhost:8089/.