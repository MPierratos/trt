# Running Triton Server
```sh
# For Model Analyzer profiling (explicit mode)
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 -v ${PWD}/model_repository:/models nvcr.io/nvidia/tritonserver:25.10-py3 tritonserver --model-repository=/models --model-control-mode=explicit --load-model=*

# Or use make command
make docker-run-triton
```

# For the perf analyzer
`docker run --gpus all --rm -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:25.10-py3-sdk bash`

# Query
# resnet50_libtorch (GPU, 2 instances)
```sh
perf_analyzer -m resnet50_libtorch \
  --service-kind=triton \
  -b 2 \
  -u localhost:8001 \
  -i grpc \
  --shape input__0:3,224,224 \
  --concurrency-range 2:16:2 \
  --percentile=95 \
  --measurement-interval 5000 \
  --measurement-mode count_windows \
  --measurement-request-count 200 \
  --async
```

# resnet50_openvino (CPU, 2 instances)
```sh
perf_analyzer -m resnet50_openvino \
  --service-kind=triton \
  -b 2 \
  -u localhost:8001 \
  -i grpc \
  --shape x:3,224,224 \
  --concurrency-range 2:16:2 \
  --percentile=95 \
  --measurement-interval 5000 \
  --measurement-mode count_windows \
  --measurement-request-count 200 \
  --async
```

# Model Analyzer - Profile with latency budget
Profile both models with 50ms latency budget and automatic config sweep:

```sh
# Remote mode (Triton server already running)
make profile
```

Results will be saved to `reports/` directory with:
- Summary reports (CSV, PDF)
- Detailed metrics
- Optimized model configurations