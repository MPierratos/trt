.PHONY: help
help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-25s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

docker-pull: ## Pull Triton Inference Server Docker image
	docker pull nvcr.io/nvidia/tritonserver:25.10-py3 --platform linux/amd64

docker-pull-sdk: ## Pull Triton Inference Server SDK Docker image
	docker pull nvcr.io/nvidia/tritonserver:25.10-py3-sdk --platform linux/amd64

docker-run-triton-gpu: ## Run Triton Inference Server with GPU support
	docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
		--platform linux/amd64 \
		-v ${PWD}/inference_servers/triton/model_repository:/models \
		nvcr.io/nvidia/tritonserver:25.10-py3 \
		tritonserver --model-repository=/models \
		--model-control-mode=explicit \
		--load-model=*

docker-run-triton-cpu: ## Run Triton Inference Server with CPU only
	docker run --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
		--platform linux/amd64 \
		-v ${PWD}/inference_servers/triton/model_repository:/models \
		nvcr.io/nvidia/tritonserver:25.10-py3 \
		tritonserver --model-repository=/models \
		--model-control-mode=explicit \
		--load-model=*

docker-run-triton-sdk: ## Run Triton Inference Server SDK container interactively
	docker run --gpus all --rm -it --net=host \
		--platform linux/amd64 \
		-v ${PWD}:/workspace/ \
		nvcr.io/nvidia/tritonserver:25.10-py3-sdk \
		bash

run-litserve: ## Run LitServe server (MODEL=libtorch|openvino, default: openvino).
	@MODEL=$${MODEL:-openvino}; \
	if [ "$$MODEL" != "libtorch" ] && [ "$$MODEL" != "openvino" ]; then \
		echo "Error: MODEL must be 'libtorch' or 'openvino'"; \
		exit 1; \
	fi; \
	uv run python inference_servers/litserve/server.py --model-type $$MODEL

perf-libtorch: ## Run performance analyzer for resnet50_libtorch model
	perf_analyzer -m resnet50_libtorch \
		--service-kind=triton \
		-u localhost:8001 \
		-i grpc \
		--shape input__0:2,3,224,224 \
		--concurrency-range 2:16:2 \
		--percentile=95 \
		--measurement-interval 5000 \
		--measurement-mode count_windows \
		--measurement-request-count 200 \
		--async

perf-openvino: ## Run performance analyzer for resnet50_openvino model
	perf_analyzer -m resnet50_openvino \
		--service-kind=triton \
		-u localhost:8001 \
		-i grpc \
		--shape x:2,3,224,224 \
		--concurrency-range 2:16:2 \
		--percentile=95 \
		--measurement-interval 5000 \
		--measurement-mode count_windows \
		--measurement-request-count 200 \
		--async

model-analyzer: ## Run model analyzer to profile models
	mkdir -p inference_servers/triton/profiling/reports
	model-analyzer profile \
		--model-repository ${PWD}/inference_servers/triton/model_repository \
		--output-model-repository-path ${PWD}/inference_servers/triton/profiling/reports/output_model_repo \
		--export-path ${PWD}/inference_servers/triton/profiling/reports \
		--config-file ${PWD}/inference_servers/triton/profiling/model_analyzer_config.yaml \
		--override-output-model-repository

performance-test: ## Run distributed performance test (requires CONFIG, MODEL_NAME, and INFERENCE_SERVER)
	pkill -f 'locust.*locustfile' 2>/dev/null
	sleep 1
	pkill -9 -f 'locust.*locustfile' 2>/dev/null
	./tests/performance/distributed.sh $(CONFIG) $(MODEL_NAME) $(INFERENCE_SERVER)
