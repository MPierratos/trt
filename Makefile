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

performance-test: ## Run distributed performance test (requires CONFIG)
	@if [ -z "$(CONFIG)" ]; then \
		echo "Please specify CONFIG, e.g. make performance-test CONFIG=triton/configs/30fps_libtorch.conf"; \
		echo ""; \
		echo "Available configs:"; \
		echo "  Triton:"; \
		ls -1 tests/performance/triton/configs/*.conf 2>/dev/null | sed 's|.*/||' | sed 's/^/    - /'; \
		echo "  LitServe:"; \
		ls -1 tests/performance/litserve/configs/*.conf 2>/dev/null | sed 's|.*/||' | sed 's/^/    - /'; \
		exit 1; \
	fi
	./tests/performance/distributed.sh tests/performance/$(CONFIG) 