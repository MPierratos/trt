docker-pull:
	docker pull nvcr.io/nvidia/tritonserver:25.10-py3

docker-pull-sdk:
	docker pull nvcr.io/nvidia/tritonserver:25.10-py3-sdk

docker-run-triton:
	docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
		-v ${PWD}/model_repository:/models \
		nvcr.io/nvidia/tritonserver:25.10-py3 \
		tritonserver --model-repository=/models \
		--model-control-mode=explicit \
		--load-model=*

docker-run-triton-sdk:
	docker run --gpus all --rm -it --net=host -v ${PWD}:/workspace/ nvcr.io/nvidia/tritonserver:25.10-py3-sdk bash

perf-libtorch:
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

perf-openvino:
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

profile:
	mkdir -p profiling/reports
	model-analyzer profile \
		--model-repository ${PWD}/model_repository \
		--output-model-repository-path ${PWD}/profiling/reports/output_model_repo \
		--export-path ${PWD}/profiling/reports \
		--config-file ${PWD}/profiling/model_analyzer_config.yaml \
		--override-output-model-repository \
		--latency-budget 50
