# Load Testing with Locust

This directory contains Locust-based load testing tools for comparing Triton Inference Server and LitServe.

## Quick Start

### Basic Usage

Run from the `tests/performance/` directory:

```bash
cd tests/performance

# Triton Tests
# Run 30fps test with LibTorch model
./distributed.sh triton/configs/30fps_libtorch.conf

# Run 30fps test with OpenVINO model
./distributed.sh triton/configs/30fps_openvino.conf

# Run maximum load test with LibTorch
./distributed.sh triton/configs/max_load_libtorch.conf

# Run maximum load test with OpenVINO
./distributed.sh triton/configs/max_load_openvino.conf

# LitServe Tests
# Run 30fps test with LibTorch model
./distributed.sh litserve/configs/30fps_litserve_libtorch.conf

# Run 30fps test with OpenVINO model
./distributed.sh litserve/configs/30fps_litserve_openvino.conf

# Run maximum load test with LibTorch
./distributed.sh litserve/configs/max_load_litserve_libtorch.conf

# Run maximum load test with OpenVINO
./distributed.sh litserve/configs/max_load_litserve_openvino.conf
```

## Configuration

The test setup uses configuration files for all settings. Pre-configured test scenarios are organized by server type:

### Triton Configurations (`triton/configs/`)
- `triton/configs/30fps_libtorch.conf` - 30fps video processing with LibTorch model
- `triton/configs/30fps_openvino.conf` - 30fps video processing with OpenVINO model
- `triton/configs/max_load_libtorch.conf` - Maximum load test with LibTorch model
- `triton/configs/max_load_openvino.conf` - Maximum load test with OpenVINO model

### LitServe Configurations (`litserve/configs/`)
- `litserve/configs/30fps_litserve_libtorch.conf` - 30fps video processing with LitServe LibTorch model
- `litserve/configs/30fps_litserve_openvino.conf` - 30fps video processing with LitServe OpenVINO model
- `litserve/configs/max_load_litserve_libtorch.conf` - Maximum load test with LitServe LibTorch model
- `litserve/configs/max_load_litserve_openvino.conf` - Maximum load test with LitServe OpenVINO model

Each configuration file specifies which `locustfile` to use (`triton/triton_locustfile.py` or `litserve/litserve_locustfile.py`), ensuring the correct client implementation is used for each server type.


See [Locust Configuration Documentation](https://docs.locust.io/en/stable/configuration.html) for all available options.


## Web UI

The Locust Web UI is available at `http://localhost:8089` after starting the test.

## Stopping Tests

To stop all running Locust processes:

```bash
pkill -f 'locust.*locustfile'
```
