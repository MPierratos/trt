# Triton Load Testing with Locust

This directory contains Locust-based load testing tools for Triton Inference Server.

## Quick Start

### Basic Usage

Run from the `tests/performance/` directory:

```bash
cd tests/performance

# Run 30fps test with LibTorch model
./distributed.sh configs/30fps_libtorch.conf

# Run 30fps test with OpenVINO model
./distributed.sh configs/30fps_openvino.conf

# Run maximum load test with OpenVINO
./distributed.sh configs/max_load_openvino.conf

# Run maximum load test with Libtorch
./distributed.sh configs/max_load_libtorch.conf
```

## Configuration

The test setup uses configuration files for all settings. Pre-configured test scenarios are in the `configs/` directory:

- `configs/30fps_libtorch.conf` - 30fps video processing with LibTorch model
- `configs/30fps_openvino.conf` - 30fps video processing with OpenVINO model
- `configs/max_load_openvino.conf` - Maximum load test with OpenVINO model
- `configs/max_load_libtorch.conf` - Maximum load test with OpenVINO model


See [Locust Configuration Documentation](https://docs.locust.io/en/stable/configuration.html) for all available options.


## Web UI

The Locust Web UI is available at `http://localhost:8089` after starting the test.

## Stopping Tests

To stop all running Locust processes:

```bash
pkill -f 'locust.*locustfile'
```
