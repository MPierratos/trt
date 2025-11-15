# Load Testing with Locust

This directory contains Locust-based load testing tools for comparing Triton Inference Server and LitServe.

## Quick Start

### Basic Usage

Run from the project root:

```bash
# Low load test with LitServe
make performance-test CONFIG=configs/low.conf MODEL_NAME=resnet50_libtorch SERVER=litserve

# High load test with Triton
make performance-test CONFIG=configs/high.conf MODEL_NAME=resnet50_openvino SERVER=triton
```

Or run directly from the `tests/performance/` directory:

```bash
cd tests/performance

# Low load test with LitServe
./distributed.sh configs/low.conf resnet50_libtorch litserve

# High load test with Triton
./distributed.sh configs/high.conf resnet50_openvino triton
```

## Configuration

The test setup uses shared configuration files for all settings. Pre-configured test scenarios:

### Shared Configurations (`configs/`)
- `configs/low.conf` - Low load test (1 user, 0.5 spawn-rate, 1m runtime, 1 worker)
- `configs/high.conf` - High load test (50 users, 5.0 spawn-rate, 2m runtime, 5 workers)

The unified `locustfile.py` automatically selects the correct client implementation (LitServe or Triton) based on the `SERVER` parameter passed to the script.


See [Locust Configuration Documentation](https://docs.locust.io/en/stable/configuration.html) for all available options.


## Web UI

The Locust Web UI is available at `http://localhost:8089` after starting the test.

## Stopping Tests

To stop all running Locust processes:

```bash
pkill -f 'locust.*locustfile'
```
