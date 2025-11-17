# Load Testing with Locust

This directory contains Locust-based load testing tools for comparing Triton Inference Server and LitServe.

## Quick Start

### Basic Usage

Run from the project root:

```bash
# Low load test with LitServe
make performance-test CONFIG=configs/low.conf MODEL_NAME=resnet50_libtorch INFERENCE_SERVER=litserve

# High load test with Triton
make performance-test CONFIG=configs/high.conf MODEL_NAME=resnet50_openvino INFERENCE_SERVER=triton
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
- `configs/low.conf` - Low load test using `LowUser` class (1 user, constant wait time 1/30s, 1m runtime, 1 worker)
- `configs/high.conf` - High load test using `HighUser` class (50 users, constant wait time 0s, 2m runtime, 5 workers)

The unified `locustfile.py` automatically selects the correct client implementation (LitServe or Triton) based on the `INFERENCE_SERVER` environment variable set by the script.

### User Classes
- **LowUser**: Constant wait time of 1/30.0 seconds (30 requests per second per user)
- **HighUser**: Constant wait time of 0 seconds (maximum throughput, no wait between requests)

Each config file specifies the user class via the `class-name` field, and the class-picker UI is disabled to ensure consistent test execution.

See [Locust Configuration Documentation](https://docs.locust.io/en/stable/configuration.html) for all available options.

## Request Format

### Triton Requests
Triton requests use the **binary tensor format** for optimal performance:
- JSON header containing tensor metadata (shape, datatype, etc.)
- Binary tensor data appended after the JSON header
- Header `Inference-Header-Content-Length` indicates JSON header size
- This format matches the Triton HTTP client library's binary mode, providing similar performance (~20ms latency)

### LitServe Requests
LitServe requests use standard JSON format:
- JSON payload with `{"input": <tensor_data>}`

Both implementations use `FastHttpUser` (FastHttpSession) with identical connection pooling, timeout settings, and automatic keep-alive to ensure fair comparison. FastHttpUser provides 4-6x higher throughput compared to the standard HttpUser.

## Debugging

The script logs errors and warnings automatically. For more detailed debugging:

- Check Locust logs in the terminal output
- Review the Locust Web UI at `http://localhost:8089` for request statistics
- Verify binary format is being used by checking that Triton requests include the `Inference-Header-Content-Length` header

## Apples-to-Apples Comparison

To ensure fair comparison between Triton and LitServe:

1. **Use the same model**: Compare `resnet50_openvino` on Triton vs LitServe, or `resnet50_libtorch` on both
2. **Use the same config**: Run both servers with the same config file (e.g., `low.conf`)
3. **Same network conditions**: Run tests on the same machine/network
4. **Same input data**: Both use the same preprocessed image (`data/img1.jpg`)
5. **Same HTTP client**: Both use `FastHttpUser` (FastHttpSession) with identical timeout settings (60s) and automatic connection pooling

### Expected Performance
- **Triton with binary format**: ~20ms response time (matching Triton client library performance)
- **LitServe**: Performance depends on backend implementation

If Triton requests show unexpectedly high latency, verify:
- Binary format is being used (default; can be overridden with `TRITON_USE_JSON=true`)
- `Inference-Header-Content-Length` header is set correctly
- Tensor data is sent as binary bytes, not JSON
- Check Locust Web UI statistics for detailed request/response metrics


## Web UI

The Locust Web UI is available at `http://localhost:8089` after starting the test.

## Stopping Tests

To stop all running Locust processes:

```bash
pkill -f 'locust.*locustfile'
```
