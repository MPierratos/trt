#!/bin/bash
# Usage: ./distributed.sh <config_file> <model_name> [server]
# Example: ./distributed.sh configs/low.conf resnet50_libtorch litserve

# Script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Validate arguments
if [ -z "$1" ]; then
    echo "Error: Missing configuration file"
    echo ""
    echo "Usage: $0 <config_file> <model_name> [server]"
    echo ""
    echo "Available configs:"
    ls -1 "$SCRIPT_DIR/configs/"*.conf 2>/dev/null | sed 's|.*/|    - |'
    echo ""
    echo "Example:"
    echo "  $0 configs/low.conf resnet50_libtorch litserve"
    echo "  $0 configs/high.conf resnet50_openvino triton"
    exit 1
fi

if [ -z "$2" ]; then
    echo "Error: Missing model name"
    echo ""
    echo "Usage: $0 <config_file> <model_name> [server]"
    exit 1
fi

CONFIG_FILE="$1"
export MODEL_NAME="$2"

# Make path absolute if relative
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="${SCRIPT_DIR}/${CONFIG_FILE}"
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Detect inference server from argument or config path
if [ -n "$3" ]; then
    INFERENCE_SERVER="$3"
elif [[ "$CONFIG_FILE" == *"/litserve/"* ]]; then
    INFERENCE_SERVER="litserve"
elif [[ "$CONFIG_FILE" == *"/triton/"* ]]; then
    INFERENCE_SERVER="triton"
else
    echo "Error: Could not detect inference server"
    echo "Please specify server as third argument: litserve or triton"
    exit 1
fi

# Validate server
if [ "$INFERENCE_SERVER" != "litserve" ] && [ "$INFERENCE_SERVER" != "triton" ]; then
    echo "Error: Invalid server '$INFERENCE_SERVER'"
    echo "Server must be 'litserve' or 'triton'"
    exit 1
fi

# Export inference server for locustfile
export INFERENCE_SERVER="$INFERENCE_SERVER"

# Parse standard Locust config values
parse_locust_config() {
    local var_name=$1
    local value=$(grep "^${var_name}[[:space:]]*=" "$CONFIG_FILE" | sed 's/^[^=]*=[[:space:]]*//' | tr -d '\r')
    echo "$value"
}

# Get workers from config for display
WORKERS=$(parse_locust_config "expect-workers")
WORKERS=${WORKERS:-1}

# Get host from config for display
HOST=$(grep "^host[[:space:]]*=" "$CONFIG_FILE" | sed 's/^[^=]*=[[:space:]]*//' | tr -d '\r')

# Organize results by inference server
RESULTS_DIR="${SCRIPT_DIR}/results/${INFERENCE_SERVER}"
mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo "Locust Distributed Load Test"
echo "=========================================="
echo "Config:     $CONFIG_FILE"
echo "Host:       $HOST"
echo "Model:      $MODEL_NAME"
echo "Server:     $INFERENCE_SERVER"
echo "Workers:    $WORKERS"
echo "Results:    $RESULTS_DIR"
echo "=========================================="

# Use unified locustfile
LOCUSTFILE="${SCRIPT_DIR}/locustfile.py"

# Array to track all Locust process PIDs
LOCUST_PIDS=()

# Cleanup function to kill all Locust processes
cleanup() {
    echo ""
    echo "Cleaning up Locust processes..."
    if [ ${#LOCUST_PIDS[@]} -gt 0 ]; then
        for pid in "${LOCUST_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Killing process $pid"
                kill "$pid" 2>/dev/null
            fi
        done
        # Wait a moment for processes to terminate
        sleep 1
        # Force kill if still running
        for pid in "${LOCUST_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Force killing process $pid"
                kill -9 "$pid" 2>/dev/null
            fi
        done
    fi
    # Also kill any remaining locust processes matching our pattern
    pkill -f "locust.*locustfile" 2>/dev/null
    echo "Cleanup complete."
    exit 0
}

# Set up trap handlers for cleanup on exit signals
trap cleanup SIGINT SIGTERM EXIT

# Run Locust Master
echo "Starting Locust Master..."
(cd "$RESULTS_DIR" && locust \
    --config "$CONFIG_FILE" \
    --locustfile "$LOCUSTFILE" \
    --master) &

MASTER_PID=$!
LOCUST_PIDS+=($MASTER_PID)
sleep 2  # Give master time to start

# Run Locust Workers
echo "Starting $WORKERS worker(s)..."
for c in $(seq 1 $WORKERS); do
    (cd "$RESULTS_DIR" && locust \
        --config "$CONFIG_FILE" \
        --locustfile "$LOCUSTFILE" \
        --worker \
        --master-host localhost) &
    WORKER_PID=$!
    LOCUST_PIDS+=($WORKER_PID)
    echo "  Worker $c started (PID: $WORKER_PID)"
done

echo "=========================================="
echo "All processes started!"
echo "Web UI available at: http://localhost:8089"
echo "Master PID: $MASTER_PID"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop all Locust processes"
echo ""

wait
