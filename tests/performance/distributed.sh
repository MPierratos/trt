#!/bin/bash
# Usage: ./distributed.sh <config_file>
# Example: ./distributed.sh configs/30fps_libtorch.conf

# Script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Validate config file argument
if [ -z "$1" ]; then
    echo "Error: Missing configuration file"
    echo ""
    echo "Usage: $0 <config_file>"
    echo ""
    echo "Available configs:"
    echo "  Triton:"
    ls -1 "$SCRIPT_DIR/triton/configs/"*.conf 2>/dev/null | sed 's|.*/|    - |'
    echo "  LitServe:"
    ls -1 "$SCRIPT_DIR/litserve/configs/"*.conf 2>/dev/null | sed 's|.*/|    - |'
    echo ""
    echo "Example:"
    echo "  $0 triton/configs/30fps_libtorch.conf"
    echo "  $0 litserve/configs/30fps_litserve_libtorch.conf"
    exit 1
fi

CONFIG_FILE="$1"

# Make path absolute if relative
if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="${SCRIPT_DIR}/${CONFIG_FILE}"
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Parse custom variables from config file comments
# Look for lines like: # @MODEL_NAME: resnet50_libtorch
parse_config_var() {
    local var_name=$1
    local value=$(grep "^#[[:space:]]*@${var_name}:" "$CONFIG_FILE" | sed 's/^#[[:space:]]*@[^:]*:[[:space:]]*//' | tr -d '\r')
    echo "$value"
}

# Parse standard Locust config values
parse_locust_config() {
    local var_name=$1
    local value=$(grep "^${var_name}[[:space:]]*=" "$CONFIG_FILE" | sed 's/^[^=]*=[[:space:]]*//' | tr -d '\r')
    echo "$value"
}

export MODEL_NAME=$(parse_config_var "MODEL_NAME")
export LOCUST_PROFILE=$(parse_config_var "LOCUST_PROFILE")

# Set defaults if not found in config
export LOCUST_PROFILE=${LOCUST_PROFILE:-max_load}

# Get workers from config for display
WORKERS=$(parse_locust_config "expect-workers")
WORKERS=${WORKERS:-1}

# Validate required variables
if [ -z "$MODEL_NAME" ]; then
    echo "Error: MODEL_NAME not set in config file"
    echo "Add to config: MODEL_NAME = resnet50_libtorch"
    exit 1
fi

# Get host from config for display
HOST=$(grep "^host[[:space:]]*=" "$CONFIG_FILE" | sed 's/^[^=]*=[[:space:]]*//' | tr -d '\r')

# Organize results by model name and profile
RESULTS_DIR="${SCRIPT_DIR}/results/${MODEL_NAME}_${LOCUST_PROFILE}"
mkdir -p "${RESULTS_DIR}"

echo "=========================================="
echo "Locust Distributed Load Test"
echo "=========================================="
echo "Config:     $CONFIG_FILE"
echo "Host:       $HOST"
echo "Model:      $MODEL_NAME"
echo "Profile:    $LOCUST_PROFILE"
echo "Workers:    $WORKERS"
echo "Results:    $RESULTS_DIR"
echo "=========================================="

# Get locustfile from config, default to triton_locustfile.py for backward compatibility
LOCUSTFILE_NAME=$(parse_locust_config "locustfile")
LOCUSTFILE_NAME=${LOCUSTFILE_NAME:-triton_locustfile.py}
LOCUSTFILE="${SCRIPT_DIR}/${LOCUSTFILE_NAME}"

# Run Locust Master
echo "Starting Locust Master..."
(cd "$RESULTS_DIR" && locust \
    --config "$CONFIG_FILE" \
    --locustfile "$LOCUSTFILE" \
    --master \
    --profile "$LOCUST_PROFILE") &

MASTER_PID=$!
sleep 2  # Give master time to start

# Run Locust Workers
echo "Starting $WORKERS worker(s)..."
for c in $(seq 1 $WORKERS); do
    (cd "$RESULTS_DIR" && locust \
        --config "$CONFIG_FILE" \
        --locustfile "$LOCUSTFILE" \
        --worker \
        --master-host localhost) &
    echo "  Worker $c started (PID: $!)"
done

echo "=========================================="
echo "All processes started!"
echo "Web UI available at: http://localhost:8089"
echo "Master PID: $MASTER_PID"
echo "=========================================="
echo ""
echo "To stop all Locust processes:"
echo "  pkill -f 'locust.*locustfile'"
echo ""

wait
