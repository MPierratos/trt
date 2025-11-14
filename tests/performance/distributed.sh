#!/bin/bash
# replace with your endpoitn name in format http(s)://host:port (Triton HTTP)
export ENDPOINT_NAME=$1
# provide the Triton model name as the second arg (i.e. resnet50_libtorch, resnet50_openvino)
export MODEL_NAME=$2

export USERS=1 # number of users to spawn
export SPAWN_RATE=0.5 # conrtols the ramp up of users (number spawned per second)
export WORKERS=1 # number of workers that send user requests (i.e. if 5 worksers, 30 Users - each worker will send 30 in parallel)
export RUN_TIME=1m
export LOCUST_UI=true

# Locust script for Triton
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export SCRIPT="${SCRIPT_DIR}/load_test_triton.py"

# Organize results by model name
RESULTS_DIR="${SCRIPT_DIR}/${MODEL_NAME}"
mkdir -p "${RESULTS_DIR}"

# Run Locust with custom settings
if $LOCUST_UI; then
    (cd "$RESULTS_DIR" && locust -f "$SCRIPT" --host "$ENDPOINT_NAME" --master --expect-workers "$WORKERS" -u $USERS -r $SPAWN_RATE -t $RUN_TIME --csv results) &
else
    (cd "$RESULTS_DIR" && locust -f "$SCRIPT" --host "$ENDPOINT_NAME" --master --expect-workers "$WORKERS" -u $USERS -r $SPAWN_RATE -t $RUN_TIME --csv results --headless) &
fi

for c in $(seq 1 $WORKERS); do
    (cd "$RESULTS_DIR" && locust -f "$SCRIPT" -H $ENDPOINT_NAME --worker --master-host=localhost) &
done
