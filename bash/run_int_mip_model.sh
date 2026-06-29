#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS="$PROJECT_ROOT/minizinc"
DATA="$MODELS/data"
PYTHON_DIR="$PROJECT_ROOT/python"
PYTHON_SCRIPT="$PYTHON_DIR/run_and_log_minizinc_model.py"
TIMEOUT=300
PYTHON_CMD="python3"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at: $PYTHON_SCRIPT"
    exit 1
fi

if [ ! -d "$MODELS" ]; then
    echo "Error: models directory not found at: $MODELS"
    exit 1
fi

run_model() {
    local model=$1
    local instance=$2
    local solver=$3
    
    model_file="int_mip_model.mzn"
    data_file="$DATA/${instance}.dzn"
    
    full_model_name="${model}"
    
    cd "$MODELS"
    $PYTHON_CMD "$PYTHON_SCRIPT" "$model_file" "$data_file" \
        --solver "$solver" \
        --timeout "$TIMEOUT" \
        --model-name "${full_model_name}" \
        --instance-name "$instance"
}

process_workpiece() {
    local instance=$1
    
    # Integer MIP Model
    run_model "int_mip_model" "$instance" "gurobi"
}

process_workpiece "spiral_stair_step"
process_workpiece "simple_stair_step"
process_workpiece "dashboard"
process_workpiece "speaker"
process_workpiece "coffee_table"

exit 0
