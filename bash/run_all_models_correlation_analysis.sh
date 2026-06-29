    #!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS="$PROJECT_ROOT/minizinc"
DATA="$MODELS/data"
PYTHON_DIR="$PROJECT_ROOT/python"
PYTHON_SCRIPT="$PYTHON_DIR/intermediate_solution_analysis.py"
TIMEOUT=300

if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

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

    if [[ "$model" == "LNS_cp"* ]]; then
        model_file="LNS_cp_model.mzn"
    elif [[ "$model" == "mip"* ]]; then
        model_file="mip_model.mzn"
    #elif [[ "$model" == "int_mip"* ]]; then
    #    model_file="int_mip_model.mzn"
    else
        model_file="cp_model.mzn"
    fi

    data_file="$DATA/${instance}.dzn"

    cd "$MODELS" || exit 1
    "$PYTHON_CMD" "$PYTHON_SCRIPT" "$model_file" "$data_file" \
        --solver "$solver" \
        --timeout "$TIMEOUT" \
        --model-name "$model" \
        --instance-name "$instance"
}

process_workpiece() {
    local instance=$1

    echo "Processing $instance..."

    run_model "cp_model" "$instance" "chuffed"
    run_model "cp_model" "$instance" "gecode"
    run_model "cp_model" "$instance" "cp-sat"
    run_model "LNS_cp_model" "$instance" "gecode"
    run_model "mip_model" "$instance" "gurobi"
    run_model "int_mip_model" "$instance" "gurobi"

    echo "Done $instance"
}

echo "Starting correlation analysis..."

process_workpiece "spiral_stair_step"
process_workpiece "simple_stair_step"
process_workpiece "dashboard"
process_workpiece "speaker"
process_workpiece "coffee_table"

echo "Intermediate analysis completed"
echo "JSON outputs: $PROJECT_ROOT/correlation_analysis/json"
echo "Report: $PROJECT_ROOT/correlation_analysis/reports/objective_inertia_reports.txt"

exit 0
