#!/bin/bash

REPORT_FILE="../solutions/reports/inertia_reports.txt"

if [ ! -f "$REPORT_FILE" ]; then
    echo "Error: Report file not found at: $REPORT_FILE"
    exit 1
fi

echo "=========================================================================="
echo "Best CP/LNS Models by Instance (Highest Inertia Value)"
echo "=========================================================================="
echo ""

instances=$(grep -E "cp_model_|LNS_cp_model" "$REPORT_FILE" | awk '{print $4}' | sort -u)

for instance in $instances; do
    echo "Instance: $instance"
    echo "----------"
    
    best_entry=$(grep -E "cp_model_|LNS_cp_model" "$REPORT_FILE" | grep " $instance " | awk '{printf "%.0f %s\n", $5, $0}' | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -z "$best_entry" ]; then
        echo "  No cp_model entries found for this instance"
        echo ""
        continue
    fi
    
    timestamp=$(echo "$best_entry" | awk '{print $1, $2}')
    solver=$(echo "$best_entry" | awk '{print $3}')
    objective=$(echo "$best_entry" | awk '{print $5}')
    
    total_models=$(grep -E "cp_model_|LNS_cp_model" "$REPORT_FILE" | grep " $instance " | wc -l)
    
    printf "  %-20s MOI: %20.0f\n" "Best Model: $solver" "$objective"
    printf "  %-20s %s\n" "Time:" "$timestamp"
    printf "  %-20s %d model(s) tested\n" "Alternatives:" "$total_models"
    
    echo "  All models for this instance:"
    grep -E "cp_model_|LNS_cp_model" "$REPORT_FILE" | grep " $instance " | awk '{printf "%.0f %s\n", $5, $0}' | sort -rn | cut -d' ' -f2- | while read line; do
        model_name=$(echo "$line" | awk '{print $3}')
        obj_value=$(echo "$line" | awk '{print $5}')
        printf "    - %-20s %20.0f\n" "$model_name:" "$obj_value"
    done
    
    echo ""
done

echo "=========================================================================="
echo "Summary Table (Best Model per Instance)"
echo "=========================================================================="
echo ""
printf "%-20s %-20s %20s\n" "Instance" "Best Model" "Inertia Value"
echo "------------------------------------------------------------"

for instance in $instances; do
    best_entry=$(grep -E "cp_model_|LNS_cp_model" "$REPORT_FILE" | grep " $instance " | awk '{printf "%.0f %s\n", $5, $0}' | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -n "$best_entry" ]; then
        model=$(echo "$best_entry" | awk '{print $3}')
        objective=$(echo "$best_entry" | awk '{print $5}')
        printf "%-20s %-20s %20.0f\n" "$instance" "$model" "$objective"
    fi
done

echo ""
echo "=========================================================================="
