#!/bin/bash

REPORT_FILE="../solutions/reports/inertia_reports.txt"

if [ ! -f "$REPORT_FILE" ]; then
    echo "Error: Report file not found at: $REPORT_FILE"
    exit 1
fi

echo "=========================================================================="
echo "Best Solution Provider by Instance (Highest Inertia Value)"
echo "=========================================================================="
echo ""

# Extract unique instances (skip header)
instances=$(tail -n +5 "$REPORT_FILE" | awk '{print $4}' | sort -u)

for instance in $instances; do
    echo "Instance: $instance"
    echo "----------"
    
    # Find best entry for this instance (numeric sort on objective value)
    best_entry=$(grep " $instance " "$REPORT_FILE" | awk '{printf "%.0f %s\n", $5, $0}' | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -z "$best_entry" ]; then
        echo "  No entries found for this instance"
        echo ""
        continue
    fi
    
    timestamp=$(echo "$best_entry" | awk '{print $1, $2}')
    provider=$(echo "$best_entry" | awk '{print $3}')
    objective=$(echo "$best_entry" | awk '{print $5}')
    
    total_providers=$(grep " $instance " "$REPORT_FILE" | wc -l)
    
    printf "  %-30s MOI: %20.0f\n" "Best Provider: $provider" "$objective"
    printf "  %-30s %s\n" "Time:" "$timestamp"
    printf "  %-30s %d solution provider(s) tested\n" "Alternatives:" "$total_providers"
    
    echo "  All providers for this instance (ranked by MOI):"
    grep " $instance " "$REPORT_FILE" | awk '{printf "%.0f %s\n", $5, $0}' | sort -rn | cut -d' ' -f2- | while read line; do
        provider_name=$(echo "$line" | awk '{print $3}')
        obj_value=$(echo "$line" | awk '{print $5}')
        printf "    - %-30s %20.0f\n" "$provider_name:" "$obj_value"
    done
    
    echo ""
done

echo "=========================================================================="
echo "Summary Table (Best Provider per Instance)"
echo "=========================================================================="
echo ""
printf "%-20s %-25s %20s\n" "Instance" "Best Provider" "Inertia Value"
echo "------------------------------------------------------------"

for instance in $instances; do
    best_entry=$(grep " $instance " "$REPORT_FILE" | awk '{printf "%.0f %s\n", $5, $0}' | sort -rn | head -1 | cut -d' ' -f2-)
    
    if [ -n "$best_entry" ]; then
        provider=$(echo "$best_entry" | awk '{print $3}')
        objective=$(echo "$best_entry" | awk '{print $5}')
        printf "%-20s %-25s %20.0f\n" "$instance" "$provider" "$objective"
    fi
done

echo ""
echo "=========================================================================="
echo "Provider Performance Summary (by number of best solutions)"
echo "=========================================================================="
echo ""

# Count how many times each provider is the best
declare -A provider_wins
declare -a unique_providers

for instance in $instances; do
    best_entry=$(grep " $instance " "$REPORT_FILE" | awk '{printf "%.0f %s\n", $5, $0}' | sort -rn | head -1 | cut -d' ' -f2-)
    provider=$(echo "$best_entry" | awk '{print $3}')
    provider_wins["$provider"]=$((${provider_wins["$provider"]:-0} + 1))
    
    # Track unique providers
    if [[ ! " ${unique_providers[@]} " =~ " ${provider} " ]]; then
        unique_providers+=("$provider")
    fi
done

# Sort providers by number of wins
echo "Provider | Best Solutions | Instances Won"
echo "---------|----------------|---------------"

for provider in "${unique_providers[@]}"; do
    wins=${provider_wins["$provider"]:-0}
done | sort -t'|' -k2 -rn

# Alternative: Using a temporary array for sorting
{
    for provider in "${unique_providers[@]}"; do
        wins=${provider_wins["$provider"]:-0}
        printf "%s %d\n" "$provider" "$wins"
    done
} | sort -k2 -rn | while read provider wins; do
    printf "%-30s %d\n" "$provider" "$wins"
done

echo ""
echo "=========================================================================="
