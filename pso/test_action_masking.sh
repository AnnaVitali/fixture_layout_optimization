#!/bin/bash

# Test suite for action masking and reward modes
# Compares different configurations: (1) No masking, (2) With masking, (3) Delta rewards

WORKPIECE="coffee_table"
BASELINE="cp"
TIMESTEPS=5000
PENALTY_SCHEDULE="linear"
PENALTY_START=100.0
PENALTY_END=5000.0
ENTROPY=0.15
GAMMA=0.95

echo "=========================================="
echo "ACTION MASKING & REWARD MODE TEST SUITE"
echo "=========================================="
echo ""
echo "Workpiece: $WORKPIECE"
echo "Baseline: $BASELINE"
echo "Timesteps: $TIMESTEPS"
echo "Penalty Schedule: $PENALTY_SCHEDULE ($PENALTY_START -> $PENALTY_END)"
echo ""

cd /c/Users/anvitali/Desktop/fixture_layout_optimization/pso
source Hyperparameters_Tuning/.venv/Scripts/activate

# Test 1: Baseline (Deb rewards, no masking)
echo "=========================================="
echo "TEST 1: Deb Rewards (No Action Masking)"
echo "=========================================="
python python/rl_fixture_layout_gymnasium.py \
  --workpiece "$WORKPIECE" \
  --baseline "$BASELINE" \
  --timesteps $TIMESTEPS \
  --penalty-schedule "$PENALTY_SCHEDULE" \
  --penalty-start $PENALTY_START \
  --penalty-end $PENALTY_END \
  --entropy $ENTROPY \
  --gamma $GAMMA \
  --reward-mode deb \
  --seed 42

echo ""
echo ""

# Test 2: Delta rewards (no masking)
echo "=========================================="
echo "TEST 2: Delta Rewards (No Action Masking)"
echo "=========================================="
python python/rl_fixture_layout_gymnasium.py \
  --workpiece "$WORKPIECE" \
  --baseline "$BASELINE" \
  --timesteps $TIMESTEPS \
  --penalty-schedule "$PENALTY_SCHEDULE" \
  --penalty-start $PENALTY_START \
  --penalty-end $PENALTY_END \
  --entropy $ENTROPY \
  --gamma $GAMMA \
  --reward-mode delta \
  --seed 42

echo ""
echo ""

# Test 3: Deb rewards WITH masking
echo "=========================================="
echo "TEST 3: Deb Rewards (WITH Action Masking)"
echo "=========================================="
python python/rl_fixture_layout_gymnasium.py \
  --workpiece "$WORKPIECE" \
  --baseline "$BASELINE" \
  --timesteps $TIMESTEPS \
  --penalty-schedule "$PENALTY_SCHEDULE" \
  --penalty-start $PENALTY_START \
  --penalty-end $PENALTY_END \
  --entropy $ENTROPY \
  --gamma $GAMMA \
  --reward-mode deb \
  --use-action-masking \
  --seed 42

echo ""
echo ""

# Test 4: Delta rewards WITH masking
echo "=========================================="
echo "TEST 4: Delta Rewards (WITH Action Masking)"
echo "=========================================="
python python/rl_fixture_layout_gymnasium.py \
  --workpiece "$WORKPIECE" \
  --baseline "$BASELINE" \
  --timesteps $TIMESTEPS \
  --penalty-schedule "$PENALTY_SCHEDULE" \
  --penalty-start $PENALTY_START \
  --penalty-end $PENALTY_END \
  --entropy $ENTROPY \
  --gamma $GAMMA \
  --reward-mode delta \
  --use-action-masking \
  --seed 42

echo ""
echo "=========================================="
echo "TEST SUITE COMPLETE"
echo "=========================================="
