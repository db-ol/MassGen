#!/bin/bash

# Script to run three benchmark tests

echo "Starting benchmark tests..."

# Run BigBenchHard test
echo "\n[1/3] Running BigBenchHard test..."
uv run python -m massgen.cli --benchmark --benchmark-config massgen/configs/benchmark_bigbenchhard.yaml

# Run MuSR test
echo "\n[2/3] Running MuSR test..."
uv run python -m massgen.cli --benchmark --benchmark-config massgen/configs/benchmark_musr.yaml

# Run Hendrycks Math Level 5 test
echo "\n[3/3] Running Hendrycks Math Level 5 test..."
uv run python -m massgen.cli --benchmark --benchmark-config massgen/configs/benchmark_hendrycks_math.yaml

echo "\nAll benchmark tests completed!"
echo "Result files saved in the project root directory:"
echo "- benchmark_bigbenchhard_results.json"
echo "- benchmark_musr_results.json"
echo "- benchmark_hendrycks_math_results.json"