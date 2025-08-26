#!/bin/bash

# 运行三个benchmark测试的脚本

echo "开始运行benchmark测试..."

# 运行BigBenchHard测试
echo "\n[1/3] 运行BigBenchHard测试..."
uv run python -m massgen.cli --benchmark --benchmark-config massgen/configs/benchmark_bigbenchhard.yaml

# 运行MuSR测试
echo "\n[2/3] 运行MuSR测试..."
uv run python -m massgen.cli --benchmark --benchmark-config massgen/configs/benchmark_musr.yaml

# 运行Hendrycks Math Level 5测试
echo "\n[3/3] 运行Hendrycks Math Level 5测试..."
uv run python -m massgen.cli --benchmark --benchmark-config massgen/configs/benchmark_hendrycks_math.yaml

echo "\n所有benchmark测试已完成！"
echo "结果文件保存在项目根目录下："
echo "- benchmark_bigbenchhard_results.json"
echo "- benchmark_musr_results.json"
echo "- benchmark_hendrycks_math_results.json"