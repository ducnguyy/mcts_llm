#!/bin/bash
#SBATCH --job-name=ollama_srv
#SBATCH --partition=mesonet
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --account=m25146
#SBATCH --output=logs/ollama_server_%j.out
#SBATCH --error=logs/ollama_server_%j.err

set -e

echo "=== Ollama Server started: $(date) ==="
echo "=== Node: $(hostname) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

export OLLAMA_MODELS="$HOME/.ollama/models"
export OLLAMA_HOST="0.0.0.0:11434"
export OLLAMA_KEEP_ALIVE="-1"

# Make CUDA libs visible to Ollama's bundled runtime
export LD_LIBRARY_PATH=/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}

unset CUDA_VISIBLE_DEVICES
unset ROCR_VISIBLE_DEVICES
unset GPU_DEVICE_ORDINAL

# Start ollama server in background
ollama serve &
OLLAMA_PID=$!
echo "=== Ollama PID: $OLLAMA_PID ==="

# Wait for server to be ready
echo "Waiting for Ollama server to start..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama server is ready!"
        break
    fi
    sleep 2
done

# Pull the model
echo "=== Pulling model... ==="
ollama pull qwen2.5:14b

# Write connection info
echo "$(hostname):11434" > ollama_server_info.txt
echo "=== Server info written to ollama_server_info.txt ==="
echo "=== Server running on $(hostname):11434 ==="

# Quick test
echo "=== Testing model... ==="
curl -s http://localhost:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "qwen2.5:14b", "messages": [{"role": "user", "content": "Say hello"}], "max_tokens": 10}' \
    | head -c 200
echo ""
echo "=== Model test complete ==="

# Keep server running
echo "=== Ollama server running. Cancel this job when evaluation is done. ==="
wait $OLLAMA_PID
