#!/bin/bash
#SBATCH --job-name=mcts_eval
#SBATCH --partition=mesonet
#SBATCH --nodes=1
#SBATCH --cpus-per-task=28
#SBATCH --gres=gpu:1
#SBATCH --mem=256G
#SBATCH --time=8:00:00
#SBATCH --account=m25146
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -e

echo "=== Evaluation started: $(date) ==="
echo "=== Node: $(hostname) ==="

# Read Ollama server info
if [ ! -f ollama_server_info.txt ]; then
    echo "ERROR: ollama_server_info.txt not found. Start the Ollama server first."
    echo "Run: sbatch start_ollama.sh"
    exit 1
fi

OLLAMA_SERVER=$(cat ollama_server_info.txt)
echo "=== Ollama server: $OLLAMA_SERVER ==="

# Set environment variables for llm_client.py
export OLLAMA_BASE_URL="http://${OLLAMA_SERVER}/v1"
export OLLAMA_MODEL="qwen2.5:14b"

# Check Ollama is reachable
echo "=== Testing connection to Ollama... ==="
curl -s "http://${OLLAMA_SERVER}/api/tags" > /dev/null 2>&1 || {
    echo "ERROR: Cannot reach Ollama server at ${OLLAMA_SERVER}"
    exit 1
}
echo "=== Ollama server reachable ==="

# Activate Python env
source ~/mcts_llm/mcts_env/bin/activate
echo "=== Python: $(which python) ==="

# Run evaluation
echo "=== Running evaluation... ==="
python evaluate_mctsr.py \
    --task math \
    --output results_mctsr.json

echo "=== Evaluation finished: $(date) ==="
echo "=== Results saved to results_mctsr.json ==="
