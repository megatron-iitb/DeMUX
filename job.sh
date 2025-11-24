#!/bin/bash
#SBATCH --job-name=Exp_2_Improved
#SBATCH --partition=l40
#SBATCH --qos=l40
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/home/medal/anupam.rawat/Experiment_2/logs/Exp_2_modular.%j.log
#SBATCH --error=/home/medal/anupam.rawat/Experiment_2/logs/Exp_2_modular.%j.err

# Setup CUDA paths
export CUDA_HOME=/home/medal/anupam.rawat/miniconda3/envs/myenv
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# Enable offline mode for HuggingFace
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Set cache directories to use local pre-downloaded models
export HF_HOME=/home/medal/anupam.rawat/.cache/huggingface
export TORCH_HOME=/home/medal/anupam.rawat/.cache/torch
export HF_HUB_CACHE=/home/medal/anupam.rawat/.cache/huggingface/hub

# Activate conda environment
source /home/medal/anupam.rawat/miniconda3/etc/profile.d/conda.sh
conda activate myenv

# Change to experiment directory
cd /home/medal/anupam.rawat/Experiment_2

echo "======================================"
echo "Starting Experiment 2 on $(hostname) at $(date)"
echo "Using CUDA_HOME: $CUDA_HOME"
echo "HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
echo "Working directory: $(pwd)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "======================================"

# Run the improved experiment (with all fixes)
python experiment_2_improved.py

echo "======================================"
echo "Experiment 2 (Improved) completed at $(date)"
echo "======================================"
