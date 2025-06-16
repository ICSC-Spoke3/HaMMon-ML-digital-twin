#!/bin/bash
#SBATCH --job-name=sattunet-ddp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=gprod
#SBATCH --gres=gpu:a100:4
#SBATCH --time=08:00:00
#SBATCH --output=.logs/%x-%j.out
#SBATCH --error=.logs/%x-%j.err


# Ensure hidden logs directory exists
mkdir -p .logs

# Initialize Conda environment (assuming module already loaded)
source /software/prod/build/spack/spack_v2/opt/spack/linux-rocky8-broadwell/intel-2021.9.0/miniconda3-22.11.1-hvf7epomuxpgxwj66akf3t3mvqwmsxcv/etc/profile.d/conda.sh
conda activate ml-digital-twin-env

# Run the training script with user-provided arguments
python ddp.py "$@"

