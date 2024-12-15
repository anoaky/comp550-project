#!/bin/bash
#
#SBATCH --job-name=bart
#SBATCH --time=06:00:00
#SBATCH --partition=nodegpupool12
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --wait-all-nodes=1
#SBATCH --mem=64G
#SBATCH --output=output/job_output_%j.out
#SBATCH --error=output/job_output_%j.err

source /home/asures3/setup-env.sh
export HF_HOME="${SLURM_TMPDIR}/huggingface"
export CUDA_LAUNCH_BLOCKING=1

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index -qr requirements.txt

srun --oversubscribe python $@