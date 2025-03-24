#!/bin/bash
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=normal
#SBATCH --gres=gpu:A100.80gb:4
#SBATCH --nodes=1
#SBATCH --mem=50G
#SBATCH --ntasks-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=./log/%x-%N-%j.out
#SBATCH --error=./log/%x-%N-%j.err

source /home/miniconda/bin/activate cdfsod

# Set CUDA environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export MASTER_PORT=$((12000 + $RANDOM % 1000))

sh main_results.sh