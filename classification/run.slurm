#!/bin/bash
#SBATCH -J slurm_work
#SBATCH -p normal
#SBATCH --mail-type=all
#SBATCH --mail-user=panbk_shanghaitech@outlook.com
#SBATCH -N 1
#SBATCH -t 10-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%j.out
#SBATCH --error=%j.err

#SBATCH --cpus-per-task=8
# echo helloworld
python -u train.py mnist ours