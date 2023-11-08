#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=batch
#SBATCH -J RobotSweep
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1
#SBATCH --constraint=v100

# conda init bash
# conda activate epymarl
cd /home/$USER/code/epymarl/src
# git pull
python run_ibex.py gridworld_paper/5opcfwfu -o