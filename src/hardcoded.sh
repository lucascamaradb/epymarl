#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=batch
#SBATCH -J RobotSweep_Hardcoded
#SBATCH -o /ibex/scratch/camaral/runs/RobotSweep.%J.out
#SBATCH -e /ibex/scratch/camaral/runs/RobotSweep.%J.err
#SBATCH --mail-user=lucas.camaradantasbezerra@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=96:00:00
#SBATCH --mem=64G


# conda init bash
# conda activate epymarl
cd /home/camaral/code/epymarl/src
# git pull
# python run_ibex.py ofx576a7 -o
python run_ibex.py gridworld_cnn_vs_mlp/kv8kdfuo -o