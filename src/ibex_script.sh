#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=batch
#SBATCH -J RobotSweep
#SBATCH -o /ibex/scratch/camaral/runs/RobotSweep.%J.out
#SBATCH -e /ibex/scratch/camaral/runs/RobotSweep.%J.err
#SBATCH --mail-user=lucas.camaradantasbezerra@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint=[gpu]

# conda init bash
# conda activate epymarl
cd /home/camaral/code/epymarl/src
# git pull
python run_ibex.py ofx576a7 -o