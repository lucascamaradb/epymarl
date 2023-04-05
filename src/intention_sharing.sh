#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=batch
#SBATCH -J RobotSweep
#SBATCH -o /ibex/user/camaral/runs/RobotSweep.%J.out
#SBATCH -e /ibex/user/camaral/runs/RobotSweep.%J.err
#SBATCH --mail-user=lucas.camaradantasbezerra@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=48:00:00
#SBATCH --mem=100G
#SBATCH --gpus-per-node=1
#SBATCH --constraint="[p6000|p100|v100]"


# conda init bash
# conda activate epymarl
cd /home/camaral/code/epymarl/src
# git pull
python run_ibex.py gridworld_intention/tqsyzvjx -o