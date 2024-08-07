import argparse
from math import sqrt
import wandb
import numpy as np
import os
from datetime import datetime

import run_trained_agent as Run

IBEX = True if os.path.exists("/ibex/") else False
usr_name = os.environ["USER"]
scratch_dir = f"/ibex/user/{usr_name}/runs/scalability/" if IBEX \
    else f"/home/{usr_name}/scratch/runs/scalability/"

parser = argparse.ArgumentParser(description="Gridworld Agent Scalability Evaluation")

N_agents = np.round(10**np.arange(1,3.1,.1)).astype(int)
agent_density = 0.025 # = 10/(20^2)
obj_density = 0.025 # = 10/(20^2)
obj_ratio = [.4, .3, .3]
# combinations = [(int(sqrt(N_agents/agent_density)), [int(4*N_agents*x) for x in obj_ratio], N_agents) for N_agents in range(10, 100, 20)]
combinations = [(int(sqrt(n/agent_density)), [int(4*n*x) for x in obj_ratio], n) for n in N_agents]

const_config = {
    # "robot_gym.respawn": True,
    "robot_gym.Lsec": 1,
    # "test_nepisode": 192,
    # "test_nepisode": 48,
    # "test_nepisode": 96,
    "test_nepisode": 32,
}

configs = [{**const_config, 
            "robot_gym.size": size,
            "robot_gym.N_obj": N_obj,
            "robot_gym.N_agents": N_agents,
            } for size, N_obj, N_agents in combinations]
stat_dict = []

def eval(wandb_run, online=False):
    now = datetime.now()

    for i,config in enumerate(configs):
        print(f"Running {i+1}/{len(configs)}")
        stats = Run.eval(wandb_run, eval_config=config)
        stats = {k:v[0][1] for k,v in stats.items() if k in ["test_return_mean", "test_return_std"]}
        print(stats)
        mod_wandb_run = "-".join(wandb_run.split("/"))
        with open(f"{scratch_dir}stats_{mod_wandb_run} {now}.txt", "a") as f:
            f.write(f"{config['robot_gym.N_agents']} agents: {stats}\n")
        stat_dict.append(stats)

    print(stat_dict)
    stats = [[n_agent, avg_rwd] for n_agent, avg_rwd in zip([config["robot_gym.N_agents"] for config in configs], [stat["test_return_mean"] for stat in stat_dict])]
    print(stats)

    if online:
        project, id = wandb_run.split("/")
        with wandb.init(project=project, id=id, resume="must") as run:
            table = wandb.Table(data=stats, columns=["Number of Agents", "Average Reward"])
            wandb.log({"agent_scalability": wandb.plot.line(table, "Number of Agents", "Average Reward", title="Agent Scalability")})
            print("Done!")

if __name__=="__main__":
    parser.add_argument("wandb_run", type=str, help="WANDB Run ID")
    parser.add_argument("-o", "--online", action="store_true", help="Upload result to WANDB")
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args(["gridworld_scalable/ez77t1jm", "-o"])

    eval(args.wandb_run, args.online)