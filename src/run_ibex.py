import os
import sys
import wandb
import numpy as np
import argparse
import threading
import robot_gym

from gym.envs.registration import register

IBEX = True if os.path.exists("/ibex/scratch/camaral/") else False
usr_name = "camaral" if os.path.exists("/home/camaral") else "lucas"
online = True if IBEX else False
scratch_dir = "/ibex/scratch/camaral/runs/" if IBEX \
    else f"/home/{usr_name}/scratch/runs/"
wandb_root = "lucascamara/gridworld/"
base_dir = f"/home/{usr_name}/code/epymarl"
# sys.path.append(base_dir)

import main
# from main import *

parser = argparse.ArgumentParser(description="Gridworld Training Script")

def config2txt(config):
    discard_start_with = "robot_gym"
    keys_to_discard = ["config", "env_config", "save_model", "save_path"]

    comb = []
    for k, v in config.items():
        if k.startswith(discard_start_with) or any([k==kd for kd in keys_to_discard]):
            continue
        comb.append(f"{k}={v}")
    if comb==[]:
        return ""
    txt = " ".join(comb)
    return txt+" "


def train(config=None):
    mode = "online" if online else "offline"
    with wandb.init(config=config, mode=mode) as run:
        config = wandb.config
        np.random.seed(config.seed)
        if IBEX: run.summary["ibex_job_id"] = os.environ["SLURM_JOBID"]

        # Save path
        save_path = scratch_dir + run.id + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        # Define environment key
        # env_key = f'robot_gym.env:GridWorld-{config["robot_gym.size"]}x{config["robot_gym.size"]}-{config["robot_gym.N_agents"]}a-{config["robot_gym.N_obj"]}o-{config["robot_gym.N_comm"]}c-{config["robot_gym.agent_range"]}v-v0'
        # env_key = f'robot_gym.env:GridWorld-MultiLevel-{config["robot_gym.size"]}x{config["robot_gym.size"]}-{config["robot_gym.N_agents"]}a-{config["robot_gym.N_obj"]}o-{config["robot_gym.N_comm"]}c-{config["robot_gym.agent_range"]}v-v0'
        env_key = register_env(run.id, config)
        print(f"Environment: {env_key}")

        # Define save path
        save_path = scratch_dir + run.id + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Define script to call
        n_parallel = os.getenv("SLURM_CPUS_PER_TASK") if IBEX else 52
        txt_args = f'main.py --config={config.config} --env-config={config.env_config} with env_args.key="{env_key}" {config2txt(config)}save_model=True save_path="{save_path}" wandb_sweep=True'
        if config.config != "qmix": txt_args += f" batch_size_run={n_parallel}"
        # txt_args = f'main.py --config=maddpg --env-config={config.env_config} with env_args.key="{env_key}" {config2txt(config)}save_model=True save_path="{save_path}" wandb_sweep=True'
        print("python3 " + txt_args)
        # print(txt_args.split(' '))

        # # Run EPyMARL training script
        main.main_from_arg(txt_args.split(' '))

def register_env(id,config):
    env_id = f"GridWorld-Custom-{id}-v0"
    register(
        id=env_id,
        entry_point="robot_gym.env:GridWorldEnv",
        kwargs={
            "sz": (config["robot_gym.size"], config["robot_gym.size"]),
            "n_agents": config["robot_gym.N_agents"],
            "n_obj": config["robot_gym.N_obj"],
            "render": False,
            "comm": config["robot_gym.N_comm"],
            # "view_range": config["robot_gym.view_range"],
            "view_range": config["robot_gym.agent_range"],
            "comm_range": config["robot_gym.agent_range"],
            "one_hot_obj_lvl": True,
            "max_obj_lvl": 3,
        },
    )
    return env_id

if __name__ == "__main__":
    parser.add_argument("wandb_sweep", type=str, help="WANDB Sweep ID")
    parser.add_argument("-o", "--online", action="store_true", help="Upload experiment to WANDB")
    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args(["lo6mqvy5"])

    sweep_id = wandb_root + args.wandb_sweep
    online = args.online
    wandb.agent(sweep_id, train)
    # wandb.agent(sweep_id, train, count=1)