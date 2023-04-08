import os
import sys
import wandb
import numpy as np
import argparse
import threading
import robot_gym

import gym
from gym.envs.registration import register

IBEX = True if os.path.exists("/ibex/user/camaral/") else False
usr_name = "camaral" if os.path.exists("/home/camaral") else "lucas"
online = True if IBEX else False
scratch_dir = "/ibex/user/camaral/runs/" if IBEX \
    else f"/home/{usr_name}/scratch/runs/"
wandb_root = "lucascamara/gridworld/"
base_dir = f"/home/{usr_name}/code/epymarl"
# sys.path.append(base_dir)

import main
# from main import *

parser = argparse.ArgumentParser(description="Gridworld Training Script")

def config2txt(config):
    discard_start_with = "robot_gym"
    keys_to_discard = ["config", "env_config", "save_model", "save_path", "strategy"]
    requires_quotes = ["critic_arch", "agent_arch"]

    comb = []
    for k, v in config.items():
        if k.startswith(discard_start_with) or k in keys_to_discard:
            continue
        if k in requires_quotes:
            comb.append(f'{k}="{v}"')
        else:
            comb.append(f"{k}={v}")
    if comb==[]:
        return ""
    txt = " ".join(comb)
    return txt+" "

DEFAULT_CONFIG = {
    "buffer_size": 10,
    # "config": "vdn",
    "config": "mappo", 
    "critic_type": "cnn_cv_critic",
    "env_config": "gridworld", "agent": "cnn",
    # "agent_arch": "resnet;conv2d,64,1;relu;interpolate,2;conv2d,1,1;relu;interpolate,1.7&",
    # "critic_arch": "resnet&batchNorm1d;linear,128;relu;linear,32;relu",
    "agent_arch": "batchNorm2d;conv2d,32,3,1,1;relu;conv2d,1,3,1,1&",
    "critic_arch": "batchNorm2d;conv2d,32,3,1,1;relu;conv2d,1,3,1,1&batchNorm1d;linear,32;relu",
    "strategy": "cnn",
    # "strategy": "hardcoded",
    # "env_config": "gymma",
    "hidden_dim": 512,
    "obs_agent_id": False,
    "robot_gym.Lsec": 2,
    "robot_gym.N_agents": 10,
    # "robot_gym.N_comm": 2,
    # "robot_gym.hardcoded_comm": False,
    "env_args.hardcoded": False,
    "robot_gym.N_comm": 0,
    "robot_gym.N_obj": [4, 3, 3],
    "robot_gym.comm_range": 8,
    # "robot_gym.size": 40,
    "robot_gym.size": 20,
    "robot_gym.view_range": 8,
    "robot_gym.action_grid": True,
    "robot_gym.respawn": False,
    "action_grid": True,
    "current_target_factor": None,
    # "robot_gym.share_intention": "path",
    # "share_intention": "path",
    "robot_gym.share_intention": False,
    "share_intention": False,
    "seed": 10,
    "t_max": 600_000,
}

def run_hardcoded(env, config):
    env = robot_gym.env.HardcodedWrapper(env)
    rwd_lst = []
    obj_pickup_rate = []
    try:
        for _ in range(100):
            obs = env.reset()
            tot_rwd = 0
            for k in range(100):
                try:
                    obs, rwd, done, _ = env.step()
                    env.render()
                    tot_rwd += sum(rwd)
                    if any(done): break
                except Exception as e:
                    pass

            print(f"TOTAL REWARD: {tot_rwd}")
            rwd_lst.append(tot_rwd)
            obj_pickup_rate.append(env.env.obj_pickup_rate())
            # print(obj_pickup_rate[-1])

        print(f"AVERAGE REWARD: {np.mean(rwd_lst)}")
        print(f"STD DEV: {np.std(rwd_lst)}")
        results = {}
        results["return_mean"] = np.mean(rwd_lst)
        results["return_std"] = np.std(rwd_lst)
        for i in range(len(obj_pickup_rate[0])):
            results[f"pickup_rate_{i+1}"] = np.mean([x[i] for x in obj_pickup_rate])
        return results

    except KeyboardInterrupt:
        env.close()


def train(config=None, default=False):
    mode = "online" if online else "offline"
    with wandb.init(config=config, mode=mode) as run:
        config = DEFAULT_CONFIG if default else wandb.config
        try:
            np.random.seed(config["seed"])
        except:
            pass
        if IBEX: run.summary["ibex_job_id"] = os.environ["SLURM_JOBID"]

        # Save path
        save_path = scratch_dir + run.id + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Define environment key
        env_key = register_env(run.id, config)
        print(f"Environment: {env_key}")

        # if config["env_args.hardcoded"]==True:
        if config.get("env_args.hardcoded", False) == True:
            results_dict = run_hardcoded(gym.make(env_key), config)
            run.summary["best_test_return_mean"] = results_dict["return_mean"]
            for k,v in results_dict.items():
                run.summary[k] = v
        else:
            # Define save path
            save_path = scratch_dir + run.id + "/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Define script to call
            n_parallel = os.getenv("SLURM_CPUS_PER_TASK") if IBEX else 2
            txt_args = f'main.py --config={config["config"]} --env-config={config["env_config"]} with env_args.key="{env_key}" {config2txt(config)}save_model=True save_path="{save_path}" wandb_sweep=True'
            # txt_args = f'main.py --config={config["config"]} --env-config=gridworld with env_args.key="{env_key}" {config2txt(config)}save_model=True save_path="{save_path}" wandb_sweep=True'
            if config["config"] not in ["qmix", "vdn"]: txt_args += f" batch_size_run={n_parallel}"
            # if True: txt_args += f" runner=\"episode\" batch_size_run={1}"
            # txt_args = f'main.py --config=vdn --env-config={config.env_config} with env_args.key="{env_key}" {config2txt(config)}save_model=True save_path="{save_path}" wandb_sweep=True'
            print("python3 " + txt_args)

            # Run EPyMARL training script
            main.main_from_arg(txt_args.split(' '))

def register_env(id,config):
    env_id = f"GridWorld-Custom-{id}-v0"
    kwargs={
            "sz": (config["robot_gym.size"], config["robot_gym.size"]),
            "n_agents": config["robot_gym.N_agents"],
            "n_obj": config["robot_gym.N_obj"],
            "render": False,#not IBEX,
            "comm": config["robot_gym.N_comm"],
            # "hardcoded_comm": config["robot_gym.hardcoded_comm"],
            "view_range": config["robot_gym.view_range"],
            "comm_range": config["robot_gym.comm_range"],
            "Lsec": config["robot_gym.Lsec"],
            "one_hot_obj_lvl": True,
            "max_obj_lvl": 3,
            "action_grid": config["robot_gym.action_grid"],
            "share_intention": config["robot_gym.share_intention"],
            "respawn": config["robot_gym.respawn"],
        }
    print(kwargs)

    register(
        id=env_id,
        entry_point="robot_gym.env:GridWorldEnv",
        kwargs=kwargs,
        order_enforce=False, # VERY IMPORTANT!!!
    )
    return env_id

if __name__ == "__main__":
    parser.add_argument("wandb_sweep", type=str, help="WANDB Sweep ID")
    parser.add_argument("-o", "--online", action="store_true", help="Upload experiment to WANDB")
    try:
        args = parser.parse_args()
        default_config = False
    except:
        args = parser.parse_args(["gridworld_intention/uut7ibba"])
        default_config = True

    # sweep_id = wandb_root + args.wandb_sweep
    sweep_id = args.wandb_sweep
    online = args.online
    wandb.agent(sweep_id, lambda *args, **kw: train(default=default_config,*args, **kw))
    # wandb.agent(sweep_id, train, count=1)