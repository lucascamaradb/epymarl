import os
import wandb
import numpy as np
import argparse
import robot_gym
from multiprocessing import cpu_count

import gym
from gym.envs.registration import register

IBEX = True if os.path.exists("/ibex/") else False
usr_name = os.environ["USER"]
DSA = usr_name=="ubuntu"
scratch_dir = f"/ibex/user/{usr_name}/runs/" if IBEX \
    else f"/home/{usr_name}/scratch/runs/"
# base_dir = f"/home/{usr_name}/code/epymarl"
# sys.path.append(base_dir)

import main

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
    # "config": "qmix",
    "config": "mappo", 
    "critic_type": "cnn_cv_critic",
    "env_config": "gridworld", "agent": "cnn",
    # "agent_arch": "resnet;conv2d,64,1;relu;interpolate,2;conv2d,1,1;relu;interpolate,1.7&",
    # "critic_arch": "resnet&batchNorm1d;linear,128;relu;linear,32;relu",
    "agent_arch": "unet,8,1,2&",
    # "critic_arch": "unet,8,1,2&batchNorm1d;linear,50;relu",
    "strategy": "cnn",
    # "strategy": "hardcoded",
    # "env_config": "gymma",
    "hidden_dim": 512,
    "obs_agent_id": False,
    "robot_gym.Lsec": 2,
    "robot_gym.N_agents": 10,
    # "env_args.hardcoded": False,
    "env_args.hardcoded": "comm",
    # "env_args.hardcoded": True,
    "agent_distance_exp": 1.,
    "robot_gym.N_comm": 4,
    "robot_gym.N_obj": [4, 3, 3],
    "robot_gym.comm_range": 8,
    # "robot_gym.size": 40,
    "robot_gym.size": 20,
    "robot_gym.view_range": 4,
    "robot_gym.action_grid": True,
    "robot_gym.respawn": False,
    "robot_gym.obj_lvl_rwd_exp": 2.,
    "robot_gym.view_self": False,
    # "action_grid": True,
    "action_grid": False,
    "current_target_factor": None,
    "agent_reeval_rate": True,
    "filter_avail_by_objects": False,
    # "robot_gym.share_intention": "path",
    # "share_intention": "path",
    "robot_gym.share_intention": False,
    "share_intention": False,
    # "robot_gym.share_intention": "channel",
    # "share_intention": "channel",
    "seed": 10,
    "t_max": 2_000_000,
    "env_args.curriculum": True, #################
    "standardise_returns": True, #################
}

def run_hardcoded(env, config):
    env = robot_gym.env.HardcodedWrapper(env, policy=robot_gym.policy.HighestLvlObjPolicy, 
                                         agent_reeval_rate=config["agent_reeval_rate"])
    rwd_lst = []
    obj_pickup_rate = []
    try:
        for _ in range(96):
            obs = env.reset()
            tot_rwd = 0
            for k in range(100):
                obs, rwd, done, _ = env.step()
                env.render()
                tot_rwd += sum(rwd)
                if any(done): break

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


def train(config=None, default=False, online=False):
    mode = "online" if online else "offline"
    with wandb.init(config=config, mode=mode) as run:
        config = DEFAULT_CONFIG if default else wandb.config
        try:
            np.random.seed(config["seed"])
        except:
            pass
        if IBEX: run.summary["ibex_job_id"] = os.environ["SLURM_JOBID"]
        run.summary["username"] = os.environ["USER"]

        # Save path
        save_path = scratch_dir + run.id + "/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Define environment key
        if config.get("env_args.curriculum", False):
            env_key = register_curriculum_env(run.id, config)
        else:
            env_key = register_env(run.id, config)
        print(f"Environment: {env_key}")

        if config.get("current_target_factor", None) is not None:
            run.summary["log_current_target_factor"] = np.log(config["current_target_factor"])
        else:
            run.summary["log_current_target_factor"] = None

        # if config["env_args.hardcoded"]==True:
        if config.get("env_args.hardcoded", False) == True:
            results_dict = run_hardcoded(gym.make(env_key), config)
            run.summary["best_test_return_mean"] = results_dict["return_mean"]
            # run.summary["best_test_return_std"] = results_dict["return_std"]
            for k,v in results_dict.items():
                run.summary[k] = v
        else:
            # Define save path
            save_path = scratch_dir + run.id + "/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Define script to call
            n_parallel = config.get("buffer_size", None)
            if n_parallel is None:
                n_parallel = 16 if IBEX else min(cpu_count()//2, 16)
                # n_parallel = int(os.getenv("SLURM_CPUS_PER_TASK")) if IBEX else min(cpu_count()//2, 16)
            config["batch_size_run"] = n_parallel # add number of parallel envs to config
            txt_args = f'main.py --config={config["config"]} --env-config={config["env_config"]} with env_args.key="{env_key}" {config2txt(config)}save_model=True save_path="{save_path}" wandb_sweep=True'
            # if config["config"] not in ["qmix", "vdn"]: txt_args += f" runner=parallel batch_size_run={n_parallel}"
            txt_args += f" runner=parallel batch_size_run={n_parallel}"
            if not IBEX and not DSA:
                txt_args += " use_cuda=False"
            # if True: txt_args += f" runner=\"episode\" batch_size_run={1}"
            print("python3 " + txt_args)

            # Run EPyMARL training script
            main.main_from_arg(txt_args.split(' '))

def register_env(id, config):
    env_id = f"GridWorld-Custom-{id}-v0"
    kwargs={
            "sz": (config["robot_gym.size"], config["robot_gym.size"]),
            "n_agents": config["robot_gym.N_agents"],
            "n_obj": config["robot_gym.N_obj"],
            "render": False,#not IBEX,
            # "render": True,
            "comm": config["robot_gym.N_comm"],
            # "hardcoded_comm": config["robot_gym.hardcoded_comm"],
            "view_range": config["robot_gym.view_range"],
            "comm_range": config["robot_gym.comm_range"],
            "Lsec": config["robot_gym.Lsec"],
            "one_hot_obj_lvl": True,
            "obj_lvl_rwd_exp": config["robot_gym.obj_lvl_rwd_exp"],
            "view_self": config.get("robot_gym.view_self", True),
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

def register_curriculum_env(id, config):
    env_id = f"GridWorld-Curriculum-{id}-v0"
    kwargs={
            "render": False,
            "train_args": config["robot_gym.train"],
            "eval_args": config["robot_gym.eval"],
        }
    print(kwargs)

    register(
        id=env_id,
        entry_point="robot_gym.env:CurriculumEnv",
        kwargs=kwargs,
        order_enforce=False, # VERY IMPORTANT!!!
    )
    return env_id

if __name__ == "__main__":
    parser.add_argument("wandb_sweep", type=str, help="WANDB Sweep ID")
    parser.add_argument("-o", "--online", action="store_true", help="Upload experiment to WANDB")
    parser.add_argument("-c", "--count", type=int, default=0, help="Run count (optional)")
    try:
        args = parser.parse_args()
        default_config = False
    except:
        # args = parser.parse_args(["gridworld_paper/v7f1ijmt", "-c", "1"])
        args = parser.parse_args(["gridworld_curriculum/sb75iye0", "-c", "1"])
        default_config = False # overrides config sent from W&B

    sweep_id = args.wandb_sweep
    run_count = args.count if args.count > 0 else None
    online = args.online
    wandb.agent(sweep_id, lambda *args, **kw: train(default=default_config, online=online, *args, **kw), count=run_count)