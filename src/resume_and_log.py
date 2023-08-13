import pandas as pd 
import wandb
import numpy as np
api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("lucascamara/gridworld_paper")

# sweep_ids = ["lbt3s00m", "1sems41q", "zayq4n3l"]
sweep_ids = ["lbt3s00m", "zayq4n3l"]
summary_list, config_list, name_list = [], [], []
ids = {}
run_mean = {}
for run in runs: 
    if run.sweep.id not in sweep_ids or run.state != "finished":
        continue

    # assert run.summary.get("norm_by_respawn_test_return_mean",None) is not None

    if ids.get(run.config["robot_gym.respawn"], None) is None:
        ids[run.config["robot_gym.respawn"]] = []
        run_mean[run.config["robot_gym.respawn"]] = []

    ids[run.config["robot_gym.respawn"]].append(run.id)
    # ids[run.config["robot_gym.respawn"]].append("/".join(run.path))
    try:
        run_mean[run.config["robot_gym.respawn"]].append(run.summary["best_test_return_mean"])
    except:
        pass

stats = {}
for k,v in run_mean.items():
    stats[k] = {"avg": np.mean(v), "std": np.std(v) }

# Add normalized performance to all runs
for spawn_rate, id_list in ids.items():
    mean = stats[spawn_rate]["avg"]
    std  = stats[spawn_rate]["std"]
    for id in id_list:
        with wandb.init(project="gridworld_paper", id=id, resume="must") as run:
            try:
                run.summary["norm_by_respawn_test_return_mean"] = \
                    (run.summary["best_test_return_mean"]-mean)/std
            except:
                pass


print(ids)