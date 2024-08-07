import argparse
from math import sqrt
import wandb
import numpy as np
import os
from datetime import datetime

from eval_agent_scalability import eval as eval_scalability

IBEX = True if os.path.exists("/ibex/") else False
usr_name = os.environ["USER"]
scratch_dir = f"/ibex/user/{usr_name}/runs/scalability/" if IBEX \
    else f"/home/{usr_name}/scratch/runs/scalability/"

# Go over all agents in wandb
api = wandb.Api()
project = "gridworld_scalable"
runs = api.runs(f"lucascamara/{project}")

sweeps = ["6rnqmobc", "x413s9f2"] # later add f7zhhc9u

for run in runs:
    try:
        if run.sweep.id not in sweeps or run.state != "finished": 
            continue
    except:
        continue
    wandb_run = f"{project}/{run.id}"
    mod_wandb_run = "-".join(wandb_run.split("/"))

    # Test if file already exists
    for file in os.listdir(scratch_dir):
        if file.startswith(f"stats_{mod_wandb_run}"):
            print(f"Skipping {mod_wandb_run}")
            continue
    print(f"Computing {mod_wandb_run}")

    eval_scalability(wandb_run, online=False)