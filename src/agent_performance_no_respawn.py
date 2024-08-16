# %%
import wandb
import numpy as np
from tqdm import tqdm

# %% [markdown]
# ### Label runs according to their training setting

# %%
sweeps = ["ely91qns", "wp0f6uqu"]

# %%
api = wandb.Api()
path = "lucascamara/mrta_paper"
runs = api.runs(path)
run = runs[0]
print(run.summary)

# %%
import run_trained_agent as Run

# %%
# Evaluate each run in each level
import pickle
from datetime import datetime

try:
    raise ValueError
    with open("noresp_perf_2024-08-16 15:54:34.420192.pkl", "rb") as f:
        perf_dict = pickle.load(f)
    print(f"Loaded {len(perf_dict.keys())} runs from file")
except:
    perf_dict = {}

now = datetime.now()
start_from = 0
for i,run in enumerate(tqdm(runs)):
    if i<start_from:
        continue
    try:
        if run.sweep.id not in sweeps or run.state != "finished": 
            continue
    except:
        continue

    id = run.id
    if run.id in perf_dict.keys():
        print(f"Run {id} previously computed")
        continue
    print("#############################################")
    print("#############################################")
    print("#############################################")
    print(f"\t\tSTARTING NEW RUN: {i}/{len(runs)}, {id}")
    run_path = f"{path}/{id}"

    eval_config = {"robot_gym.respawn": False, "test_nepisode": 96, "seed": 111,}
    stats = Run.eval(run_path, eval_config)
    perf = {"test_return_mean": stats["test_return_mean"][0][1], "test_return_std": stats["test_return_std"][0][1]}

    perf_dict[id] = perf
    with open(f"noresp_perf_{now}.pkl", "wb") as f:
        pickle.dump(perf_dict, f)

# %% [markdown]
# ### Load `marl.csv`, replace performance, and save to `marl_noresp.csv`

# %%
import pandas as pd

path = "/home/lucas/code/mrta_paper/data/"
marl_df = pd.read_csv(path+"marl.csv")

# %%
marl_df.head()

# %%
marl_df.count()

# %%
marl_df.columns

# %%
# Check if all runs are in the dataframe
for run_id in perf_dict.keys():
    if run_id not in marl_df["run_id"].values:
        print(f"Run {run_id} not in dataframe")

# Check if there are runs in the dataframe that are not in the perf_dict
# count = 0
# for run_id in marl_df["run_id"].values:
#     if run_id not in perf_dict.keys():
#         print(f"Run {run_id} not in perf_dict")
#         count += 1
# print(f"{count} runs not in perf_dict")

# %%
# Create a new dataframe replacing "best_test_return_mean" with the new values in perf_dict
new_df = marl_df.copy()
# Get only the rows that have the run_id in perf_dict
new_df = new_df[new_df["run_id"].isin(perf_dict.keys())]
for run_id in perf_dict.keys():
    new_df.loc[new_df["run_id"]==run_id, "best_test_return_mean"] = perf_dict[run_id]["test_return_mean"]
    new_df.loc[new_df["run_id"]==run_id, "test_return_mean"] = perf_dict[run_id]["test_return_mean"]
    new_df.loc[new_df["run_id"]==run_id, "test_return_std"] = perf_dict[run_id]["test_return_std"]

new_df.head()

# %%
new_df.count()

# %%
# Save
new_df.to_csv(path+"marl_noresp3.csv", index=False)