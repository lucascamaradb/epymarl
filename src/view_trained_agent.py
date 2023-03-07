import torch
import wandb
import os
import robot_gym
from modules.agents import CNNAgent
from types import SimpleNamespace as SN

def build_env(config):
    kwargs={
            "sz": (config["robot_gym.size"], config["robot_gym.size"]),
            "n_agents": config["robot_gym.N_agents"],
            "n_obj": config["robot_gym.N_obj"],
            "render": True,
            "comm": config["robot_gym.N_comm"],
            "hardcoded_comm": config["robot_gym.hardcoded_comm"],
            "view_range": config["robot_gym.view_range"],
            "comm_range": config["robot_gym.comm_range"],
            "Lsec": config["robot_gym.Lsec"],
            "one_hot_obj_lvl": True,
            "max_obj_lvl": 3,
        }
    return robot_gym.env.GridWorldEnv(**kwargs)

def build_agent(config, env):
    obs = env.reset()
    input_shape = obs[0].shape
    input_shape = (input_shape[2], input_shape[0], input_shape[1])
    n_actions = env.map.robots[0].action_space.n
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args = SN(**{
        "agent_arch": config["agent_arch"],
        "n_actions": n_actions,
        "device": device,
    })
    return CNNAgent(input_shape, args)

def get_actions(agent, obs):
    outs = [0]*len(obs)
    for i,o in enumerate(obs):
        o = torch.permute(torch.Tensor(o).to(agent.device), dims=(2,0,1))
        out = agent(o)[0]
        out = torch.nn.functional.softmax(out, dim=-1)
        out = torch.argmax(out)
        outs[i] = out.item()
    return outs

run_id = "k39wgnw7"
model_path = "/home/camaral/scratch/runs/"+run_id+"/model/"
if not os.path.exists(model_path): os.makedirs(model_path)

api = wandb.Api()
run = api.run("lucascamara/gridworld_cnn_vs_mlp/"+run_id)
model = [x for x in run.logged_artifacts() if x.type=="model"][0]
model.download(model_path)

env = build_env(run.config)
agent = build_agent(run.config, env)
agent.load_state_dict(torch.load(model_path+"agent.th", map_location=lambda storage, loc: storage))

with torch.no_grad():
    try:
        while True:
            tot_rwd = 0
            obs = env.reset()
            for i in range(100):
                actions = get_actions(agent, obs)
                obs, rwd, done, _ = env.step(actions)
                env.render()
                tot_rwd += sum(rwd)
    except KeyboardInterrupt:
        env.close()
    

print("done")