from utils.nn_utils import net_from_string

import torch
import torch.nn as nn

import numpy as np
from math import ceil

class CustomAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CustomAgent, self).__init__()

    def forward(self, input):
        raise NotImplementedError("Custom agent not implemented")


class CNNAgent(CustomAgent):
    def __init__(self, input_shape, args):
        super().__init__(input_shape, args)

        self.device = torch.device(args.device)
        self.in_shape = input_shape
        self.in_channels = input_shape[0]
        self.intention = None
        self.current_target_factor = args.current_target_factor

        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.agent_distance_exp = self.dist_grid(input_shape[1], gamma=args.agent_distance_exp)["0"]

        self.n_out_channels = int(args.n_actions/(input_shape[1]*input_shape[2]))
        self.target_shape = (self.n_out_channels,*input_shape[1:])
        assert args.action_grid, "Only action grid supported"
        assert self.n_out_channels == 1, "Comms not supported"
        self.net, self.out_shape = net_from_string(args.agent_arch, self.in_shape, target_shape=self.target_shape)
        self.net = self.net.to(self.device)

        # self.dist_given_act = self.dist_grid(self.out_shape[-1], gamma=.5)
        # self.multiplier_act = torch.cat([x.unsqueeze(0) for x in self.dist_given_act.values()],0).to(self.device)

    def forward(self, input, hidden_state=None, env_info=None):
        if input.dim() == 3:
            input = input.unsqueeze(0)
        elif input.dim()<3:
            raise ValueError("Not enough dimensions in input")

        v = self.net(input)
        v = self.agent_distance_exp*v # Apply distance factor
        self.intention = v
        try:
            v = torch.flatten(v, start_dim=-3)
        except:
            print(v)

        v, target_update = self.target_update(v, env_info)
        env_info = self._flatten_env_info(env_info)
        return v, torch.Tensor([0]), target_update, env_info

    def init_hidden(self):
        return torch.Tensor([0])

    def _get_output_shape(self):
        with torch.no_grad():
            v = torch.randn(self.in_shape).unsqueeze(0).to(self.device)
            out = self.net(v)[0]
        return out.shape

        # v = torch.randn(self.in_shape)
        # v, _ = self.forward(v)
        # return v.shape

    def target_update(self, v, env_info=None):
        # TODO: Consider communications
        # Decides whether to update the controller's target or not
        L = v.shape[0]
        v = v.view(-1,self.n_agents,self.n_actions)
        bs = v.shape[0]
        target_update = torch.ones(bs, self.n_agents, 1, dtype=torch.uint8)
        if not isinstance(env_info, list):
            env_info = [env_info]

        for i,worker_env_info in enumerate(env_info):
            if worker_env_info in [None,0] or worker_env_info.get("robot_info", None) is None:
                # If there's no information, take all actions
                continue # target_update=1
            worker_env_info = worker_env_info["robot_info"]
            assert len(worker_env_info)==self.n_agents, "Expected len(env_info)==n_agents, from each worker"
            for j,(pos,cmd) in enumerate(worker_env_info):
                if cmd is None:
                    continue
                dif = (cmd[0]-pos[0], cmd[1]-pos[1])
                v[i,j,:], target_update[i,j,0] = self.target_update_policy(v[i,j,:], dif)

        v = v.view(L, self.n_actions)
        target_update = target_update.view(L, 1)
        return v, target_update

    def target_update_policy(self, actions, current_dif):
        if self.current_target_factor is not None:
            # current_dif = self._clip_to_obs_range(current_dif)
            actions[self._dif_to_flat(current_dif)] += np.log(self.current_target_factor)
            # actions[self._dif_to_flat(current_dif)] += 10
            return actions, 1
        # If no target factor, never reevaluate a target
        if self._dif_within_obs(current_dif):
            # current_dif = self._clip_to_obs_range(current_dif)
            actions = -1e10*torch.ones_like(actions)
            actions[self._dif_to_flat(current_dif)] = 1e10
            return actions, 0
        else:
            return actions, 1

    def _dif_within_obs(self, dif):
        sz = self.out_shape[1:]
        return 0 <= dif[0]+sz[0]//2 < sz[0] and 0 <= dif[1]+sz[1]//2 < sz[1]
    
    def _clip_to_obs_range(self, dif):
        sz = self.out_shape[1:]
        bound = (sz[0]//2, sz[1]//2)
        return (min(max(-bound[0], dif[0]), bound[0]-1), min(max(-bound[1], dif[1]), bound[1]-1))
    
    def _dif_to_flat(self, dif):
        sz = self.out_shape[1:]
        dif = (dif[0]+sz[0]//2, dif[1]+sz[1]//2)
        try:
            return np.ravel_multi_index(dif, sz)
        except:
            return -1
    # def _dif_to_flat(self, dif):
    #     sz = self.out_shape
    #     assert len(sz)==3, "Output shape should be 3D"
    #     difs = []
    #     dif = (dif[0]+sz[1]//2, dif[1]+sz[2]//2)
    #     for c in range(self.out_shape[0]):
    #         comm_dif = (c, *dif)
    #         try:
    #             print(comm_dif)
    #             print(np.ravel_multi_index(comm_dif, sz))
    #             difs.append(np.ravel_multi_index(comm_dif, sz))
    #         except:
    #             difs.append(-1)
    #     return difs
        
    def _flatten_env_info(self, env_info):
        if not isinstance(env_info, list):
            env_info = [env_info]
        info_list = []
        for i,worker_env_info in enumerate(env_info):
            # worker_info_list = [[-1 for _ in range(self.n_out_channels)]]*self.n_agents
            worker_info_list = [-1]*self.n_agents
            if worker_env_info in [None,0] or worker_env_info.get("robot_info", None) is None:
                # If there's no information, take all actions
                info_list.append(worker_info_list)
                continue
            worker_env_info = worker_env_info["robot_info"]
            assert len(worker_env_info)==self.n_agents, "Expected len(env_info)==n_agents, from each worker"
            for j,(pos,cmd) in enumerate(worker_env_info):
                if cmd is None:
                    continue
                dif = (cmd[0]-pos[0], cmd[1]-pos[1])
                # worker_info_list[j] = self._dif_to_flat(dif) if self._dif_within_obs(dif) else [-1 for _ in range(self.n_out_channels)]
                worker_info_list[j] = self._dif_to_flat(dif) if self._dif_within_obs(dif) else -1
            info_list.append(worker_info_list)
        info_list = {"act_info": info_list}
        return info_list
    
    def _flat_to_dif(self, act):
        sz = self.out_shape[1:]
        dif = np.unravel_index(act, sz)
        dif = (dif[0]-sz[0]//2, dif[1]-sz[1]//2)
        return dif
        
    def grid_to_act(self, grid):
        o = self.multiplier_act * grid
        # Sanity check
        o = torch.sum(o, dim=(-1,-2))
        return o

    def dist_grid(self,sz,gamma=.5):
        assert sz%2==1, "Only odd grid sizes supported when converting to action"
        sz = sz+2
        small_sz = ceil(sz/2)
        small_grid = np.full(shape=(small_sz,small_sz), fill_value=small_sz-1)
        for i in range(small_sz-2,-1,-1):
            small_grid[:i+1,:i+1] = i
        large_grid = np.block([[np.flip(small_grid[1:,1:]),np.flipud(small_grid[1:,:])],
                                [np.fliplr(small_grid[:,1:]),small_grid]])
        
        c_grid = gamma**large_grid[1:-1,1:-1]
        u_grid  = gamma**large_grid[1:-1,2:]
        ur_grid = gamma**large_grid[:-2,2:]

        dist_given_act = {
            "0":    torch.Tensor( c_grid ),
            "U":    torch.Tensor( u_grid ),
            "UR":   torch.Tensor( ur_grid ),
            "R":    torch.Tensor( np.rot90(u_grid).copy() ),
            "DR":   torch.Tensor( np.rot90(ur_grid).copy() ),
            "D":    torch.Tensor( np.fliplr(u_grid).copy() ),
            "DL":   torch.Tensor( np.flip(ur_grid).copy() ),
            "L":    torch.Tensor( np.rot90(u_grid, k=-1).copy() ),
            "UL":   torch.Tensor( np.rot90(ur_grid, k=-1).copy() ),
        }
        return dist_given_act

def xy_print(g):
    for y in range(g.shape[1]):
        print("[", end=" ")
        for x in range(g.shape[0]):
            print(g[x,y], end=" ")
        print(" ]")


class CurriculumAgent(CNNAgent):
    pass


if __name__=="__main__":
    
    ag = CNNAgent((6,9,9), 1)
    v = torch.rand(ag.in_shape)
    act = ag.forward(v)

    # for k,v in dist_given_act.items():
    #     print(k)
    #     xy_print(v)
    #     print("")

