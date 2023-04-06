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

        self.net, self.out_shape = net_from_string(args.agent_arch, self.in_shape, 
                                                   target_shape=(args.n_actions,) if not args.action_grid else "same")
        self.net = self.net.to(self.device)

        # self.dist_given_act = self.dist_grid(self.out_shape[-1], gamma=.5)
        # self.multiplier_act = torch.cat([x.unsqueeze(0) for x in self.dist_given_act.values()],0).to(self.device)

    def forward(self, input, hidden_state=None, env_info=None):
        # if len(input.shape)==3:
        if input.dim() == 3:
            input = input.unsqueeze(0)
        elif input.dim()<3:
            raise ValueError("Not enough dimensions in input")

        v = self.net(input)
        self.intention = v
        try:
            v = torch.flatten(v, start_dim=-3)
        except:
            print(v)

        v, target_update = self.target_update(v, env_info)
        return v, torch.Tensor([0]), target_update

        act_prob = self.grid_to_act(v)
        # act = torch.argmax(act_prob)
        return act_prob, torch.Tensor([0]) # returns hidden state

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
        # Decides whether to update the controller's target or not
        L = v.shape[0]
        v = v.view(-1,self.n_agents,self.n_actions)
        bs = v.shape[0]
        target_update = torch.ones(bs, self.n_agents, 1, dtype=torch.uint8)
        if not isinstance(env_info, list):
            env_info = [env_info]

        for i,worker_env_info in enumerate(env_info):
            if worker_env_info is None or worker_env_info.get("robot_info", None) is None:
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
    

        if env_info is None or env_info.get("robot_info", None) is None:
            # If there's no information, take all actions
            return v, torch.ones((L,1), dtype=torch.uint8)
        if not isinstance(env_info, list):
            env_info = [env_info]

        env_info = env_info.get("robot_info", None)
        
        assert L==len(env_info), "Length of env_info doesn't match"
        target_update = torch.zeros((L,1), dtype=torch.uint8) # 1 if a new target is set, 0 otherwise
        for i,(pos,cmd) in enumerate(env_info):
            if cmd is None:
                target_update[i] = 1
                continue
            dif = (cmd[0]-pos[0], cmd[1]-pos[1])
            v[i, :], target_update[i] = self.target_update_policy(v[i, :], dif)
            # Sanity check: check if chosen action corresponds to existing cmd
            if not target_update[i]:
                act = torch.argmax(v[i,:])
                new_dif = self._flat_to_dif(act)
                assert new_dif==dif, f"Unexpected behavior, dif: {dif}, new dif: {new_dif}"

        return v, target_update

    def target_update_policy(self, actions, current_dif):
        if self.current_target_factor is not None:
            current_dif = self._clip_to_obs_range(current_dif)
            actions[self._dif_to_flat(current_dif)] += np.log(self.current_target_factor)
            # actions[self._dif_to_flat(current_dif)] += 10
            return actions, 1
        # TEMPORARY: never reevaluate a target
        if self._dif_within_obs(current_dif):
            current_dif = self._clip_to_obs_range(current_dif)
            actions = -1e10*torch.ones_like(actions)
            actions[self._dif_to_flat(current_dif)] = 1e10
            return actions, 0
        else:
            return actions, 1

    # def target_update_policy(self, actions, current_dif):
    #     # TEMPORARY: never reevaluate a target
    #     if self._dif_within_obs(current_dif):
    #         current_dif = self._clip_to_obs_range(current_dif)
    #         actions = -1e10*torch.ones_like(actions)
    #         actions[self._dif_to_flat(current_dif)] = 1e10
    #         return actions, 0
    #     else:
    #         return actions, 1

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
            pass
    
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

