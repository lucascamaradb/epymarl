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
        out_channels = 1 ###########
        kernel_size = 1 ###########

        self.net, self.out_shape = net_from_string(args.agent_arch, self.in_shape, target_shape=(args.n_actions,))
        self.net = self.net.to(self.device)

        # self.dist_given_act = self.dist_grid(self.out_shape[-1], gamma=.5)
        # self.multiplier_act = torch.cat([x.unsqueeze(0) for x in self.dist_given_act.values()],0).to(self.device)

    def forward(self, input, hidden_state=None):
        # if len(input.shape)==3:
        if input.dim() == 3:
            input = input.unsqueeze(0)
        elif input.dim()<3:
            raise ValueError("Not enough dimensions in input")

        v = self.net(input)
        return v, torch.Tensor([0])

        act_prob = self.grid_to_act(v)
        # act = torch.argmax(act_prob)
        return act_prob, torch.Tensor([0]) # returns hidden state
        # raise NotImplementedError("Custom agent not implemented")

    def init_hidden(self):
        return torch.Tensor([0])

    def _get_output_shape(self):
        v = torch.randn(self.in_shape).unsqueeze(0).to(self.device)
        return self.net(v)[0].shape

        # v = torch.randn(self.in_shape)
        # v, _ = self.forward(v)
        # return v.shape

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

if __name__=="__main__":
    
    ag = CNNAgent((6,9,9), 1)
    v = torch.rand(ag.in_shape)
    act = ag.forward(v)

    # for k,v in dist_given_act.items():
    #     print(k)
    #     xy_print(v)
    #     print("")

