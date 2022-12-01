import torch
import torch.nn as nn

class CustomAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CustomAgent, self).__init__()

    def forward(self, input):
        raise NotImplementedError("Custom agent not implemented")