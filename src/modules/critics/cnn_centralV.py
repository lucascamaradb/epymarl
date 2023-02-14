import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CNNCentralVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(CNNCentralVCritic, self).__init__()

        assert not args.obs_individual_obs, "obs_individual_obs not supported in CNN critic"
        assert not args.obs_last_action, "obs_last_action not supported in CNN critic"

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.input_shape = self._get_input_shape(scheme)
        self.in_channels = self.input_shape[0]
        self.output_type = "v"

        # Set up network layers
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, 16, 3, 1, 0),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, 1, 0),
            # nn.AvgPool2d(5,stride=1), # 5 instead of 3
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, batch, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)
        # x = F.relu(self.fc1(inputs))
        # x = F.relu(self.fc2(x))
        # q = self.fc3(x)
        # return q
        inputs -= .5 ##########################
        assert not th.isnan(inputs).any(), "NaN in input"
        orig_shape = inputs.shape[:-3]
        if len(inputs.shape)>4:
            inputs = inputs.view(-1,*self.input_shape)
            out = th.sum(self.net(inputs), (-1,-2,-3))
            out = out.view(*orig_shape, 1)
        elif len(inputs.shape)<3:
            raise ValueError(f"Unexpected input dimension: {inputs.shape}")
        else:
            out = th.sum(self.net(inputs), (-1,-2,-3))
        
        # Sanity check
        if th.isnan(out).any():
            raise ValueError("NaN in output")

        return out

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        repeat_arg = [1]*(len(batch["state"][:, ts].shape)+1)
        repeat_arg[2] = self.n_agents
        inputs = batch["state"][:, ts].unsqueeze(2).repeat(repeat_arg)

        # inputs = []
        # # state
        # inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))
        # inputs = th.cat(inputs, dim=-1)

        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        return input_shape
        ######################################################################################
        # observations
        if self.args.obs_individual_obs:
            input_shape += scheme["obs"]["vshape"] * self.n_agents
        # last actions
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        input_shape += self.n_agents
        return input_shape