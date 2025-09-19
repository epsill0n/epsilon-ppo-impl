import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBase(nn.Module):
    def __init__(self, in_channels, hidden_dim=512):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc = nn.Linear(7 * 7 * 64, hidden_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        return F.relu(self.fc(x))

class CNNPolicy(nn.Module):
    def __init__(self, obs_shape, hidden_dim, act_dim):
        super().__init__()
        self.base = CNNBase(obs_shape[0], hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, act_dim)
        
    def forward(self, x):
        z = self.base(x)
        return self.fc_out(z)

class CNNValue(nn.Module):
    def __init__(self, obs_shape, hidden_dim):
        super().__init__()
        self.base = CNNBase(obs_shape[0], hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        z = self.base(x)
        return self.fc_out(z)

class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        super().__init__()
        self.actor = actor
        self.critic = critic
        
    def forward(self, state):
        action_pred = self.actor(state)
        value_pred = self.critic(state)
        return action_pred, value_pred