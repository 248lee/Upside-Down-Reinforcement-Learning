import torch.nn as nn
import torch
import numpy as np

class OptimisticValueNetwork(nn.Module):
    def __init__(self, state_space, action_space, hidden_size, gamma, seed, device):
        super(OptimisticValueNetwork, self).__init__()
        torch.manual_seed(seed)
        self.device = device
        self.actions = np.arange(action_space)
        self.action_space = action_space
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3_optimistic = nn.Linear(hidden_size, 1)
        self.fc3_pessimistic = nn.Linear(hidden_size, 1)

        self.gamma = gamma

        
    def forward(self, state):       
               
        out = torch.relu(self.fc1(state))
        out = torch.relu(self.fc2(out))
        optimistic_out = self.fc3_optimistic(out)
        pessimistic_out = self.fc3_pessimistic(out)

        return optimistic_out, pessimistic_out
    