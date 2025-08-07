import torch.nn as nn
import torch
import numpy as np
from torch.distributions import Categorical

class BF(nn.Module):  # "actor" is an alias of policy, and "critic" is an alias of q-value function
    def __init__(self, state_space, action_space, hidden_size, return_scale, gamma, seed, device):
        super(BF, self).__init__()
        torch.manual_seed(seed)
        self.return_scale = return_scale
        self.device = device
        self.actions = np.arange(action_space)
        self.action_space = action_space
        self.fc1 = nn.Linear(state_space, hidden_size)
        self.commands1 = nn.Linear(1, hidden_size)  # Here, we assume 1 command: desired return
        self.commands2 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)

        self.actor_fc4 = nn.Linear(hidden_size, hidden_size)
        self.actor_fc5 = nn.Linear(hidden_size, action_space)
        
        self.critic_fc4 = nn.Linear(hidden_size, hidden_size)
        self.critic_fc5 = nn.Linear(hidden_size, action_space)

        self.sigmoid = nn.Sigmoid()

        self.gamma = gamma
        
    def forward(self, state, command):       
               
        out = torch.relu(self.fc1(state))
        command = torch.relu(self.commands1(command))
        command_out = self.sigmoid(self.commands2(command))
        out = out * command_out
        out = torch.relu(self.fc2(out))
        feature = torch.relu(self.fc3(out))

        # Actor
        actor_out = torch.relu(self.actor_fc4(feature))
        actor_out = self.actor_fc5(actor_out)

        # Critic
        critic_out = torch.relu(self.critic_fc4(feature))
        critic_out = self.critic_fc5(critic_out)
        
        return actor_out, critic_out
    
    def action(self, state, desire):
        """
        Samples the action based on their probability
        """
        command = (desire*self.return_scale)
        action_prob, q_value = self.forward(state, command)
        probs = torch.softmax(action_prob, dim=-1)
        m = Categorical(probs)
        action = m.sample()
        return action, m.log_prob(action), q_value
    
    def forward_to_get_logprob_and_q_value(self, state, desire, action):
        """
        Returns the log probability of the action and the state value
        """
        command = (desire*self.return_scale)
        action_prob, q_value = self.forward(state, command)
        probs = torch.softmax(action_prob, dim=-1)
        m = Categorical(probs)
        log_prob = m.log_prob(action)
        return log_prob, q_value[action]
    
    def greedy_action(self, state, desire):
        """
        Returns the greedy action 
        """
        command = (desire*self.return_scale).unsqueeze(0)  # Ensure command is a 2D tensor
        action_prob, _ = self.forward(state, command)
        probs = torch.softmax(action_prob, dim=-1)
        action = torch.argmax(probs).item()
        return action