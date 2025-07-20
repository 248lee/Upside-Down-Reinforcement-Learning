import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, states, actions, log_probs, rewards, values, next_values, desired_returns, dones, gamma, gae_lambda):
        assert len(states) == len(actions) == len(log_probs) == len(rewards) == len(values) == len(next_values) == len(desired_returns) == len(dones), "All lists must have the same length"
        # We do not accept torch as input, only NumPy arrays
        # Check if all inputs are NumPy arrays
        for arr in [states, actions, log_probs, rewards, values, next_values, desired_returns]:
            if not isinstance(arr[0], np.ndarray):
                raise TypeError(f"Expected NumPy array, got {type(arr)}")
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Convert to NumPy arrays
        self.states = np.array(states)
        self.actions = np.array(actions)
        self.log_probs = np.array(log_probs)
        self.rewards = np.array(rewards)
        self.values = np.array(values)
        self.next_values = np.array(next_values)
        self.desired_returns = np.array(desired_returns)
        self.dones = np.array(dones)

        # Compute advantages (GAE) BEFORE permutation
        self.advantages = np.zeros_like(self.rewards, dtype=np.float32)
        last_gae_lam = 0
        for t in reversed(range(len(self.rewards))):
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + self.gamma * self.next_values[t] * next_non_terminal - self.values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[t] = last_gae_lam

        # Initialize random indices for sampling
        self.indices = np.random.permutation(len(self.rewards))

        # Permute all arrays including advantages
        self.states = self.states[self.indices]
        self.actions = self.actions[self.indices]
        self.log_probs = self.log_probs[self.indices]
        # self.rewards = self.rewards[self.indices]
        self.values = self.values[self.indices]
        # self.next_values = self.next_values[self.indices]
        self.desired_returns = self.desired_returns[self.indices]
        self.dones = self.dones[self.indices]
        self.advantages = self.advantages[self.indices]

        self.start_index = 0
        print("length of rollout buffer:", len(self.rewards))

    def SampleBatch(self, batch_size, device):
        if batch_size > len(self.indices):
            raise ValueError("Batch size cannot be larger than the number of samples in the buffer.")
        
        # Generate batch indices
        batch_indices = np.arange(self.start_index, self.start_index + batch_size) % len(self.indices)
        self.start_index = (self.start_index + batch_size) % len(self.indices)
        
        # Return sampled data
        return {
            'states': torch.tensor(self.states[batch_indices].astype(np.float32)).to(device),
            'actions': torch.tensor(self.actions[batch_indices].astype(np.float32)).to(device),
            'log_probs': torch.tensor(self.log_probs[batch_indices].astype(np.float32)).to(device),
            # 'rewards': torch.tensor(self.rewards[batch_indices].astype(np.float32)).to(device),
            'values': torch.tensor(self.values[batch_indices].astype(np.float32)).to(device),
            # 'next_values': torch.tensor(self.next_values[batch_indices].astype(np.float32)).to(device),
            'desired_returns': torch.tensor(self.desired_returns[batch_indices].astype(np.float32)).to(device),
            'advantages': torch.tensor(self.advantages[batch_indices].astype(np.float32)).to(device)
        }