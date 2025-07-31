import numpy as np
import torch

class RolloutBuffer:
    def __init__(self, rollout, gamma):
        states = rollout["observations"]
        next_states = rollout["next_observations"]
        rewards = rollout["rewards"]
        dones = rollout["dones"]
        
        assert len(states) == len(rewards) == len(dones) == len(next_states), "All lists must have the same length"
        # We do not accept torch as input, only NumPy arrays
        # Check if all inputs are NumPy arrays
        for arr in [states, rewards, next_states]:
            if not isinstance(arr[0], np.ndarray):
                raise TypeError(f"Expected NumPy array, got {type(arr[0])}")
        self.gamma = gamma

        # Convert to NumPy arrays
        self.states = np.array(states)
        self.rewards = np.array(rewards)
        self.dones = np.array(dones)
        self.next_states = np.array(next_states)

        # Initialize random indices for sampling
        self.indices = np.random.permutation(len(self.rewards))

        # Permute all arrays including advantages
        self.states = self.states[self.indices]
        self.rewards = self.rewards[self.indices]
        self.dones = self.dones[self.indices]
        self.next_states = self.next_states[self.indices]

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
            'rewards': torch.tensor(self.rewards[batch_indices].astype(np.float32)).to(device),
            'dones': torch.tensor(self.dones[batch_indices].astype(np.float32)).to(device),
            'next_states': torch.tensor(self.next_states[batch_indices].astype(np.float32)).to(device)
        }
    
    def SampleBatchGivenIndices(self, batch_indices, device):
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