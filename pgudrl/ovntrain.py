import torch
from torch.nn import functional as F
from pgudrl.optimisticvaluenetwork import OptimisticValueNetwork

def CalculateLoss(ovn: OptimisticValueNetwork, batch_of_data):
    required_keys = ['states', 'opt_targets']
    for key in required_keys:
        if key not in batch_of_data:
            raise ValueError(f"Batch of data must contain the key: {key}")
    
    delta_value = batch_of_data['opt_targets'] - ovn(batch_of_data['states'])
    # If an element of delta_value is negative, multiply it by 0.5
    delta_value = torch.where(delta_value < 0, delta_value * 0.7, delta_value)
    # Reduce to scalar for backward()
    return (delta_value**2).mean()