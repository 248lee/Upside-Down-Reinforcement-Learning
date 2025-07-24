import torch
from torch.nn import functional as F
from pgudrl.optimisticvaluenetwork import OptimisticValueNetwork

def CalculateLoss(ovn: OptimisticValueNetwork, batch_of_data):
    required_keys = ['states', 'opt_targets']
    for key in required_keys:
        if key not in batch_of_data:
            raise ValueError(f"Batch of data must contain the key: {key}")
        
    return torch.mean((torch.clamp(batch_of_data['opt_targets'] - ovn(batch_of_data['states']), min=0.0, max=None))**2)