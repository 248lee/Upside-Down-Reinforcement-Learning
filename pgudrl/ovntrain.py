import torch
from torch.nn import functional as F
from pgudrl.optimisticvaluenetwork import OptimisticValueNetwork

def CalculateLoss(ovn: OptimisticValueNetwork, batch_of_data, gamma, is_optimistic=True):
    required_keys = ['states', 'rewards', 'dones', 'next_states']
    for key in required_keys:
        if key not in batch_of_data:
            raise ValueError(f"Batch of data must contain the key: {key}")
    
    with torch.no_grad():
        next_ovn_values = ovn(batch_of_data['next_states'])
        ovn_target = batch_of_data['rewards'] + gamma * next_ovn_values * (1 - batch_of_data['dones'])
    delta_value = ovn_target - ovn(batch_of_data['states'])
    # If an element of delta_value is negative, multiply it by 0.9
    if is_optimistic:
        delta_value_squared = torch.where(delta_value < 0, 0.5 * delta_value**2, delta_value**2)
    else:
        delta_value_squared = delta_value**2

    # Warn if any value is nan, or larger than 1e6
    if torch.any(torch.isnan(delta_value_squared)):
        raise ValueError("NaN values found in delta_value_squared")
    if torch.any(torch.abs(delta_value_squared) > 1e6):
        max_val = torch.max(torch.abs(delta_value_squared))
        print(f"⚠️  Warning: delta_value_squared contains values as large as {max_val:.2e} (>|1e6|).")
    # Reduce to scalar for backward()
    return delta_value_squared.mean()