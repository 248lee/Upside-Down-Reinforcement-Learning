import torch
from pgudrl.udmodel import BF
from torch.nn import functional as F

def CalculateLoss(bf: BF, batch_of_data, clip_range):
    # Check that the batch has the following keys: states, actions, log_probs, advantages, values, desired_returns
    required_keys = ['states', 'actions', 'log_probs', 'advantages', 'values', 'desired_returns']
    for key in required_keys:
        if key not in batch_of_data:
            raise ValueError(f"Batch of data must contain the key: {key}")
        
    # Time for forwarding! for both actor and critic
    log_probs, state_values = bf.forward_to_get_logprob_and_value(
        batch_of_data['states'],
        batch_of_data['desired_returns'],
        batch_of_data['actions']
    )

    # Calculate the actor loss
    
    # Normalize the advantages
    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
    with torch.no_grad():
        advantages = batch_of_data['advantages']
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Calculate ratio between old and new policy, should be one at the first call
    ratio = torch.exp(log_probs - batch_of_data['log_probs'])

    # clipped surrogate loss
    with torch.no_grad():
        upside_down = (batch_of_data['values'] - batch_of_data['desired_returns'])
    policy_loss1 = upside_down * advantages * ratio
    policy_loss2 = upside_down * advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = torch.max(policy_loss1, policy_loss2).mean()

    # Calculate the critic loss
    with torch.no_grad():
        target = batch_of_data['advantages'] + batch_of_data['values']
    value_loss = F.mse_loss(state_values, target)

    return policy_loss, value_loss


        
    