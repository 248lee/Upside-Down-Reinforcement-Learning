from pgudrl.udmodel import BF
import torch
import torch.nn.functional as F
from pgudrl.optimisticvaluenetwork import OptimisticValueNetwork
from pgudrl.replaybuffer import ReplayBufferSamples

def CalculateLoss(bf: BF, ovn: OptimisticValueNetwork, batch: ReplayBufferSamples):
    """
    Trains the BF with on a cross entropy loss were the inputs are the action probabilities based on the state and command.
    The targets are the actions appropriate to the states from the replay buffer.
    """
    if bf.gamma != ovn.gamma:
        print("Are you kidding me...")
        input()
    # Supervised loss for the Behavior Function
    state = batch.observations.detach().clone()
    input_commands = batch.return_to_goes.detach().clone()
    command = input_commands * bf.return_scale
    y_ = bf(state.to(bf.device), command.to(bf.device))[0].float()  # Get only the action probabilities (the first output of the BF)
    y = batch.actions.detach().clone().long()#.squeeze(-1)  # Convert y to be a 1D tensor
    pred_loss = F.cross_entropy(y_, y)

    # Exploration loss for the Behavior Function
    with torch.no_grad():
        desired_return_to_go = ovn(state)
        command_exploratory = desired_return_to_go * bf.return_scale
    
    y_tmp, q_tmp = bf(state.to(bf.device), command_exploratory.to(bf.device))
    policy_tmp = torch.softmax(y_tmp.float(), dim=-1)
    
    with torch.no_grad():
        value_tmp = torch.sum(q_tmp * policy_tmp, dim=1, keepdim=True)  # Dot product of the q-values and the policy
        a = (desired_return_to_go - value_tmp) * (q_tmp - value_tmp )
        support_tmp = torch.where(a > 0, torch.ones_like(a), torch.zeros_like(a))
        support_tmp = support_tmp / (torch.sum(support_tmp, dim=1, keepdim=True) + 1e-10) # Normalize the support
    log_policy_tmp = torch.log(policy_tmp + 1e-10)  # Add a small value to avoid log(0)
    exploration_losses = -torch.sum(support_tmp * log_policy_tmp, dim=1, keepdim=True)  # Cross entropy loss

    if torch.any(torch.isnan(exploration_losses)):
        raise ValueError("NaN values found in exploration_losses")
    if torch.any(torch.abs(exploration_losses) > 1e6):
        max_val = torch.max(torch.abs(exploration_losses))
        print(f"⚠️  Warning: exploration_losses contains values as large as {max_val:.2e} (>|1e6|).")

    exploration_loss = torch.mean(exploration_losses)

    # Value loss for the critic function
    with torch.no_grad():
        next_states = batch.next_observations.detach().clone()
        next_desired_return_to_go = ovn(next_states.to(bf.device))
        next_command = next_desired_return_to_go * bf.return_scale
        next_y, next_q_values = bf(next_states.to(bf.device), next_command.to(bf.device))
        next_policy = torch.softmax(next_y.float(), dim=-1)
        # This is the expected-SARSA
        next_value = torch.sum(next_q_values * next_policy, dim=1, keepdim=True)  # Dot product of the next q-values and the next policy
        targets = batch.rewards + bf.gamma * next_value * (1 - batch.dones)
    q_values_selected = torch.gather(q_tmp, 1, batch.actions.to(torch.int64).unsqueeze(1))
    value_loss = F.mse_loss(q_values_selected, targets.detach())
    
    return pred_loss, exploration_loss, value_loss