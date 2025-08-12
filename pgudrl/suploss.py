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
    command = input_commands# * bf.return_scale
    y_, q_ = bf(state.to(bf.device), command.to(bf.device))
    y_ = y_.float()  # Get only the action probabilities (the first output of the BF)
    y = batch.actions.detach().clone().long()#.squeeze(-1)  # Convert y to be a 1D tensor
    
    # counts = torch.tensor([19980, 20230, 40656, 19134], dtype=torch.float32)  # from your matrix
    # class_weights = (counts.sum() / counts).to(bf.device)   # inverse frequency
    pred_loss = F.cross_entropy(y_, y)

    # Exploration loss for the Behavior Function
    ovn_tmp, pes_tmp = ovn(state)
    with torch.no_grad():
        desired_return_to_go = ovn_tmp.detach().clone()  # Get the desired return to go from the optimistic value network
        command_exploratory = desired_return_to_go# * bf.return_scale
    
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
        next_desired_return_to_go, _ = ovn(next_states.to(bf.device))
        next_command = next_desired_return_to_go# * bf.return_scale
        next_y, next_q_values = bf(next_states.to(bf.device), next_command.to(bf.device))
        next_policy = torch.softmax(next_y.float(), dim=-1)
        # This is the expected-SARSA
        next_value = torch.sum(next_q_values * next_policy, dim=1, keepdim=True)  # Dot product of the next q-values and the next policy
        targets = batch.rewards + bf.gamma * next_value * (1 - batch.dones)
    q_values_selected = torch.gather(q_tmp, 1, batch.actions.to(torch.int64).unsqueeze(1))
    value_loss_explore = F.mse_loss(q_values_selected, targets.detach())

    with torch.no_grad():  # although old q values are not used in the training process, it is useful in the testing process
        next_states = batch.next_observations.detach().clone()
        next_desired_return_to_go = (batch.return_to_goes- batch.rewards) / bf.gamma
        next_command = next_desired_return_to_go# * bf.return_scale
        next_y, next_q_values = bf(next_states.to(bf.device), next_command.to(bf.device))
        next_policy = torch.softmax(next_y.float(), dim=-1)
        # This is the expected-SARSA
        next_value = torch.sum(next_q_values * next_policy, dim=1, keepdim=True)  # Dot product of the next q-values and the next policy
        targets = batch.rewards + bf.gamma * next_value * (1 - batch.dones)
    q_values_selected = torch.gather(q_, 1, batch.actions.to(torch.int64).unsqueeze(1))
    value_loss_recall = F.mse_loss(q_values_selected, targets.detach())

    value_loss = (value_loss_explore + value_loss_recall) / 2

    # Optimistic value network loss
    with torch.no_grad():
        buffer_return = batch.return_to_goes.detach().clone()
        max_q = torch.max(q_tmp, dim=1, keepdim=True)[0]
        ovn_target = torch.maximum(buffer_return, max_q)
    delta_value_opt = ovn_target - ovn_tmp
    delta_value_squared_opt = torch.where(delta_value_opt < 0, delta_value_opt**2, 12 * delta_value_opt**2)
    delta_value_pes = buffer_return - pes_tmp  # although pessimistic values are not used in the training process, it is useful in the testing process
    delta_value_squared_pes = torch.where(delta_value_pes < 0, 15 * delta_value_pes**2, delta_value_pes**2)
    ovn_loss = delta_value_squared_opt.mean() + delta_value_squared_pes.mean()
    
    return pred_loss, exploration_loss, value_loss, ovn_loss

def test_accuracy(bf: BF, batch: ReplayBufferSamples):
    with torch.no_grad():
        state = batch.observations.detach().clone()
        input_commands = batch.return_to_goes.detach().clone()
        command = input_commands# * bf.return_scale
        y_ = bf(state.to(bf.device), command.to(bf.device))[0].float()  # Get only the action probabilities (the first output of the BF)
        y = batch.actions.detach().clone().long()#.squeeze(-1)  # Convert y to be a 1D tensor
        greedy_action = torch.argmax(y_, dim=1)
        is_correct = torch.where(greedy_action == y, torch.ones_like(greedy_action), torch.zeros_like(greedy_action))
        accuracy = torch.sum(is_correct).item() / is_correct.shape[0]

        return accuracy, greedy_action, y