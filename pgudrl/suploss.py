from pgudrl.udmodel import BF
import torch
import torch.nn.functional as F

def CalculateLoss(bf: BF, input_states: list[torch.Tensor], input_commands: list[torch.Tensor], output_actions: list[torch.Tensor]):
    """
    Trains the BF with on a cross entropy loss were the inputs are the action probabilities based on the state and command.
    The targets are the actions appropriate to the states from the replay buffer.
    """
    state = torch.stack(input_states)
    input_commands = torch.stack(input_commands)
    command = input_commands * bf.return_scale
    y = torch.stack(output_actions).long().squeeze(-1)  # Convert y to be a 1D tensor
    y_ = bf(state.to(bf.device), command.to(bf.device))[0].float()  # Get only the action probabilities (the first output of the BF)
    pred_loss = F.cross_entropy(y_, y)
    return pred_loss