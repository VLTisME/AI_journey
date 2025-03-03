import torch
from torch import Tensor


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # reduce_batch_first = True only if your dimensions >= 3 and you want to reduce the batch dimension first. When it's true, the Dice is calculated over the batch, it treats the batch as one big tensor when calculating intersection and union
    # when reduce_batch_first = False, it will calculate the Dice for each mask in the batch and average them
    # Input is a segmentation mask, target is the ground truth mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first # either the input must be shape (B, H, W) so that can use reduce_batch_first or if it is 2D, then we must ensure no reduce_batch

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim = sum_dim) # oh, so it works with both cases when input contains hard classes (like 0, 1, 2,...) and soft classes (like probabilities)
    sets_sum = input.sum(dim = sum_dim) + target.sum(dim=sum_dim)
    # sets_sum is a scalar if input dim = 3 and reduce_batch_first = True, or a scalar when input dim == 2 (and obviously reduce_batch_first = False)
    # sets_sum is a tensor of shape (B,) if input dim = 3 and reduce_batch_first = False
    # sets_sum has the same shap as inter so the below is explainable. Imagine when sets_sum is a scalar and if it is equal to 0 (|A| + |B| = 0) then it is set to inter (which must be = 0 in this case)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum) # torch.where considers each element of the condition tensor, and if that elements is True, it uses the corresponding element from the first tensor, otherwise it uses the corresponding element from the second tensor

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean() # if it has shape (B,) then it will return a scalar which is averaged, if it has shape () then it will also return a scalar
    # wrong: .mean() does not convert to a Python float; instead, it returns a tensor with shape torch.Size([]). that's why they use loss.item()

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)