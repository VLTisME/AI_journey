import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval() # Puts the model in evaluation mode  
    num_val_batches = len(dataloader) # Number of batches in the validation set
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled = amp):
        for batch in tqdm(dataloader, total = num_val_batches, desc = 'Validation round', unit = 'batch', leave = False):
            image, mask_true = batch['image'], batch['mask']
            # image is preprocessed, mask_true is the ground truth mask which values are in [0, n_classes - 1]

            # move images and labels to correct device and type
            image = image.to(device = device, dtype = torch.float32, memory_format = torch.channels_last)
            mask_true = mask_true.to(device = device, dtype = torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1: # for binary segmentation, n_classes = 1 because we output a [single channel mask] which is probability of the pixel being the object (1) or not (0) --> output of UNet in this case is a single channel mask
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float() # converts raw logits to probabilities. If the probability is greater than 0.5, the pixel is considered to be part of the object and set to 1, otherwise it is set to 0
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first = False)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes]'
                # convert to one-hot format
                # A normal tensor for segmentation (e.g., mask_true) doesn’t have a channel dimension (C) because it typically stores class indices for each pixel, not multi-channel data. You basically just need to open the image and see it :D.
                # then you would ask: but if it is (H, W), how can I open it and see the image? It is because you are seeing raw image type .png or .jpg,... then you load the data into data_loading and preprocess the mask :) remember?
                
                # mask_true has shape: (B, H, W). after one-hot encoding: (B, H, W, C). permute to (B, C, H, W)
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim = 1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                # multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first = False) is the dice score of a single batch, which is averaged from all data points in the batch
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first = False)

    net.train() # put the model back in training mode
    return dice_score / max(num_val_batches, 1) # average dice_score over all batches