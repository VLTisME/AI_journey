import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet.unet_network import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss



# Check if we're running in Kaggle  
is_kaggle_env = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ  


# Set dataset path dynamically  
if is_kaggle_env:  
    # For Kaggle  
    dataset_path = Path('/kaggle/input/carvana/data/') 
    dir_checkpoint = Path('/kaggle/working/checkpoints/')   # Save checkpoints here (writable) 
else:  
    # For local environment  
    dataset_path = Path(r'D:/AI/Learning/DL/models/unet/data')  # <- Point this to your local dataset folder
    dir_checkpoint = Path('./checkpoints/')  # Save checkpoints locally  
# The . in the paths ./data/train/ and ./data/train_masks/ refers to the current working directory in your filesystem.
# ./... → Relative path starting from the current directory.
# /... → Absolute path starting from the root of the filesystem.
dir_img = dataset_path / 'train/train'
dir_mask = dataset_path / 'train_masks/train_masks'
# Ensure checkpoint directory exists on Kaggle
dir_checkpoint.mkdir(parents = True, exist_ok = True) 
# ok so if i train locally, dir_checkpoint will be ./checkpoints/ and if i train on kaggle, dir_checkpoint will be /kaggle/working/checkpoints/
# when pushing to github, i should ignore the checkpoints folder because it's too big, and it's so ok because when training on kaggle, it will create the checkpoints folder automatically

# dir_checkpoint to save the model checkpoints like checkpoint_epoch1.pth, checkpoint_epoch2.pth, or checkpoints in unet_network.py, or info about anything like:
# torch.save({  
#     'epoch': epoch,  
#     'model_state_dict': model.state_dict(),  
#     'optimizer_state_dict': optimizer.state_dict(),  
#     'scheduler_state_dict': scheduler.state_dict(),  
#     'loss': loss,  
# }, dir_checkpoint / f'checkpoint_epoch_{epoch}.pth')  

# u can even load, save, debug, deploy, and resume the model from the checkpoint
# load:
# checkpoint = torch.load('./checkpoints/checkpoint_epoch_10.pth')  
# model.load_state_dict(checkpoint['model_state_dict'])  
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  
# start_epoch = checkpoint['epoch'] + 1  
# save best model:
# if val_loss < best_val_loss:  
#     torch.save(model.state_dict(), './checkpoints/best_model.pth')  
#     best_val_loss = val_loss  

# After training, you can use the checkpointed model for inference or deployment. For example, you would load the model weights for evaluation:
# model.load_state_dict(torch.load('./checkpoints/best_model.pth'))  
# model.eval()  

def train_model(
        model,
        device,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-5,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale)
    # when calling len(dataset) or dataset[i], it will call the __len__ and __getitem__ methods of the BasicDataset class

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator = torch.Generator().manual_seed(0))
    # train_set and val_set are not instances of BasicDataset, they are instances of torch.utils.data.Subset
    # magic part:
    # though train_set and val_set have restricted access to the dataset (because they are instances of torch.utils.data.Subset), they still have access to the dataset's __getitem__ and __len__ methods
    # so when calling len(train_set) or train_set[i], it will call the __len__ and __getitem__ methods of the BasicDataset class, which means it calls the __len__ and __getitem__ methods of the 'dataset'
    

    # 3. Create data loaders
    loader_args = dict(batch_size = batch_size, num_workers = os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle = True, **loader_args)
    val_loader = DataLoader(val_set, shuffle = False, drop_last = True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project = 'UNet', resume = 'allow', anonymous = 'must')
    experiment.config.update(
        dict(epochs = epochs, batch_size = batch_size, learning_rate = learning_rate,
             val_percent = val_percent, save_checkpoint = save_checkpoint, img_scale = img_scale, amp = amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(model.parameters(),
                              lr = learning_rate, weight_decay = weight_decay, momentum = momentum, foreach = True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience = 5)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled = amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader: # already separated into batches
                images, true_masks = batch['image'], batch['mask']
                # shape of images: [batch_size, n_channels, height, width]
                # shape of true_masks: [batch_size, height, width]
                # input shape must have 3 channels (RGB)
                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    # masks_pred has shape: [batch_size, n_classes, height, width]  
                    if model.n_classes == 1:
                        # squeeze(1) = remove the second dimension -> [batch_size, 1, height, width] -> [batch_size, height, width]
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        # combine the two losses to resolve each loss's shortcomings
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            # In the code, dice_coeff is adapted to work with probabilities instead of discrete binary masks. Here's how it works:
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True) # clear gradients, ensures no leftover gradients from the previous step interfere with the current backpropagation.
                grad_scaler.scale(loss).backward() # part of the AMP API, scales the loss value to prevent underflow or overflow in the gradients (16 and 32 bit floating point precision)
                grad_scaler.unscale_(optimizer) # After calculating the gradients, this unscales the gradients (divides them by the same scaling factor applied earlier).
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping) #  Limits the magnitude of the gradients to prevent exploding gradients (which can cause instability, particularly in deep networks or when using large learning rates).
                grad_scaler.step(optimizer) # calls the optimizer's .step() function, which updates the model's parameters.
                grad_scaler.update() # This updates the loss scaling factor used by GradScaler.

                pbar.update(images.shape[0])
                global_step += 1 # A step is one forward and backward pass through a batch of data.
                epoch_loss += loss.item() # loss is a tensor maybe,... because 
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()}) # update the progress bar with additional information about the current loss value of current batch

                # Evaluation round
                division_step = (n_train // (2 * batch_size)) # meaning do the evaluation every 20% of each epoch
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load: # if the model is loaded from a checkpoint
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        # if it runs out of memory, it will enable gradient checkpointing to reduce memory usage, then train like above
        # otherwise just train like above
        torch.cuda.empty_cache() # Frees GPU memory.
        model.use_checkpointing() # Enables gradient checkpointing. 
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )