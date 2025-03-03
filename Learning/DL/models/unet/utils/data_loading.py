# for better understand, visit sider chat unet part 3.
import logging
import torch
import numpy as np
from PIL import Image
from functools import partial # this one freezes some arguments of a function and generate a new function with the remaining arguments
from functools import lru_cache # this one is a decorator that caches the result of the function, so that if the function is called again with the same arguments, the result is returned from the cache instead of being recalculated
from itertools import repeat # this one is used to repeat a value indefinitely
from multiprocessing import Pool # Pool is a way to use multiple CPU cores for parallel processing in Python. It splits a task into smaller chunks, assigns them to multiple processes, and collects the results.

from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename)) # convert a numpy array to a PIL image
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy()) # convert a torch tensor to a PIL image
    else:
        return Image.open(filename) # open an image file, files extension can be .jpg, .png, .bmp, etc.

def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + ".*"))[0]
    # idx and mask_suffix are strings
    # mask_dir is a Path object
    # mask_dir.glob() returns a generator object for example (Path('mask_dir/idx_mask_suffix_1.png'), Path('mask_dir/idx_mask_suffix_2.png'))
    # list(mask_dir.glob()) returns a list of the paths that matches the pattern
    # mask_file is the first element of the list, which is a Path object to the mask picture
    mask = np.asarray(load_image(mask_file))
    # np.asarray is faster than np.array because it doesn't make a copy of the array
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis = 0)
    else:
        raise ValueError(f'Loaded mask must have 2 or 3 dimensions, got {mask.ndim}.')

# Dataset is a base class provided by PyTorch for handling data, designed to work with custom datasets, allowing you to load data samples and their corresponding labels
# If you want to inherit from Dataset, you must implement at least:
# __len__: Returns the size of the dataset (i.e., the total number of samples).
# __getitem__: Retrieves a sample and its label based on an index.
# So that you can use DataLoader from PyTorch to load the data in batches, then shuffle, do whatever you want!

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir) # images_dir is a string but self.images_dir is a Path object
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        
        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        # iterate over all files inside images_dir and check if the current one is file not folder and not a hidden file (which starts with '.'), then it will return the name of the file. For example, when checking image0.png, it returns image0
        if not self.ids:
            raise ValueError(f'No images found in {images_dir}, make sure you pew pew some pictures in here')
        
        logging.info(f'Creating dataset with {len(self.ids)} samples')
        logging.info('Scanning mask files to determine unique classes')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir = self.mask_dir, mask_suffix = self.mask_suffix), self.ids),
                total = len(self.ids)
            ))
            
        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis = 0).tolist()))
        logging.info(f'Unique values found: {self.mask_values}')
    
    def __len__(self):
        return len(self.ids)
    
    @staticmethod # no need to pass self to this method
    #@lru_cache(maxsize = None) # cache the result of the function, so that if the function is called again with the same arguments, the result is returned from the cache instead of being recalculated
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small :('
        pil_img = pil_img.resize((newW, newH), resample = Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)
        # imagine when pil_img is a RGB image, pil_img has size (w, h) but when converting to numpy, it has shape (h, w, 3)
        # if pil_img is a grayscale image, pil_img has size (w, h) and when converting to numpy, it has shape (h, w)
        
        if is_mask:
            mask = np.zeroes((newH, newW), dtype = np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i
            return mask
        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else: # normall input has shape (H, W, C), but we need (C, H, W) for PyTorch
                img = img.transpose((2, 0, 1))
            if img.max() > 1:
                img = img / 255.0
            return img
    
    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + ".*"))
        img_file = list(self.images_dir.glob(name + ".*"))
        
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        
        img = load_image(img_file[0])
        mask = load_image(mask_file[0])
        
        assert img.size == mask.size, f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'
        
        img = self.preprocess(self.mask_values, img, self.scale, is_mask = False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask = True)
        
        return{
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }
        
class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale = 1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix = '_mask') # __init__ has arguments inside () because it inherits from BasicDataset, and __init__ from BasicDataset expects these arguments
        