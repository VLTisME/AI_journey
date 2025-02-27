def unique_mask_values(idx, mask_dir, mask_suffix):  
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]  
    mask = np.asarray(load_image(mask_file))  
    if mask.ndim == 2:  
        return np.unique(mask)  
    elif mask.ndim == 3:  
        mask = mask.reshape(-1, mask.shape[-1])  
        return np.unique(mask, axis=0)  
    else:  
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')