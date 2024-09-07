import os 
import numpy as np
from Processing_tools import tranform_nii_to_npy


def load_images(subset_subjects_paths):
    images = list()
    masks = list()
    for sub in subset_subjects_paths:
        curr_volumes_paths = list() 
        for filename in os.listdir(sub):
            if "seg" in filename or "Seg" in filename : # due to the misnaming that exists in the dataset 
                curr_mask_path = os.path.join(sub, filename)    
            else:
                whole_path = os.path.join(sub, filename)
                curr_volumes_paths.append(whole_path)
        
        image, mask = tranform_nii_to_npy(curr_volumes_paths, curr_mask_path)
        images.append(image)
        masks.append(mask)
    
    images = np.array(images, dtype = np.float32)
    masks = np.array(masks, dtype = np.float32)
    return (images, masks) 


def imageLoader(subjects_list_paths, batch_size):
    L = len(subjects_list_paths)

    while True:
        batch_start = 0 
        batch_end = batch_size

        while batch_start< L:
            upper_limit = min(batch_end, L)

            X, Y = load_images(subjects_list_paths[batch_start:upper_limit])
            
            yield (X, Y) # batches generated on the fly  (for defining generator)

            batch_start += batch_size
            batch_end += batch_size
