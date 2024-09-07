import numpy as np 
import nibabel as nib
import os
from tensorflow.keras.utils import to_categorical 
from sklearn.preprocessing import MinMaxScaler 



def filter_based_on_masks(subjects_list):
    included_subjects = list()
    for sub in subjects_list:
        for filename in os.listdir(sub):
            if "seg" in filename or "Seg" in filename : # due to the misnaming that exists in the dataset 
                curr_mask_path = os.path.join(sub, filename) 
            else:
                continue
        curr_mask = nib.load(curr_mask_path).get_fdata().astype(np.uint8)
        curr_mask[curr_mask==4] = 3 # label 3 is missing, thus we replace 4 with 3 for the sake of continuity and one-hot encoding
        curr_mask = curr_mask[56:(240-56),56:(240-56),13:(155-14)]
        
        vals, counts = np.unique(curr_mask, return_counts=True)
        
        if (1 - (counts[0]/counts.sum())) > 0.01: # if more than 1% of the data is useful, so keep it 
            included_subjects.append(sub)
    return included_subjects 


def tranform_nii_to_npy(subject_volumes_paths, subject_mask_path):
    curr_subject = list()
    for i in range(len(subject_volumes_paths)):
        scalar =  MinMaxScaler()
        curr_volume = nib.load(subject_volumes_paths[i]).get_fdata().astype(np.float32)
        rescaled_volume = scalar.fit_transform(curr_volume.reshape(-1,curr_volume.shape[-1])).reshape(curr_volume.shape)
        curr_subject.append(rescaled_volume)
    curr_mask = nib.load(subject_mask_path).get_fdata().astype(np.uint8)
    curr_mask[curr_mask == 4] = 3 # label 3 is missing, thus we replace 4 with 3 for the sake of continuity and one-hot encoding

    combined_subject_volumes = np.stack([curr_subject[0],curr_subject[1],curr_subject[2],curr_subject[3]], axis=3)

    # cropping to save computational resource and accelerate it 
    cropped_subject_volumes = combined_subject_volumes[56:(240-56),56:(240-56),13:(155-14)] # current shape: (128, 128, 128, 4)
    curr_mask = curr_mask[56:(240-56),56:(240-56),13:(155-14)]
    encoded_mask = to_categorical(curr_mask, num_classes=4)

    return (cropped_subject_volumes, encoded_mask)