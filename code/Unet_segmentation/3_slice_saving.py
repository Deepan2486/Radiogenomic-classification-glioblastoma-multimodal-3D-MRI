from sklearn.preprocessing import MinMaxScaler
import numpy as np
import nibabel as nib
import glob
import os
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

TRAIN_DATASET_PATH = '/content/drive/MyDrive/Colab Notebooks/data/'

t2_npy_list = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data/3_slices_group/t2/t2_train_images/*'))
flair_npy_list = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data/3_slices_group/flair/*'))
mask_npy_list = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data/3_slices_group/t1/mask_single/*'))


t2_val_npy_list = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data/3_slices_group/train_test_split/t2/val/images/*'))
mask_val_npy_list = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data/3_slices_group/train_test_split/t2/val/masks_single/*'))

mask_list = sorted(glob.glob(TRAIN_DATASET_PATH+ 'BRATS_2021_niftii/*/*seg.nii.gz'))
t2_list= sorted(glob.glob(TRAIN_DATASET_PATH+ 'BRATS_2021_niftii/*/*t2.nii.gz'))


for img in range(len(mask_list)):
  temp_image_mask=nib.load(mask_list[img]).get_fdata()
  temp_image_mask=scaler1.fit_transform(temp_image_mask.reshape(-1, temp_image_mask.shape[-1])).reshape(temp_image_mask.shape)
  
  temp_image_t2= nib.load(t2_list[img]).get_fdata()
  temp_image_t2=scaler2.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)


  #now you have the scaled niftii file
  # Run the loop which goes from 51 to 90 slice and extracts 3 consecutive slices 

  slice=51

  while(slice<=90):
    #slice, slice +1, slice +2
    #extract the slices, stack them, convert them into numpy arrays and save them

    temp_slice_1=np.array(temp_image_mask[:, :, slice])
    temp_slice_2=np.array(temp_image_mask[:, :, slice+1])
    temp_slice_3=np.array(temp_image_mask[:, :, slice+2])

    t1_1=np.array(temp_image_t2[:, :, slice])
    t1_2=np.array(temp_image_t2[:, :, slice+1])
    t1_3=np.array(temp_image_t2[:, :, slice+2])

    

    mask_slice=np.stack([temp_slice_1, temp_slice_2, temp_slice_3], axis=2)
    mask_slice=to_categorical(mask_slice, num_classes=4)
    t2_slice=np.stack([t1_1, t1_2, t1_3], axis=2)

    if(img<119):
      #Validation data
      np.save(TRAIN_DATASET_PATH+'3_slices_group/train_test_split/t2/val/images/3group_'+ str(img)+ '_' + str(slice) + '.npy', t2_slice)
      np.save(TRAIN_DATASET_PATH+'3_slices_group/train_test_split/t2/val/masks/3group_'+ str(img)+ '_' + str(slice) + '.npy', mask_slice)
    else:
      #Training data
      np.save('/content/drive/MyDrive/Colab Notebooks/data/3_slices_group/t2/mask/3group_' +str(img)+ '_' + str(slice) + '.npy', mask_slice)
      np.save('/content/drive/MyDrive/Colab Notebooks/data/3_slices_group/t2/t2_train_images/3group_' +str(img)+ '_' + str(slice) + '.npy', t2_slice)
    
    slice=slice+3

  print('Done for image= ' + str(img))


  