import numpy as np
import nibabel as nib
import glob
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
scaler = MinMaxScaler()


def mask_percentage(mask):
  num_zeros = (mask == 0).sum()
  num_not_zero = (mask != 0).sum()

  percent=num_not_zero/(num_zeros+num_not_zero)
  if(percent < 0.002):
    return False
  else:
    return True
  

TRAIN_DATASET_PATH = '/content/drive/MyDrive/Colab Notebooks/data/'

mask_list = sorted(glob.glob(TRAIN_DATASET_PATH+ 'BRATS_2021_niftii/*/*seg.nii.gz'))
flair_list = sorted(glob.glob(TRAIN_DATASET_PATH+ 'BRATS_2021_niftii/*/*flair.nii.gz'))

datapoints=sorted(os.listdir(TRAIN_DATASET_PATH+ 'BRATS_2021_niftii/'))
datapoints=[d[-3:] for d in datapoints]
class_labels=pd.read_csv(TRAIN_DATASET_PATH+ 'classification_train_labels.csv')
class_labels.set_index('BraTS21ID', inplace=True)
datapoints=datapoints[:575]

class_labels_useful=pd.DataFrame(columns=['ID', 'MGMT_value'])
print(class_labels_useful.head())

for i in range(len(datapoints)):
  #i holds the current counter
  # Step 1: load the image, load the mask, scale the values
  data=datapoints[i]

  #data exists as a classification label
  if(int(data) in class_labels.index):

    image=nib.load(flair_list[i]).get_fdata()
    mask=nib.load(mask_list[i]).get_fdata()
    mask=mask.astype(np.uint8)
    mask[mask==4] = 3 
    image=scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)

    #start from slice = 40
    #Check if the slice of the mask has required % cutoff, if yes, increment data_count
    # Increment slice number

    slice=50
    data_count=0

    while(data_count<10 and slice<155):
      mask_slice=mask[:, :, slice]
      image_slice=image[:, :, slice]

      mask_slice=np.array(mask_slice)
      image_slice=np.array(image_slice)

      if(mask_percentage(mask_slice)==True): 
        #required cutoff is met
        data_count=data_count+1

        #now save the mask slice and the image slice
        if(i>500):  #put in the test folder
          np.save('/content/drive/MyDrive/Colab Notebooks/data/10_slices_patient/flair/test/img_' +str(i)+ '_' + str(data_count)+'.npy', image_slice)
          np.save('/content/drive/MyDrive/Colab Notebooks/data/10_slices_patient/mask/test/mask_' +str(i)+ '_' + str(data_count)+'.npy', mask_slice)
        else:   #put in the train folder
          np.save('/content/drive/MyDrive/Colab Notebooks/data/10_slices_patient/flair/train/img_' +str(i)+ '_' + str(data_count)+'.npy', image_slice)
          np.save('/content/drive/MyDrive/Colab Notebooks/data/10_slices_patient/mask/train/mask_' +str(i)+ '_' + str(data_count)+'.npy', mask_slice)
        dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
        class_labels_useful = class_labels_useful.append(dict, ignore_index = True)


      slice=slice+3
    print('Done for image=' + str(i))

print(len(class_labels_useful))
class_labels_useful.to_csv('/content/drive/MyDrive/Colab Notebooks/data/labels.csv')




