import numpy as np
import nibabel as nib
import glob
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os

scaler = MinMaxScaler()

TRAIN_DATASET_PATH = '/content/drive/MyDrive/Colab Notebooks/data/'

mask_list = sorted(glob.glob(TRAIN_DATASET_PATH+ 'BRATS_2021_niftii/*/*seg.nii.gz'))
flair_list = sorted(glob.glob(TRAIN_DATASET_PATH+ 'BRATS_2021_niftii/*/*flair.nii.gz'))

datapoints=sorted(os.listdir(TRAIN_DATASET_PATH+ 'BRATS_2021_niftii/'))
datapoints=[d[-3:] for d in datapoints]
class_labels=pd.read_csv(TRAIN_DATASET_PATH+ 'classification_train_labels.csv')
class_labels.set_index('BraTS21ID', inplace=True)

datapoints=datapoints[:575]
class_labels_useful=pd.DataFrame(columns=['ID', 'MGMT_value'])


def mask_percentage(mask):
  num_zeros = (mask == 0).sum()
  num_not_zero = (mask != 0).sum()

  percent=num_not_zero/(num_zeros+num_not_zero)
  return percent


for i in range(len(datapoints)):
  data=datapoints[i]

  #data exists as a classification label
  if(int(data) in class_labels.index):

    image=nib.load(flair_list[i]).get_fdata()
    mask=nib.load(mask_list[i]).get_fdata()
    mask=mask.astype(np.uint8)
    mask[mask==4] = 3 
    image=scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)

    slice=0
    best_slice_index=-1
    best_slice_mask_percent=0.0

    while(slice<155):
      mask_slice=mask[:, :, slice]
      image_slice=image[:, :, slice]

      mask_slice=np.array(mask_slice)
      image_slice=np.array(image_slice)

      if(mask_percentage(mask_slice)>0.01 and mask_percentage(mask_slice)>best_slice_mask_percent):
        best_slice_index=slice
        best_slice_mask_percent= mask_percentage(mask_slice)


      slice=slice+1

    if(i>500 and best_slice_index !=-1):  #put in the test folder
      # np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/best slice/flair/test/img_' +str(i)+ '_' + 'best_slice' + '.npy', image[:, :, best_slice_index])
      # np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/best slice/mask/test/mask_' +str(i)+ '_' + 'best_slice'+'.npy', mask[:, :, best_slice_index])
      dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
      class_labels_useful = class_labels_useful.append(dict, ignore_index = True)
    elif (i<=500 and best_slice_index !=-1):   #put in the train folder
      # np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/best slice/flair/train/img_' +str(i)+ '_' + 'best_slice' +'.npy', image[:, :, best_slice_index])
      # np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/best slice/mask/train/mask_' +str(i)+ '_' + 'best_slice' +'.npy', mask[:, :, best_slice_index])
      dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
      class_labels_useful = class_labels_useful.append(dict, ignore_index = True)
    

    print('Done for image=' + str(i))

mask_npy_list = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/best slice/mask/train/*'))

mask_npy_list_test = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/best slice/mask/test/*'))

mask_npy=[]

for img in mask_npy_list:
    mask_npy.append(np.load(img))
    
mask_npy=np.array(mask_npy)

plt.imshow(mask_npy[450])

class_labels_useful.to_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/best slice/best_slice_labels.csv')
