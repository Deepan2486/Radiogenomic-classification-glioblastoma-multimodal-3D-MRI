

import numpy as np
import nibabel as nib
import glob
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import cv2
scaler = MinMaxScaler()

TRAIN_DATASET_PATH = '/content/drive/MyDrive/Colab Notebooks/data/'

mask_list = sorted(glob.glob(TRAIN_DATASET_PATH + 'BRATS_2021_niftii/*/*seg.nii.gz'))
flair_list = sorted(glob.glob(TRAIN_DATASET_PATH + 'BRATS_2021_niftii/*/*flair.nii.gz'))

datapoints=sorted(os.listdir(TRAIN_DATASET_PATH+ 'BRATS_2021_niftii/'))
datapoints=[d[-3:] for d in datapoints]
class_labels=pd.read_csv(TRAIN_DATASET_PATH+ 'classification_train_labels.csv')

class_labels.set_index('BraTS21ID', inplace=True)
datapoints=datapoints[:575]
class_labels_train=pd.DataFrame(columns=['ID', 'MGMT_value'])
class_labels_test=pd.DataFrame(columns=['ID', 'MGMT_value'])


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
    mask[mask!=0] = 1
    image=scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)

    slice=0
    best_slice_index=-1
    best_slice_mask_percent=0.0

    while(slice<240):
      mask_slice=mask[slice,:, :]
      image_slice=image[slice,:, :]

      mask_slice=np.array(mask_slice)
      image_slice=np.array(image_slice)

      if(mask_percentage(mask_slice)>0.01 and mask_percentage(mask_slice)>best_slice_mask_percent):
        best_slice_index=slice
        best_slice_mask_percent= mask_percentage(mask_slice)


      slice=slice+1

    #at this point, the best_slice_index has the slice index with the best tumour visibility
    #take one slice up and down of the best found slice, and stack them together
    img_1=image[best_slice_index-1, :, :]
    img_2=image[best_slice_index,:, :]
    img_3=image[best_slice_index+1, :,:]

    mask_1=mask[best_slice_index-1,:, :]
    mask_2=mask[best_slice_index,:, :]  #the actual best slice mask
    mask_3=mask[best_slice_index+1,:, :]

    rect= cv2.boundingRect(mask_2) #use the same bounding box since the slices will not differ that much

    cropped_img_1 = img_1[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
    cropped_img_2 = img_2[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
    cropped_img_3= img_3[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]

    cropped_mask_1= mask_1[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
    cropped_mask_2= mask_2[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
    cropped_mask_3= mask_3[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]

    cropped_img=np.stack([cropped_img_1, cropped_img_2, cropped_img_3], axis=2)
    cropped_mask=np.stack([cropped_mask_1, cropped_mask_2, cropped_mask_3], axis=2)
  
    if(i>500 and mask_percentage(mask_2)>0.01):  #put in the test folder
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_best-slice-stack_bounded/flair/test/img_' +str(i)+ '_' + 'middle' + '.npy', cropped_img)
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_best-slice-stack_bounded/mask/test/mask_' +str(i)+ '_' + 'middle'+'.npy', cropped_mask)
      dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
      class_labels_test = class_labels_test.append(dict, ignore_index = True)
    elif (i<=500 and mask_percentage(mask_2)>0.01):   #put in the train folder
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_best-slice-stack_bounded/flair/train/img_' +str(i)+ '_' + 'middle' +'.npy', cropped_img)
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_best-slice-stack_bounded/mask/train/mask_' +str(i)+ '_' + 'middle' +'.npy', cropped_mask)
      dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
      class_labels_train = class_labels_train.append(dict, ignore_index = True)
    

    print('Done for image=' + str(i))

class_labels_train.to_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_best-slice-stack_bounded/train_labels.csv')
class_labels_test.to_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_best-slice-stack_bounded/test_labels.csv')



