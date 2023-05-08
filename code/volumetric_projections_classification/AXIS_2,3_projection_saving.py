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
  return percent



TRAIN_DATASET_PATH = '/content/drive/MyDrive/Colab Notebooks/data/'
mask_list = sorted(glob.glob(TRAIN_DATASET_PATH+ 'BRATS_2021_niftii/*/*seg.nii.gz'))
flair_list = sorted(glob.glob(TRAIN_DATASET_PATH+ 'BRATS_2021_niftii/*/*flair.nii.gz'))

datapoints=sorted(os.listdir(TRAIN_DATASET_PATH+ 'BRATS_2021_niftii/'))
datapoints=[d[-3:] for d in datapoints]
class_labels=pd.read_csv(TRAIN_DATASET_PATH+ 'classification_train_labels.csv')

class_labels.set_index('BraTS21ID', inplace=True)
datapoints=datapoints[:575]

class_labels_train=pd.DataFrame(columns=['ID', 'MGMT_value'])
class_labels_train.head()

class_labels_test=pd.DataFrame(columns=['ID', 'MGMT_value'])
class_labels_test.head()


################################SD PROJECTION FOR AXIS 2######################

for i in range(len(datapoints)):
  data=datapoints[i]

  #data exists as a classification label
  if(int(data) in class_labels.index):

    image=nib.load(flair_list[i]).get_fdata()
    mask=nib.load(mask_list[i]).get_fdata()
    mask=mask.astype(np.uint8)
    mask[mask!=0] = 1
    image=scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)

    image_npy=[]
    mask_npy=[]

    for slice in range(0,155):
      image_slice=image[: ,: , slice]
      mask_slice=mask[: ,: , slice]
      image_npy.append(np.array(image_slice))
      mask_npy.append(np.array(mask_slice))


    image_npy=np.array(image_npy)
    mask_npy=np.array(mask_npy)

    image_std= np.std(image_npy, axis=1)
    mask_std= np.std(mask_npy, axis=1)
  
    if(i>500):  #put in the test folder
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_2_proj/flair/test/img_' +str(i)+ '_' + 'std' + '.npy', image_std)
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_2_proj/mask/test/mask_' +str(i)+ '_' + 'std'+'.npy', mask_std)
      dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
      class_labels_test = class_labels_test.append(dict, ignore_index = True)
    elif (i<=500 and mask_percentage(mask_std)>0.01):   #put in the train folder
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_2_proj/flair/train/img_' +str(i)+ '_' + 'std' +'.npy', image_std)
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_2_proj/mask/train/mask_' +str(i)+ '_' + 'std' +'.npy', mask_std)
      dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
      class_labels_train = class_labels_train.append(dict, ignore_index = True)
    

    print('Done for image=' + str(i))

class_labels_test.to_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_2_proj/test_labels.csv')
class_labels_train.to_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_2_proj/train_labels.csv')


#############################SD PROJECTION FOR AXIS 3######################


for i in range(len(datapoints)):
  data=datapoints[i]

  #data exists as a classification label
  if(int(data) in class_labels.index):

    image=nib.load(flair_list[i]).get_fdata()
    mask=nib.load(mask_list[i]).get_fdata()
    mask=mask.astype(np.uint8)
    mask[mask!=0] = 1
    image=scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)

    image_npy=[]
    mask_npy=[]

    for slice in range(0,155):
      image_slice=image[: ,: , slice]
      mask_slice=mask[: ,: , slice]
      image_npy.append(np.array(image_slice))
      mask_npy.append(np.array(mask_slice))


    image_npy=np.array(image_npy)
    mask_npy=np.array(mask_npy)

    image_std= np.std(image_npy, axis=2)
    mask_std= np.std(mask_npy, axis=2)
  
    if(i>500):  #put in the test folder
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_proj/flair/test/img_' +str(i)+ '_' + 'std' + '.npy', image_std)
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_proj/mask/test/mask_' +str(i)+ '_' + 'std'+'.npy', mask_std)
      dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
      class_labels_test = class_labels_test.append(dict, ignore_index = True)
    elif (i<=500 and mask_percentage(mask_std)>0.01):   #put in the train folder
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_proj/flair/train/img_' +str(i)+ '_' + 'std' +'.npy', image_std)
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_proj/mask/train/mask_' +str(i)+ '_' + 'std' +'.npy', mask_std)
      dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
      class_labels_train = class_labels_train.append(dict, ignore_index = True)
    

    print('Done for image=' + str(i))

class_labels_test.to_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_proj/test_labels.csv')
class_labels_train.to_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_proj/train_labels.csv')




