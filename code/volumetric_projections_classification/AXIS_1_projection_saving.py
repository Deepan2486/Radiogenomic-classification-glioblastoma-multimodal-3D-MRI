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

i=10

image=nib.load(flair_list[i]).get_fdata()
image_npy=[]

for slice in range(0,155):
  image_slice=image[: ,: , slice]
  image_npy.append(np.array(image_slice))


image_npy=np.array(image_npy)
print(image_npy.shape)

plt.imshow(image[: ,: , 70], cmap='gray')

image_max= np.max(image_npy, axis=0)
plt.imshow(image_max, cmap='gray')

datapoints=sorted(os.listdir(TRAIN_DATASET_PATH+ 'BRATS_2021_niftii/'))
datapoints=[d[-3:] for d in datapoints]
class_labels=pd.read_csv(TRAIN_DATASET_PATH+ 'classification_train_labels.csv')
class_labels.set_index('BraTS21ID', inplace=True)
datapoints=datapoints[:575]

class_labels_tested=pd.DataFrame(columns=['ID', 'MGMT_value'])

for i in range(len(datapoints)):
  data=datapoints[i]

  #data exists as a classification label
  if(int(data) in class_labels.index):
    dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
    class_labels_tested = class_labels_tested.append(dict, ignore_index = True)

len(class_labels_tested)

class_labels_tested['MGMT_value'].value_counts()

class_labels_train=pd.DataFrame(columns=['ID', 'MGMT_value'])
class_labels_train.head()

class_labels_test=pd.DataFrame(columns=['ID', 'MGMT_value'])
class_labels_test.head()


########################   MAXIMUM INTENSITY PROJECTION   #####################

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

    image_max= np.max(image_npy, axis=0)
    mask_max= np.max(mask_npy, axis=0)
  
    if(i>500):  #put in the test folder
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/MIP/flair/test/img_' +str(i)+ '_' + 'best_slice' + '.npy', image_max)
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/MIP/mask/test/mask_' +str(i)+ '_' + 'best_slice'+'.npy', mask_max)
      dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
      class_labels_test = class_labels_test.append(dict, ignore_index = True)
    elif (i<=500 and mask_percentage(mask_max)>0.01):   #put in the train folder
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/MIP/flair/train/img_' +str(i)+ '_' + 'best_slice' +'.npy', image_max)
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/MIP/mask/train/mask_' +str(i)+ '_' + 'best_slice' +'.npy', mask_max)
      dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
      class_labels_train = class_labels_train.append(dict, ignore_index = True)
    

    print('Done for image=' + str(i))

class_labels_test.to_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/MIP/test_labels.csv')
class_labels_train.to_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/MIP/train_labels.csv')



############################## STANDARD DEVIATION PROJECTION#########################
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

    image_std= np.std(image_npy, axis=0)
    mask_std= np.std(mask_npy, axis=0)
  
    if(i>500):  #put in the test folder
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/std_proj/flair/test/img_' +str(i)+ '_' + 'std' + '.npy', image_std)
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/std_proj/mask/test/mask_' +str(i)+ '_' + 'std'+'.npy', mask_std)
      dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
      class_labels_test = class_labels_test.append(dict, ignore_index = True)
    elif (i<=500 and mask_percentage(mask_std)>0.01):   #put in the train folder
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/std_proj/flair/train/img_' +str(i)+ '_' + 'std' +'.npy', image_std)
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/std_proj/mask/train/mask_' +str(i)+ '_' + 'std' +'.npy', mask_std)
      dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
      class_labels_train = class_labels_train.append(dict, ignore_index = True)
    

    print('Done for image=' + str(i))

class_labels_test.to_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/std_proj/test_labels.csv')
class_labels_train.to_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/std_proj/train_labels.csv')




