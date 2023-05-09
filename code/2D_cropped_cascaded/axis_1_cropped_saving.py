import os
import glob
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
scaler = MinMaxScaler()

datapoints=sorted(os.listdir(TRAIN_DATASET_PATH+ 'BRATS_2021_niftii/'))
datapoints=[d[-3:] for d in datapoints]
class_labels=pd.read_csv(TRAIN_DATASET_PATH+ 'classification_train_labels.csv')
class_labels.set_index('BraTS21ID', inplace=True)
datapoints=datapoints[:575]

class_labels_train=pd.DataFrame(columns=['ID', 'MGMT_value'])
#class_labels_train.head()

class_labels_test=pd.DataFrame(columns=['ID', 'MGMT_value'])
#class_labels_test.head()

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

    image_npy=[]
    mask_npy=[]

    for slice in range(0,155):
      image_slice=image[: ,: , slice]
      mask_slice=mask[: ,: , slice]
      image_npy.append(np.array(image_slice))
      mask_npy.append(np.array(mask_slice))


    image_npy=np.array(image_npy)
    mask_npy=np.array(mask_npy)

    image_middle=image_npy[70, :, :]
    mask_middle=mask_npy[70,: , :]

    rect= cv2.boundingRect(mask_middle) 
    cropped_img = image_middle[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
    cropped_mask= mask_middle[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
  
    if(i>500 and mask_percentage(mask_middle)>0.01):  #put in the test folder
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_bounded/flair/test/img_' +str(i)+ '_' + 'middle' + '.npy', cropped_img)
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_bounded/mask/test/mask_' +str(i)+ '_' + 'middle'+'.npy', cropped_mask)
      dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
      class_labels_test = class_labels_test.append(dict, ignore_index = True)
    elif (i<=500 and mask_percentage(mask_middle)>0.01):   #put in the train folder
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_bounded/flair/train/img_' +str(i)+ '_' + 'middle' +'.npy', cropped_img)
      np.save('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_bounded/mask/train/mask_' +str(i)+ '_' + 'middle' +'.npy', cropped_mask)
      dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
      class_labels_train = class_labels_train.append(dict, ignore_index = True)
    

    print('Done for image=' + str(i))

class_labels_train.to_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_slice/train_labels.csv')
class_labels_test.to_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_slice/test_labels.csv')

train_images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_bounded/flair/train/*'))
train_masks_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_bounded/mask/train/*'))

test_images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_bounded/flair/test/*'))
test_masks_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_bounded/mask/test/*'))

train_images=[]
train_masks=[]

for img in train_images_list:
    train_images.append(cv2.resize(np.load(img), dsize=(100, 100), interpolation=cv2.INTER_LINEAR))
    
train_images=np.array(train_images)


for i in range(len(train_masks_list)):
  mask=np.load(train_masks_list[i])
  mask=cv2.resize(mask, dsize=(100, 100), interpolation=cv2.INTER_LINEAR_EXACT)
  mask=np.where(mask!=0, 1, 0)
  train_masks.append(to_categorical(mask, num_classes=2))

train_masks=np.array(train_masks)

plt.imshow(train_images[120,:,:], cmap='gray')

train_images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_slice/flair/train/*'))
train_masks_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_slice/mask/train/*'))

train_images=[]
train_masks=[]

for img in train_images_list:
    train_images.append(cv2.resize(np.load(img), dsize=(100, 100), interpolation=cv2.INTER_LINEAR))
    
train_images=np.array(train_images)


for i in range(len(train_masks_list)):
  mask=np.load(train_masks_list[i])
  mask=cv2.resize(mask, dsize=(100, 100), interpolation=cv2.INTER_LINEAR_EXACT)
  mask=np.where(mask!=0, 1, 0)
  train_masks.append(to_categorical(mask, num_classes=2))

train_masks=np.array(train_masks)

plt.imshow(train_masks[120,:,:,1], cmap='gray')



