
import numpy as np
import nibabel as nib
import glob
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import cv2
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
from tensorflow.keras.utils import normalize
scaler = MinMaxScaler()

TRAIN_DATASET_PATH = '/content/drive/MyDrive/Colab Notebooks/data/'
mask_list = sorted(glob.glob(TRAIN_DATASET_PATH + 'BRATS_2021_niftii/*/*seg.nii.gz'))
flair_list = sorted(glob.glob(TRAIN_DATASET_PATH + 'BRATS_2021_niftii/*/*flair.nii.gz'))

datapoints=sorted(os.listdir(TRAIN_DATASET_PATH+ 'BRATS_2021_niftii/'))
datapoints=[d[-3:] for d in datapoints]
class_labels=pd.read_csv(TRAIN_DATASET_PATH+ 'classification_train_labels.csv')
class_labels.set_index('BraTS21ID', inplace=True)
datapoints=datapoints[:575]

def mask_percentage(mask):
  num_zeros = (mask == 0).sum()
  num_not_zero = (mask != 0).sum()

  percent=num_not_zero/(num_zeros+num_not_zero)
  return percent

datapoints_arr=np.array(datapoints)



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
skf.get_n_splits(datapoints, datapoints)

fold=1


for i, (train_index, test_index) in enumerate(skf.split(datapoints, datapoints)):

  class_labels_test=pd.DataFrame(columns=['ID', 'MGMT_value'])  
  if(fold>0):
    print("************************SAVING FOR FOLD" + str(fold)+"*******************************")
    datapoints_fold=list(datapoints_arr[test_index])
    UNetmodel=tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+'/UNet_fold'+str(fold)+'.hdf5')

    for i in range(len(datapoints_fold)):
      data=datapoints_fold[i]

      #data exists as a classification label 
      if(int(data) in class_labels.index):
        image=nib.load(flair_list[i]).get_fdata()
        mask=nib.load(mask_list[i]).get_fdata()
        mask=mask.astype(np.uint8)
        mask[mask!=0] = 1
        image=scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)


        ##############################FOR TRUE MASKS###################################
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

        #at this point, the best_slice_index has the slice index with the best tumour visibility
        #take one slice up and down of the best found slice, and stack them together
        # img_1=image[best_slice_index-1, :, :]
        # img_2=image[best_slice_index,:, :]
        # img_3=image[best_slice_index+1, :,:]

        mask_2_true=mask[:, :, best_slice_index]  #the actual best slice mask

        # rect= cv2.boundingRect(mask_2_true) #use the same bounding box since the slices will not differ that much

        # cropped_img_1 = img_1[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
        # cropped_img_2 = img_2[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
        # cropped_img_3= img_3[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]

        # cropped_img_true=np.stack([cropped_img_1, cropped_img_2, cropped_img_3], axis=2)


        ######################################FOR PREDICTED MASKS###################################
        image_npy=[]
        mask_npy=[]
        for slice in range(0,155):
          mask_slice=mask[:, :, slice]
          image_slice=image[:, :, slice]
          mask_slice=np.where(mask_slice!=0, 1, 0)
          image_npy.append(np.array(image_slice))
          mask_npy.append(np.array(mask_slice))

        image_npy=np.array(image_npy)
        mask_npy=np.array(mask_npy)

        
        # #Now we have the entire Test 3D volume loaded, take the slices 55 to 100

        image_ROI=image_npy[30:121, :, :]
        mask_ROI=mask_npy[30:121, :, :]
        image_ROI=normalize(image_ROI)
  
        y_pred=UNetmodel.predict(image_ROI)
        y_pred_argmax=np.argmax(y_pred, axis=3)

        #y_pred_argmax has the predicted masks stacked together, find the best mask
        slice=0
        best_slice_index=-1
        best_slice_mask_percent=0.0

        while(slice<91): #iterating over the 46 slices of the mask_ROI
           mask_slice=y_pred_argmax[slice,:, :]
           image_slice=image_ROI[slice,:, :]

           mask_slice=np.array(mask_slice)
           image_slice=np.array(image_slice)

           if(mask_percentage(mask_slice)>0.01 and mask_percentage(mask_slice)>best_slice_mask_percent):
             best_slice_index=slice
             best_slice_mask_percent= mask_percentage(mask_slice)


           slice=slice+1

        #we have the best slice of the predicted volume as best_slice_index
        img_1=image_npy[30+best_slice_index-1, :, :]
        img_2=image_npy[30+best_slice_index,:, :]
        img_3=image_npy[30+best_slice_index+1, :,:]

        # mask_1=y_pred_argmax[best_slice_index-1,:, :]
        mask_2=y_pred_argmax[best_slice_index,:, :]  #the actual best slice mask
        # mask_3=y_pred_argmax[best_slice_index+1,:, :]

        rect= cv2.boundingRect(mask_2.astype(np.uint8)) #use the same bounding box since the slices will not differ that much

        cropped_img_1 = img_1[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
        cropped_img_2 = img_2[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
        cropped_img_3= img_3[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]

        cropped_img=np.stack([cropped_img_1, cropped_img_2, cropped_img_3], axis=2)

        # #save the cropped image stack into the /foldX/test folder

        if(mask_percentage(mask_2_true)>0.01):
          np.save('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+ '/test_verified/img_' +str(i)+ '_' + 'stacked' + '.npy', cropped_img)
          dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
          print("saved stack!")
          class_labels_test = class_labels_test.append(dict, ignore_index = True)
          print("now saved labels too!")

        print('Done for image=' + str(i))


  #the fold has ended
  #Save classification test labels
  class_labels_test.to_csv('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+ '/test_labels_ver_'+ str(fold) +'.csv')
  fold=fold+1

