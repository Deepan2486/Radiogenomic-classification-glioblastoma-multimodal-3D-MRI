
import numpy as np
import nibabel as nib
import glob
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold
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



def mask_percentage(mask):
  num_zeros = (mask == 0).sum()
  num_not_zero = (mask != 0).sum()

  percent=num_not_zero/(num_zeros+num_not_zero)
  return percent

datapoints_arr=np.array(datapoints)



# USING A PRE-TESTED RANDOM SPLIT TO KEEP SHUFFLING SAVED
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=100)
skf.get_n_splits(datapoints, datapoints)

fold=1


for i, (train_index, test_index) in enumerate(skf.split(datapoints, datapoints)):

  class_labels_train=pd.DataFrame(columns=['ID', 'MGMT_value'])  
  if(fold>0):
    
    print("************************SAVING FOR FOLD" + str(fold)+"*******************************")
  #datapoints belonging to the particular fold
    datapoints_fold=list(datapoints_arr[train_index])

    for i in range(len(datapoints_fold)):
      data=datapoints_fold[i]

      #data exists as a classification label 
      if(int(data) in class_labels.index):
        image=nib.load(flair_list[i]).get_fdata()
        mask=nib.load(mask_list[i]).get_fdata()
        mask=mask.astype(np.uint8)
        mask[mask!=0] = 1
        image=scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)

        #STEP 3: SAMPLING SLICES FROM THE 3D VOLUMES

        slice=55
        data_count=0

        while(data_count<6 and slice<100):
          mask_slice=mask[:, :, slice]
          image_slice=image[:, :, slice]

          mask_slice=np.array(mask_slice)
          image_slice=np.array(image_slice)

          #save it in the seg-train folder of fold X
          np.save('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+ '/seg_train/images/img_' +str(i)+ '_' + str(data_count)+'.npy', image_slice)
          print("saved image!")
          np.save('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+ '/seg_train/masks/mask_' +str(i)+ '_' + str(data_count)+'.npy', mask_slice)
          print("saved mask too!")
          
          slice=slice+7
          data_count=data_count+1

        
        #STEP 4: saving the 3-slice-stack and classification labels
        

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
        img_1=image[ :, :, best_slice_index-1]
        img_2=image[:, :, best_slice_index]
        img_3=image[ :,:, best_slice_index+1]

        mask_1=mask[:, :, best_slice_index-1]
        mask_2=mask[:, :, best_slice_index]  #the actual best slice mask
        mask_3=mask[:, :, best_slice_index+1]

        rect= cv2.boundingRect(mask_2) #use the same bounding box since the slices will not differ that much

        cropped_img_1 = img_1[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
        cropped_img_2 = img_2[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
        cropped_img_3= img_3[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]

        cropped_img=np.stack([cropped_img_1, cropped_img_2, cropped_img_3], axis=2)
        if(mask_percentage(mask_2) > 0.01):
          np.save('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+ '/train/img_' +str(i)+ '_' + 'middle' + '.npy', cropped_img)
          dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
          class_labels_train = class_labels_train.append(dict, ignore_index = True)

        print('Done for image=' + str(i))


  #the fold has ended
  #Save classification labels
  class_labels_train.to_csv('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+ '/train_labels_'+ str(fold) +'.csv')
  fold=fold+1  #go to the next fold




