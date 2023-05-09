import numpy as np
import nibabel as nib
import glob
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import keras
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from focal_loss import BinaryFocalLoss
from sklearn.metrics import classification_report



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

train_images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/best slice/flair/train/*'))
train_masks_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/best slice/mask/train/*'))
class_labels_best_slice=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/best slice/best_slice_labels.csv', index_col=0)


train_images=[]
train_masks=[]

for img in train_images_list:
    train_images.append(np.load(img))
    
train_images=np.array(train_images)

for i in range(len(train_masks_list)):
  mask=np.load(train_masks_list[i])
  mask=np.where(mask!=0, 1, 0)
  train_masks.append(to_categorical(mask, num_classes=2))

train_masks=np.array(train_masks)

train_labels=class_labels_best_slice[0:478]
train_labels=np.array(list(train_labels['MGMT_value']))
train_labels_one_hot=to_categorical(train_labels, num_classes=2)


train_images = np.expand_dims(train_images, axis=3)
train_images = normalize(train_images, axis=1)

print(train_images.shape)
print(train_masks.shape)

model.compile(optimizer='adam',
              loss={'classification': BinaryFocalLoss(gamma=2), 'segmentation': 'categorical_crossentropy'},
              loss_weights={'classification': 0.9, 'segmentation': 0.1},
              metrics=['accuracy']
              )

history = model.fit({'input': train_images},
              {'classification': train_labels_one_hot, 'segmentation': train_masks},
              epochs=50, batch_size=8,
              verbose=1,
              validation_split=0.2
             )

test_images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/best slice/flair/test/*'))
test_masks_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/best slice/mask/test/*'))

test_images=[]
test_masks=[]

for img in test_images_list:
    test_images.append(np.load(img))
    
test_images=np.array(test_images)

for i in range(len(test_masks_list)):
  mask=np.load(test_masks_list[i])
  mask=np.where(mask!=0, 1, 0)
  test_masks.append(to_categorical(mask, num_classes=2))

test_masks=np.array(test_masks)

print(test_images.shape)
print(test_masks.shape)

test_labels=class_labels[501:575]
test_labels=np.array(list(test_labels['MGMT_value']))
test_labels_one_hot=to_categorical(test_labels, num_classes=2)


test_images = np.expand_dims(test_images, axis=3)
test_images = normalize(test_images, axis=1)

print(test_images.shape)
print(test_masks.shape)

image=nib.load(flair_list[10]).get_fdata()
image_npy=[]

for slice in range(0,155):
  image_slice=image[: ,: , slice]
  image_npy.append(np.array(image_slice))


image_npy=np.array(image_npy)
print(image_npy.shape)

image_npy = np.expand_dims(image_npy, axis=3)
image_npy = normalize(image_npy, axis=1)
print(image_npy.shape)

test_predicted_category=[]

def mask_percentage(mask):
  num_zeros = (mask == 0).sum()
  num_not_zero = (mask != 0).sum()

  percent=num_not_zero/(num_zeros+num_not_zero)
  return percent



#################################### MODEL TRAINING ###########################
input_size = (240,240,1)
inputs = Input(input_size,name='input')
n_classes=2
#s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
s = inputs

#Contraction path
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
c1 = Dropout(0.1)(c1)
c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
p1 = MaxPooling2D((2, 2))(c1)

c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
c2 = Dropout(0.1)(c2)
c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
p2 = MaxPooling2D((2, 2))(c2)
  
c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
c3 = Dropout(0.2)(c3)
c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
p3 = MaxPooling2D((2, 2))(c3)
  
c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
c4 = Dropout(0.2)(c4)
c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
p4 = MaxPooling2D(pool_size=(2, 2))(c4)
  
c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
c5_ = Dropout(0.3)(c5)
c5_ = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5_)

#Expansive path 
u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5_)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
c6 = Dropout(0.2)(c6)
c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
  
u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
c7 = Dropout(0.2)(c7)
c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
  
u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
c8 = Dropout(0.1)(c8)
c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
  
u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
c9 = Dropout(0.1)(c9)
c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
  
#outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)

GAP_1=GlobalAveragePooling2D()(c4)
GAP_2=GlobalAveragePooling2D()(c5_)
GAP_3=GlobalAveragePooling2D()(c6)
FC_1=concatenate([GAP_1,GAP_2,GAP_3])
FC_2=Dense(100)(FC_1)
classification_output = Dense(2, activation='softmax', name='classification')(FC_2)
segmentation_output = Conv2D(n_classes, (1,1), activation='softmax',name='segmentation')(c9)

model = Model(inputs = inputs,
     outputs = [classification_output, segmentation_output])
model.summary()



################################## DATA prediction ###############################
for i in range(501,len(datapoints)):  #using only >500 for the testing cluster (without checking the visibale mask percentage)
  data=datapoints[i]

  #data exists as a classification label
  if(int(data) in class_labels.index):
    image=nib.load(flair_list[i]).get_fdata()
    mask=nib.load(mask_list[i]).get_fdata()
    mask=mask.astype(np.uint8)
    mask[mask==4] = 3 
    image=scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)

    #saving  the ground truth classification test label
    dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
    class_labels_useful = class_labels_useful.append(dict, ignore_index = True)

  
    #use all the slices and check the count of each_predicted category
    image_npy=[]

    for slice in range(0,155):
      image_slice=image[: ,: , slice]
      if(mask_percentage(mask[:,:,slice]) > 0.01):
        image_npy.append(np.array(image_slice))

    image_npy=np.array(image_npy)
    image_npy = np.expand_dims(image_npy, axis=3)
    image_npy = normalize(image_npy, axis=1)

    y_pred=model.predict(image_npy)
    y_pred_npy=np.array(y_pred[1][:])
    predicted_categories = np.argmax(y_pred[0], axis=1)

    if(np.count_nonzero(predicted_categories == 0) > np.count_nonzero(predicted_categories == 1)):
      test_predicted_category.append(0)
    else:
      test_predicted_category.append(1)

test_predicted_category=[]



for i in range(501,len(datapoints)):  #using only >500 for the testing cluster (without checking the visibale mask percentage)
  data=datapoints[i]

  #data exists as a classification label
  if(int(data) in class_labels.index):
    image=nib.load(flair_list[i]).get_fdata()
    mask=nib.load(mask_list[i]).get_fdata()
    mask=mask.astype(np.uint8)
    mask[mask==4] = 3 
    image=scaler.fit_transform(image.reshape(-1, image.shape[-1])).reshape(image.shape)

    # #saving  the ground truth classification test label
    # dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
    # class_labels_useful = class_labels_useful.append(dict, ignore_index = True)

  
    #use all the slices and check the count of each_predicted category
    image_npy=[]

    for slice in range(0,155):
      image_slice=image[: ,: , slice]
      image_npy.append(np.array(image_slice))

    image_npy=np.array(image_npy)
    image_npy = np.expand_dims(image_npy, axis=3)
    image_npy = normalize(image_npy, axis=1)

    y_pred=model.predict(image_npy)
    y_pred_npy=np.array(y_pred[1][:])
    predicted_categories = np.argmax(y_pred[0], axis=1)



    #find out from the mask % which slice has best mask visibility
    y_pred_argmax=np.argmax(y_pred_npy, axis=3)

    best_slice_index=-1
    best_slice_mask_percent=0.0
    for slice in range(0,155):
      mask_slice=y_pred_argmax[slice]
      if(mask_percentage(mask_slice)>0.01 and mask_percentage(mask_slice)>best_slice_mask_percent):
        best_slice_index=slice
        best_slice_mask_percent= mask_percentage(mask_slice)

    test_predicted_category.append(predicted_categories[best_slice_index])

test_predicted_category=np.array(test_predicted_category)



actual_categories=np.argmax(test_labels_one_hot,axis=1)
print(classification_report(actual_categories, test_predicted_category))

