from sklearn.preprocessing import MinMaxScaler
import numpy as np
import nibabel as nib
import glob
import os
import pandas as pd
import math
import cv2
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.utils import normalize
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from focal_loss import BinaryFocalLoss
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score



"""AXIS 2 (CORONAL AXIS)"""
"""AXIS 3 (Saggital axis)"""

train_images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_proj/flair/train/*'))
train_masks_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_proj/mask/train/*'))

test_images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_proj/flair/test/*'))
test_masks_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_proj/mask/test/*'))

train_labels=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_2_proj/train_labels.csv', index_col=0)
test_labels=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_2_proj/test_labels.csv', index_col=0)

train_images=[]
test_images=[]
train_masks=[]
test_masks=[]

for img in train_images_list:
    train_images.append(cv2.resize(np.load(img), dsize=(240, 240), interpolation=cv2.INTER_LINEAR))
    
train_images=np.array(train_images)

for img in test_images_list:
    test_images.append(cv2.resize(np.load(img), dsize=(240, 240), interpolation=cv2.INTER_LINEAR))
    
test_images=np.array(test_images)

for i in range(len(train_masks_list)):
  mask=np.load(train_masks_list[i])
  mask=cv2.resize(mask, dsize=(240, 240), interpolation=cv2.INTER_LINEAR)
  mask=np.where(mask!=0, 1, 0)
  train_masks.append(to_categorical(mask, num_classes=2))

train_masks=np.array(train_masks)

for i in range(len(test_masks_list)):
  mask=np.load(test_masks_list[i])
  mask=cv2.resize(mask, dsize=(240, 240), interpolation=cv2.INTER_LINEAR)
  mask=np.where(mask!=0, 1, 0)
  test_masks.append(to_categorical(mask, num_classes=2))

test_masks=np.array(test_masks)

print(train_images.shape)
print(test_images.shape)
print(train_masks.shape)
print(test_masks.shape)

test_labels=np.array(list(test_labels['MGMT_value']))
train_labels=np.array(list(train_labels['MGMT_value']))

train_labels_one_hot=to_categorical(train_labels, num_classes=2)
test_labels_one_hot=to_categorical(test_labels, num_classes=2)



train_images = np.expand_dims(train_images, axis=3)
#train_images = normalize(train_images, axis=1)

test_images = np.expand_dims(test_images, axis=3)
#test_images = normalize(test_images, axis=1)

print(train_images.shape)
print(test_images.shape)
print(train_masks.shape)
print(test_masks.shape)

plt.imshow(test_masks[10, :, :, 1], cmap='gray')

plt.imshow(train_images[300, :, :,0], cmap='gray')


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

y_pred=model.predict(test_images)
y_pred_npy=np.array(y_pred[1][:])
predicted_categories = np.argmax(y_pred[0], axis=1)
actual_categories=np.argmax(test_labels_one_hot,axis=1)

print(classification_report(actual_categories, predicted_categories))

roc_auc_score(test_labels_one_hot, y_pred[0], average="weighted", multi_class="ovr")