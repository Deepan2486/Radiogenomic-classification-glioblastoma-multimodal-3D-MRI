from sklearn.preprocessing import MinMaxScaler
import numpy as np
import nibabel as nib
import glob
import os
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import normalize
import tensorflow as tf
import segmentation_models_3D as sm
from focal_loss import BinaryFocalLoss
from keras.metrics import MeanIoU


TRAIN_DATASET_PATH = '/content/drive/MyDrive/Colab Notebooks/data/'
t1_npy_list = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data/middle_slice/flair/*'))
mask_npy_list = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data/middle_slice/mask/*'))

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

# dense_classification=Flatten()(c5)
# dense_classification=Dense(100)(dense_classification)
# classification_output = Dense(2, activation='softmax', name='classification')(dense_classification)
# segmentation_output = Conv2D(n_classes, (1,1), activation='softmax',name='segmentation')(c9)

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


class_labels=pd.read_csv(TRAIN_DATASET_PATH+ 'classification_train_labels.csv')
print(class_labels.head())
mask_list = sorted(glob.glob(TRAIN_DATASET_PATH+ 'BRATS_2021_niftii/*/*seg.nii.gz'))

datapoints=os.listdir(TRAIN_DATASET_PATH+ 'BRATS_2021_niftii/')
print(datapoints[570:])
datapoints=[d[-3:] for d in datapoints]
class_labels.set_index('BraTS21ID', inplace=True)
datapoints=datapoints[:575]
class_labels_useful=pd.DataFrame(columns=['ID', 'MGMT_value'])

mask_npy_all=np.load('/content/drive/MyDrive/Colab Notebooks/data/middle_slice/mask_single_class_combined.npy')
t1_npy_all=np.load('/content/drive/MyDrive/Colab Notebooks/data/middle_slice/cropped_flair_combined.npy')
print(t1_npy_all.shape)

def mask_percentage(mask):
  num_zeros = (mask == 0).sum()
  num_not_zero = (mask != 0).sum()

  percent=num_not_zero/(num_zeros+num_not_zero)
  if(percent < 0.002):
    return False
  else:
    return True

mask_npy=[]
t1_npy=[]

for i in range(len(datapoints)):
  data=datapoints[i]
  if (int(data) in class_labels.index and mask_percentage(mask_npy_all[i])==True):
    #index exists, so you can append the mask and image and add classifocation label too
    mask_npy.append(to_categorical(mask_npy_all[i], num_classes=2))
    t1_npy.append(np.load(t1_npy_list[i]))
    dict={'ID': int(data), 'MGMT_value': class_labels.loc[int(data)]['MGMT_value'] }
    class_labels_useful = class_labels_useful.append(dict, ignore_index = True)
    print('Done for image=' + str(i))

mask_npy=np.array(mask_npy)
t1_npy=np.array(t1_npy)

plt.imshow(t1_npy[100], cmap='gray')

t1_npy = np.expand_dims(t1_npy, axis=3)
train_images = normalize(t1_npy, axis=1)

class_labels_npy=np.array(list(class_labels_useful['MGMT_value']))
class_labels_one_hot=[]

for label in class_labels_npy:
  if label ==0:
    arr=[1,0]
  else:
    arr=[0,1]

  class_labels_one_hot.append(arr)

class_labels_one_hot=np.array(class_labels_one_hot)

class_labels_one_hot[400]

print(len(class_labels_one_hot))

x_test=t1_npy[0:80]
x_train=t1_npy[80:]

seg_test=mask_npy[0:80]
seg_train=mask_npy[80:]

class_test=class_labels_one_hot[0:80]
class_train=class_labels_one_hot[80:]

plt.imshow(seg_train[100, :, :, 1])


model.compile(optimizer='adam',
              loss={'classification': BinaryFocalLoss(gamma=2), 'segmentation': 'categorical_crossentropy'},
              loss_weights={'classification': 0.7, 'segmentation': 0.3},
              metrics=['accuracy']
              )

history = model.fit({'input': x_train},
              {'classification': class_train, 'segmentation': seg_train},
              epochs=30, batch_size=8,
              verbose=1,
              validation_split=0.2
             )

tf.keras.utils.plot_model(model, 'my_model.png')

model.save('/content/drive/MyDrive/Colab Notebooks/data/MTL_middle_20epochs.hdf5')

model.save_weights('/content/drive/MyDrive/Colab Notebooks/data/MTL_middle_20epochs.ckpt')

md=load_model('/content/drive/MyDrive/Colab Notebooks/data/MTL_middle_20epochs.hdf5')

print(model.metrics_names)

seg_loss = history.history['segmentation_loss']
class_loss = history.history['classification_loss']
seg_val_loss=history.history['val_segmentation_loss']
class_val_loss=history.history['val_classification_loss']
epochs = range(1, len(seg_loss) + 1)
plt.plot(epochs, class_loss, 'y', label='Training Classfn loss')
plt.plot(epochs, class_val_loss, 'r', label='Validation Classfn loss')
# plt.plot(epochs, seg_loss, 'b', label='Training Seg loss')
# plt.plot(epochs, seg_val_loss, 'c', label='Validation Seg loss')
plt.title('Train and val loss (Classification)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, seg_loss, 'b', label='Training Seg loss')
plt.plot(epochs, seg_val_loss, 'c', label='Validation Seg loss')
plt.title('Train and val loss (Segmentation)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


class_acc = history.history['classification_accuracy']
seg_acc= history.history['segmentation_accuracy']
val_seg_acc = history.history['val_segmentation_accuracy']
val_class_acc=history.history['val_classification_accuracy']

plt.plot(epochs, class_acc, 'y', label='Training classification Accuracy')
plt.plot(epochs, val_class_acc, 'r', label='Validation classification Accuracy')
# plt.plot(epochs, seg_acc, 'b', label='Training Seg Accuracy')
# plt.plot(epochs, val_seg_acc, 'c', label='Validation Seg Accuracy')
plt.title('Training and validation Accuracy (Classification)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, seg_acc, 'b', label='Training Seg Accuracy')
plt.plot(epochs, val_seg_acc, 'c', label='Validation Seg Accuracy')
plt.title('Training and validation Accuracy (Segmentation)')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

loss, score1, score2, acc1, acc2 = model.evaluate({'input': x_test},
                            {'classification': class_test, 'segmentation': seg_test})




y_pred=model.predict(x_test)

np.array(y_pred[1][:]).shape

y_pred_npy=np.array(y_pred[1][:])
#y_pred_npy=np.where(y_pred_npy==1, 0, 1)
plt.imshow(y_pred_npy[10,:, :, 1])
z=model.predict(x_test[10:11])


predicted_categories = np.argmax(y_pred[0], axis=1)

actual_categories=np.argmax(class_test,axis=1)

from sklearn.metrics import classification_report

print(classification_report(actual_categories, predicted_categories))

from sklearn.metrics import roc_auc_score

roc_auc_score(class_test, y_pred[0], average="weighted", multi_class="ovr")

y_pred_argmax=np.argmax(y_pred_npy, axis=3)

plt.imshow(y_pred_argmax[70])


n_classes = 2
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(seg_test[:,:,:,1], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
# class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
# class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
# class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
# class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[1,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[0,1])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)

plt.imshow(train_images[0, :,:,0], cmap='gray')

import pickle

with open('/content/drive/MyDrive/Colab Notebooks/data/trainHistoryDict', 'wb') as file_pi:
  pickle.dump(history.history, file_pi)

with open('/content/drive/MyDrive/Colab Notebooks/data/trainHistoryDict', 'rb') as file_pi:
  h=pickle.load(file_pi)

h['classification_accuracy']

import random
test_img_number = random.randint(0, len(seg_test))
test_img = x_test[test_img_number]
ground_truth=seg_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
#prediction = y_pred_npy[test_img_number]
#predicted_img = np.argmax(prediction, axis=2)


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(x_test[test_img_number][:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,1], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(y_pred_argmax[test_img_number], cmap='gray')
plt.show()


