
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import nibabel as nib
import glob
import os
import cv2
import pandas as pd
import math
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.layers import InputLayer, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Flatten, Conv2D, MaxPooling2D 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from focal_loss import BinaryFocalLoss
from sklearn.model_selection import KFold, StratifiedKFold



for fold in range(1,6):
  files = glob.glob('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+ '/test_true/*')
  print(len(files))
  labels=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+ '/test_labels_true_'+str(fold)+'.csv', index_col=0)
  print(len(labels))



def get_score(model, x_test, class_test):
    from sklearn.metrics import roc_auc_score, accuracy_score
    #model.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), metrics = ['accuracy'])
    y_pred=model.predict(x_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    actual_categories=np.argmax(class_test,axis=1)
    accuracy=accuracy_score(actual_categories, predicted_categories)
    auc=roc_auc_score(class_test, y_pred, average="weighted", multi_class="ovr")
    return accuracy, auc

def get_ensemble_score(model1, model2,  x_test, class_test):
    from sklearn.metrics import roc_auc_score, accuracy_score
    #model1.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), metrics = ['accuracy'])
    #model2.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), metrics = ['accuracy'])
    y_pred_1=model1.predict(x_test)
    y_pred_2=model2.predict(x_test)
    y_pred=(y_pred_1 + y_pred_2)/2

    predicted_categories = np.argmax(y_pred, axis=1)
    actual_categories=np.argmax(class_test,axis=1)
    accuracy=accuracy_score(actual_categories, predicted_categories)
    auc=roc_auc_score(class_test, y_pred, average="weighted", multi_class="ovr")
    return accuracy, auc


images=[]

fold=1
images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+ '/test_true/*'))
labels=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+ '/test_labels_true_'+str(fold)+'.csv', index_col=0)
for img in images_list:
    images.append(cv2.resize(np.load(img), dsize=(100, 100), interpolation=cv2.INTER_LINEAR))
labels=np.array(list(labels['MGMT_value']))


for fold in range(2,6):
  temp_images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+ '/test_true/*'))
  temp_labels=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+ '/test_labels_true_'+str(fold)+'.csv', index_col=0)
  for img in temp_images_list:
    images.append(cv2.resize(np.load(img), dsize=(100, 100), interpolation=cv2.INTER_LINEAR))
  temp_labels=np.array(list(temp_labels['MGMT_value']))
  labels=np.concatenate((temp_labels, labels), axis=0)

images=np.array(images)
labels=to_categorical(labels, num_classes=2)
print(np.shape(images))
print(np.shape(labels))


accuracy_18=[]
accuracy_34=[]
accuracy_ensemble=[]
auc_18=[]
auc_34=[]
auc_ensemble=[]


for fold in range(1,6):
  #load the Resnet models
  print("For Fold " + str(fold))
  ResNet18=tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+'/ResNet18_fold'+str(fold)+'.hdf5', custom_objects={'custom_loss': BinaryFocalLoss(gamma=2)} )
  ResNet34=tf.keras.models.load_model('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+'/ResNet34_fold'+str(fold)+'.hdf5', custom_objects={'custom_loss': BinaryFocalLoss(gamma=2)})

  #load the test images and test labels
  images=[]
  images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+ '/test_true/*'))
  labels=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+ '/test_labels_true_'+str(fold)+'.csv', index_col=0)

  for img in images_list:
    images.append(cv2.resize(np.load(img), dsize=(100, 100), interpolation=cv2.INTER_LINEAR))

  images=np.array(images)
  labels=np.array(list(labels['MGMT_value']))
  labels_one_hot=to_categorical(labels, num_classes=2)

  print(np.shape(images))
  print(np.shape(labels_one_hot))

  #pass them into get_score functions and collect accuracies and AUCs
  acc1, auc1=get_score(ResNet18,images,labels_one_hot)
  acc2, auc2=get_score(ResNet34,images,labels_one_hot)
  acc3, auc3=get_ensemble_score(ResNet18, ResNet34, images,labels_one_hot)

  accuracy_18.append(acc1)
  accuracy_34.append(acc2)
  accuracy_ensemble.append(acc3)

  auc_18.append(auc1)
  auc_34.append(auc2)
  auc_ensemble.append(auc3)

auc_18=np.array(auc_18)
auc_34=np.array(auc_34)
auc_ensemble=np.array(auc_ensemble)

print(np.mean(auc_18))
print(np.mean(auc_34))
print(np.mean(auc_ensemble))

accuracy_18=np.array(accuracy_18)
accuracy_34=np.array(accuracy_34)
accuracy_ensemble=np.array(accuracy_ensemble)

print(np.mean(accuracy_18))
print(np.mean(accuracy_34))
print(np.mean(accuracy_ensemble))

