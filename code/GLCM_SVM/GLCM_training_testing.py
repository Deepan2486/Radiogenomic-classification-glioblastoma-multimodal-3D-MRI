import numpy as np 
import matplotlib.pyplot as plt
import glob
import cv2
import os
import seaborn as sns
import pandas as pd
from skimage.filters import sobel
from skimage.feature import greycomatrix, greycoprops
from skimage.measure import shannon_entropy
import math
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')


train_images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/best slice/flair/train/*'))
train_masks_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/best slice/mask/train/*'))

test_images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/best slice/flair/test/*'))
test_masks_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/best slice/mask/test/*'))

class_labels=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/best slice/best_slice_labels.csv', index_col=0)

train_images=[]
test_images=[]

train_masks=[]
test_masks=[]

for img in train_images_list:
    train_images.append(np.load(img))
    
train_images=np.array(train_images)

for img in test_images_list:
    test_images.append(np.load(img))
    
test_images=np.array(test_images)

for i in range(len(train_masks_list)):
  mask=np.load(train_masks_list[i])
  mask=np.where(mask!=0, 1, 0)
  train_masks.append(mask)

train_masks=np.array(train_masks)

for i in range(len(test_masks_list)):
  mask=np.load(test_masks_list[i])
  mask=np.where(mask!=0, 1, 0)
  test_masks.append(mask)

test_masks=np.array(test_masks)

plt.imshow(test_images[10], cmap='gray')

plt.imshow(test_masks[10], cmap='gray')

train_images_cropped=[]

for i in range(len(train_images)):
  img=train_images[i]
  mask_img=train_masks[i]


  cropped_img=np.where(mask_img!=0, img, 0)
  train_images_cropped.append(cropped_img)

train_images_cropped=np.array(train_images_cropped)

plt.imshow(train_images_cropped[10], cmap='gray')

test_images_cropped=[]

for i in range(len(test_images)):
  img=test_images[i]
  mask_img=test_masks[i]


  cropped_img=np.where(mask_img!=0, img, 0)
  test_images_cropped.append(cropped_img)

test_images_cropped=np.array(test_images_cropped)

plt.imshow(test_images_cropped[10], cmap='gray')

train_labels=class_labels[0:478]
test_labels=class_labels[478:]

test_labels=np.array(list(test_labels['MGMT_value']))
train_labels=np.array(list(train_labels['MGMT_value']))


scaled_img=np.array(train_images_cropped[0]*255, dtype=np.uint8)
GLCM = greycomatrix(scaled_img, [3], [0, np.pi/4, np.pi/2, 3*np.pi/4])  
GLCM_Energy = greycoprops(GLCM, 'energy')[0]

z=GLCM_Energy.mean()
print(z)

def feature_extractor(dataset):
    image_dataset = pd.DataFrame()
    for image in range(dataset.shape[0]):  #iterate through each file 
        df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
        
        img = dataset[image]
        img=np.array(img*255, dtype=np.uint8)
        GLCM = greycomatrix(img, [3], [0, np.pi/2, np.pi, 3*np.pi/2])
        GLCM_Energy = greycoprops(GLCM, 'energy')[0]
        GLCM_corr = greycoprops(GLCM, 'correlation')[0]
        GLCM_diss = greycoprops(GLCM, 'dissimilarity')[0]
        GLCM_hom = greycoprops(GLCM, 'homogeneity')[0]
        GLCM_contr = greycoprops(GLCM, 'contrast')[0]

        dict={'energy': GLCM_Energy.mean(), 'correlation': GLCM_corr.mean(), 'diss': GLCM_diss.mean(), 'hom': GLCM_hom.mean(), 'contrast': GLCM_contr.mean()}

        image_dataset = image_dataset.append(dict, ignore_index=True)
        
    return image_dataset


X_train_features=feature_extractor(train_images_cropped)
X_test_features=feature_extractor(test_images_cropped)

X_test_features.shape

from sklearn import svm
SVM_model = svm.SVC()  
SVM_model.fit(X_train_features, train_labels)

test_prediction = SVM_model.predict(X_test_features)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, test_prediction))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, test_prediction)

fig, ax = plt.subplots(figsize=(6,6))         # Sample figsize in inches
sns.set(font_scale=1.6)
sns.heatmap(cm, annot=True, linewidths=.5, ax=ax)

import lightgbm as lgb
lgbm_params = {'learning_rate':0.05, 'boosting_type':'dart',    
              'objective':'multiclass',
              'metric': 'multi_logloss',
              'num_leaves':100,
              'max_depth':10,
              'num_class':2}

d_train = lgb.Dataset(X_train_features, label=train_labels)
lgb_model = lgb.train(lgbm_params, d_train, 100)

test_prediction = lgb_model.predict(X_test_features)
test_prediction=np.argmax(test_prediction, axis=1)

from sklearn import metrics
print ("Accuracy = ", metrics.accuracy_score(test_labels, test_prediction))


