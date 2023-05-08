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
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Flatten, Conv2D, MaxPooling2D 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from tensorflow.keras.metrics import AUC
from focal_loss import BinaryFocalLoss




train_images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/partition_projections/axis_3/flair/train/*'))
#train_masks_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/partition_projections/axis_1/mask/train/*'))

test_images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/partition_projections/axis_3/flair/test/*'))
# test_masks_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/partition_projections/axis_2/mask/test/*'))

train_labels=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/partition_projections/axis_3/train_labels.csv', index_col=0)
test_labels=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/partition_projections/axis_3/test_labels.csv', index_col=0)

train_images=[]
test_images=[]
train_masks=[]
test_masks=[]

for img in train_images_list:
    train_images.append(np.load(img))
    
train_images=np.array(train_images)
plt.imshow(train_images[100,:,:,2], cmap='gray')
for img in test_images_list:
    test_images.append(np.load(img))
    
test_images=np.array(test_images)

# for i in range(len(train_masks_list)):
#   mask=np.load(train_masks_list[i])
#   #mask=cv2.resize(mask, dsize=(240, 240), interpolation=cv2.INTER_LINEAR)
#   mask=np.where(mask!=0, 1, 0)
#   train_masks.append(to_categorical(mask, num_classes=2))

# train_masks=np.array(train_masks)

# for i in range(len(test_masks_list)):
#   mask=np.load(test_masks_list[i])
#   #mask=cv2.resize(mask, dsize=(240, 240), interpolation=cv2.INTER_LINEAR)
#   mask=np.where(mask!=0, 1, 0)
#   test_masks.append(to_categorical(mask, num_classes=2))

# test_masks=np.array(test_masks)

print(train_images.shape)
print(test_images.shape)
# print(train_masks.shape)
# print(test_masks.shape)

test_labels=np.array(list(test_labels['MGMT_value']))
train_labels=np.array(list(train_labels['MGMT_value']))

train_labels_one_hot=to_categorical(train_labels, num_classes=2)
test_labels_one_hot=to_categorical(test_labels, num_classes=2)

input_size = (155,240,3)

model = tf.keras.applications.efficientnet.EfficientNetB3(
    include_top=False,
    weights='imagenet',
    input_tensor=Input(input_size,name='input'),
    input_shape=(155,240,3),
    classes=2
)

# input_layer_modified = InputLayer(input_shape=(240, 240, 1), name="input_modified")
# model.layers[0]=input_layer_modified

# new_model = tf.keras.models.model_from_json(model.to_json())

inputs = Input(input_size,name='input')
n_classes=2

model.trainable = False
    
x = GlobalAveragePooling2D(name="avg_pool")(model.output)
x = BatchNormalization()(x)
top_dropout_rate = 0.4
x = Dropout(top_dropout_rate)(x)
x = Dense(32, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(top_dropout_rate)(x)
class_outputs = Dense(2, activation="softmax", name="class_pred")(x)

model = Model(inputs=model.inputs, outputs=class_outputs, name="EfficientNetB5")


optimizer =  tf.keras.optimizers.Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss=BinaryFocalLoss(gamma=2), metrics=["accuracy"])

history = model.fit(train_images, train_labels_one_hot, 
              epochs=50, batch_size=8,
              verbose=1,
              validation_split=0.2)

y_pred=model.predict(test_images)

predicted_categories = np.argmax(y_pred, axis=1)
actual_categories=np.argmax(test_labels_one_hot,axis=1)

print(classification_report(actual_categories, predicted_categories))

roc_auc_score(test_labels_one_hot, y_pred, average="weighted", multi_class="ovr")




