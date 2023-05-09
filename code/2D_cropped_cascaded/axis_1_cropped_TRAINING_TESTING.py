
import numpy as np
import nibabel as nib
import glob
import os
import cv2
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB5
from tensorflow.keras.layers import InputLayer, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Flatten, Conv2D, MaxPooling2D 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import AUC
from focal_loss import BinaryFocalLoss
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold



train_images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_bounded/flair/train/*'))
train_masks_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_bounded/mask/train/*'))

test_images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_bounded/flair/test/*'))
test_masks_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_bounded/mask/test/*'))

train_labels=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_slice/train_labels.csv', index_col=0)
test_labels=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_1_middle_slice/test_labels.csv', index_col=0)

train_images_list[:6]



train_images=[]
test_images=[]
train_masks=[]
test_masks=[]

images=[]

for img in train_images_list:
    images.append(cv2.resize(np.load(img), dsize=(100, 100), interpolation=cv2.INTER_LINEAR))
    
for img in test_images_list:
    images.append(cv2.resize(np.load(img), dsize=(100, 100), interpolation=cv2.INTER_LINEAR))
    
images=np.array(images)

images = np.expand_dims(images, axis=3)
#train_images = normalize(train_images, axis=1)

masks=[]

for img in train_images_list:
    train_images.append(cv2.resize(np.load(img), dsize=(100, 100), interpolation=cv2.INTER_LINEAR))
    
train_images=np.array(train_images)

for img in test_images_list:
    test_images.append(cv2.resize(np.load(img), dsize=(100, 100), interpolation=cv2.INTER_LINEAR))
    #cv2.resize(np.load(img), dsize=(100, 100), interpolation=cv2.INTER_LINEAR)
    
test_images=np.array(test_images)

np.shape(images)

for i in range(len(train_masks_list)):
  mask=np.load(train_masks_list[i])
  mask=cv2.resize(mask, dsize=(100, 100), interpolation=cv2.INTER_LINEAR_EXACT)
  mask=np.where(mask!=0, 1, 0)
  train_masks.append(to_categorical(mask, num_classes=2))

train_masks=np.array(train_masks)

for i in range(len(test_masks_list)):
  mask=np.load(test_masks_list[i])
  mask=cv2.resize(mask, dsize=(100, 100), interpolation=cv2.INTER_LINEAR_EXACT)
  mask=np.where(mask!=0, 1, 0)
  test_masks.append(to_categorical(mask, num_classes=2))

test_masks=np.array(test_masks)

test_labels=np.array(list(test_labels['MGMT_value']))
train_labels=np.array(list(train_labels['MGMT_value']))
train_labels_one_hot=to_categorical(train_labels, num_classes=2)
test_labels_one_hot=to_categorical(test_labels, num_classes=2)
labels=np.concatenate((train_labels_one_hot, test_labels_one_hot), axis=0)

x_train, x_test, class_train, class_test = train_test_split(images, labels, test_size = 0.20, random_state = 0)



input_size = (100,100,3)

model = tf.keras.applications.efficientnet.EfficientNetB3(
    include_top=False,
    weights='imagenet',
    input_tensor=Input(input_size,name='input'),
    input_shape=(100,100,3),
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

history = model.fit(x_train, class_train, 
              epochs=50, batch_size=8,
              verbose=1,
              validation_split=0.2)

y_pred=model.predict(test_images)
predicted_categories = np.argmax(y_pred, axis=1)
actual_categories=np.argmax(test_labels_one_hot,axis=1)
from sklearn.metrics import classification_report

print(classification_report(actual_categories, predicted_categories))


roc_auc_score(test_labels_one_hot, y_pred, average="weighted", multi_class="ovr")


############################### CREATED METHODS TO IMPORT MODELS##################

def get_score(model, x_train, x_test, class_train, class_test):
    from sklearn.metrics import roc_auc_score, accuracy_score
    model.fit(x_train, class_train, 
              epochs=50, batch_size=8,
              verbose=1,
              validation_split=0.2)
    
    y_pred=model.predict(x_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    actual_categories=np.argmax(class_test,axis=1)
    accuracy=accuracy_score(actual_categories, predicted_categories)
    auc=roc_auc_score(class_test, y_pred, average="weighted", multi_class="ovr")
    return accuracy, auc

def EfficientNet(input_size, n_classes):
  import tensorflow as tf
  from tensorflow.keras.layers import InputLayer, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Flatten, Conv2D, MaxPooling2D 
  from tensorflow.keras.models import Sequential, Model
  from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
  from tensorflow.keras.layers import Conv2D, MaxPooling2D

  #input_size = (100,100,3)

  model = tf.keras.applications.efficientnet.EfficientNetB7(
      include_top=False,
      weights='imagenet',
      input_tensor=Input(input_size,name='input'),
      input_shape=input_size,
      classes=n_classes
  )

  inputs = Input(input_size,name='input')

  model.trainable = False
      
  x = GlobalAveragePooling2D(name="avg_pool")(model.output)
  x = BatchNormalization()(x)
  top_dropout_rate = 0.4
  x = Dropout(top_dropout_rate)(x)
  x = Dense(32, activation="relu")(x)
  x = BatchNormalization()(x)
  x = Dropout(top_dropout_rate)(x)
  class_outputs = Dense(2, activation="softmax", name="class_pred")(x)

  model = Model(inputs=model.inputs, outputs=class_outputs, name="EfficientNetB7")

  from focal_loss import BinaryFocalLoss

  optimizer =  tf.keras.optimizers.Adam(learning_rate=1e-3)
  model.compile(optimizer=optimizer, loss=BinaryFocalLoss(gamma=2), metrics=["accuracy"])

  return model

def building_block(X, filter_size, filters, stride=1):

    # Save the input value for shortcut
    X_shortcut = X

    # Reshape shortcut for later adding if dimensions change
    if stride > 1:

        X_shortcut = Conv2D(filters, (1, 1), strides=stride, padding='same')(X_shortcut)
        X_shortcut = BatchNormalization(axis=3)(X_shortcut)

    # First layer of the block
    X = Conv2D(filters, kernel_size = filter_size, strides=stride, padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Second layer of the block
    X = Conv2D(filters, kernel_size = filter_size, strides=(1, 1), padding='same')(X)
    X = BatchNormalization(axis=3)(X)
    X = add([X, X_shortcut])  # Add shortcut value to main path
    X = Activation('relu')(X)

    return X

def create_model(input_shape, classes, name):

    # Define the input
    X_input = Input(input_shape)

    # Stage 1
    X = Conv2D(filters=16, kernel_size=3, strides=(1, 1), padding='same')(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    # Stage 2
    X = building_block(X, filter_size=3, filters=16, stride=1)
    X = building_block(X, filter_size=3, filters=16, stride=1)
    X = building_block(X, filter_size=3, filters=16, stride=1)
    X = building_block(X, filter_size=3, filters=16, stride=1)
    X = building_block(X, filter_size=3, filters=16, stride=1)

    # Stage 3
    X = building_block(X, filter_size=3, filters=32, stride=2)  # dimensions change (stride=2)
    X = building_block(X, filter_size=3, filters=32, stride=1)
    X = building_block(X, filter_size=3, filters=32, stride=1)
    X = building_block(X, filter_size=3, filters=32, stride=1)
    X = building_block(X, filter_size=3, filters=32, stride=1)

    # Stage 4
    X = building_block(X, filter_size=3, filters=64, stride=2)  # dimensions change (stride=2)
    X = building_block(X, filter_size=3, filters=64, stride=1)
    X = building_block(X, filter_size=3, filters=64, stride=1)
    X = building_block(X, filter_size=3, filters=64, stride=1)
    X = building_block(X, filter_size=3, filters=64, stride=1)

    # Average pooling and output layer
    X = GlobalAveragePooling2D()(X)
    X = Dense(classes, activation='softmax')(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name=name)

    return model



################################# EFFICIENTNET TESTING####################

kf = KFold(n_splits=5)

accuracy=[]
auc=[]

for train_index, test_index in kf.split(images,labels):
    X_train, X_test, y_train, y_test = images[train_index], images[test_index], labels[train_index], labels[test_index]
    Effmodel=EfficientNet((100,100,3), 2)
    acc, au = get_score(Effmodel, X_train, X_test, y_train, y_test)
    accuracy.append(acc)
    auc.append(au)

auc=np.array(auc)
accuracy=np.array(accuracy)

print("auc mean:" + str(np.mean(auc)))
print("auc std:" + str(np.std(auc)))
print("acc mean:" + str(np.mean(accuracy)))
print("acc std:" + str(np.std(accuracy)))

plt.imshow(train_images[100,:,:,0], cmap='gray')

plt.imshow(test_masks[10,:,:,1], cmap='gray')


############################ RESNET TESTING ###########################

classes=2
kf = KFold(n_splits=5)
ResNet32=create_model((100,100,1), 2, name='ResNet32')
ResNet32.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), metrics = ['accuracy'])

accuracy=[]
auc=[]

for train_index, test_index in kf.split(images,labels):
    X_train, X_test, y_train, y_test = images[train_index], images[test_index], labels[train_index], labels[test_index]
    acc, au = get_score(ResNet32, X_train, X_test, y_train, y_test)
    accuracy.append(acc)
    auc.append(au)

auc=np.array(auc)
accuracy=np.array(accuracy)

print("auc mean:" + str(np.mean(auc)))
print("auc std:" + str(np.std(auc)))
print("acc mean:" + str(np.mean(accuracy)))
print("acc std:" + str(np.std(accuracy)))







