
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
import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from focal_loss import BinaryFocalLoss
from tensorflow.keras.layers import InputLayer, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Flatten, Conv2D, MaxPooling2D 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D



kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

############################ DEFINING ResNet Model #########################

def conv3x3(x, out_planes, stride=1, name=None):
    x = layers.ZeroPadding2D(padding=1, name=f'{name}_pad')(x)
    return layers.Conv2D(filters=out_planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=name)(x)

def basic_block(x, planes, stride=1, downsample=None, name=None):
    identity = x

    out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn1')(out)
    out = layers.ReLU(name=f'{name}.relu1')(out)

    out = conv3x3(out, planes, name=f'{name}.conv2')
    out = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.bn2')(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = layers.Add(name=f'{name}.add')([identity, out])
    out = layers.ReLU(name=f'{name}.relu2')(out)

    return out

def make_layer(x, planes, blocks, stride=1, name=None):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
            layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.0.downsample.1'),
        ]

    x = basic_block(x, planes, stride, downsample, name=f'{name}.0')
    for i in range(1, blocks):
        x = basic_block(x, planes, name=f'{name}.{i}')

    return x

def resnet(input_shape, blocks_per_layer, num_classes=2):
    X_input = Input(input_shape)
    x = layers.ZeroPadding2D(padding=3, name='conv1_pad')(X_input)
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal, name='conv1')(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='bn1')(x)
    x = layers.ReLU(name='relu1')(x)
    x = layers.ZeroPadding2D(padding=1, name='maxpool_pad')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, name='maxpool')(x)

    x = make_layer(x, 64, blocks_per_layer[0], name='layer1')
    x = make_layer(x, 128, blocks_per_layer[1], stride=2, name='layer2')
    x = make_layer(x, 256, blocks_per_layer[2], stride=2, name='layer3')
    x = make_layer(x, 512, blocks_per_layer[3], stride=2, name='layer4')

    x = layers.GlobalAveragePooling2D(name='avgpool')(x)
    initializer = keras.initializers.RandomUniform(-1.0 / math.sqrt(512), 1.0 / math.sqrt(512))
    x = layers.Dense(units=num_classes, activation='softmax', kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)

    model = Model(inputs=X_input, outputs=x)

    return model




################################## CLASSIFIER TRAINING ###################################

for fold in range(1,6):
  images=[]
  images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+ '/train/*'))
  labels=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+ '/train_labels_'+str(fold)+'.csv', index_col=0)

  for img in images_list:
    images.append(cv2.resize(np.load(img), dsize=(100, 100), interpolation=cv2.INTER_LINEAR))

  images=np.array(images)
  labels=np.array(list(labels['MGMT_value']))
  labels_one_hot=to_categorical(labels, num_classes=2)

  print(np.shape(images))
  print(np.shape(labels_one_hot))

  ResNet18=resnet((100,100,3), [2, 2, 2, 2], 2)
  ResNet18.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), metrics = ['accuracy'])
  ResNet18.fit(images, labels_one_hot, epochs=50, batch_size=8, verbose=1, validation_split=0.2)

  ResNet34=resnet((100,100,3), [3, 4, 6, 3], 2)
  ResNet34.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), metrics = ['accuracy'])
  ResNet34.fit(images, labels_one_hot, epochs=50, batch_size=8, verbose=1, validation_split=0.2)

  print("************TRAINED Resnet MODEL " + str(fold)+"******************")
  ResNet18.save('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+'/ResNet18_fold'+str(fold)+'.hdf5')
  ResNet34.save('/content/drive/MyDrive/Colab Notebooks/5fold_CROPPED_BOUNDED/axis1/fold'+ str(fold)+'/ResNet34_fold'+str(fold)+'.hdf5')
  print("************SAVED Resnet MODEL " + str(fold)+"******************")

