
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
from focal_loss import BinaryFocalLoss
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from sklearn.model_selection import KFold, StratifiedKFold
from focal_loss import BinaryFocalLoss




train_images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_best-slice-stack_bounded/flair/train/*'))
#train_masks_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_best-slice-stack_bounded/mask/train/*'))

test_images_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_best-slice-stack_bounded/flair/test/*'))
#test_masks_list=sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_best-slice-stack_bounded/mask/test/*'))

train_labels=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_best-slice-stack_bounded/train_labels.csv', index_col=0)
test_labels=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/data_cp303/axis_3_best-slice-stack_bounded/test_labels.csv', index_col=0)

train_images_list[:6]

images=[]
for img in train_images_list:
    images.append(cv2.resize(np.load(img), dsize=(100, 100), interpolation=cv2.INTER_LINEAR))
    
for img in test_images_list:
    images.append(cv2.resize(np.load(img), dsize=(100, 100), interpolation=cv2.INTER_LINEAR))
    
images=np.array(images)

np.shape(images)

test_labels=np.array(list(test_labels['MGMT_value']))
train_labels=np.array(list(train_labels['MGMT_value']))

#train_labels_one_hot=to_categorical(train_labels, num_classes=2)
#test_labels_one_hot=to_categorical(test_labels, num_classes=2)

#labels=np.concatenate((train_labels_one_hot, test_labels_one_hot), axis=0)
labels=np.concatenate((train_labels, test_labels), axis=0)
np.shape(labels)



def EfficientNet(input_size, n_classes):
  import tensorflow as tf
  from tensorflow.keras.layers import InputLayer, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Flatten, Conv2D, MaxPooling2D 
  from tensorflow.keras.models import Sequential, Model
  from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
  from tensorflow.keras.layers import Conv2D, MaxPooling2D

  #input_size = (100,100,3)

  model = tf.keras.applications.efficientnet.EfficientNetB3(
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

  model = Model(inputs=model.inputs, outputs=class_outputs, name="EfficientNetB3")

  from focal_loss import BinaryFocalLoss

  optimizer =  tf.keras.optimizers.Adam(learning_rate=1e-3)
  model.compile(optimizer=optimizer, loss=BinaryFocalLoss(gamma=2), metrics=["accuracy"])

  return model



kaiming_normal = keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')

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


#Use these function parameters to get ResNet18 or ResNet34
# ResNet18=resnet((100,100,3), [2, 2, 2, 2], 2)
# ResNet34=resnet((100,100,3), [3, 4, 6, 3], 2)

def get_score(model, x_train, x_test, class_train, class_test):
    from sklearn.metrics import roc_auc_score, accuracy_score
    model.fit(x_train, class_train, epochs=50, batch_size=8, verbose=1, validation_split=0.2)

    #epochs=50, batch_size=8, verbose=1, validation_split=0.2
    
    y_pred=model.predict(x_test)
    predicted_categories = np.argmax(y_pred, axis=1)
    actual_categories=np.argmax(class_test,axis=1)
    accuracy=accuracy_score(actual_categories, predicted_categories)
    auc=roc_auc_score(class_test, y_pred, average="weighted", multi_class="ovr")
    return accuracy, auc

def get_ensemble_score(model1, model2,  x_train, x_test, class_train, class_test):
    from sklearn.metrics import roc_auc_score, accuracy_score
    model1.fit(x_train, class_train, epochs=50, batch_size=8, verbose=1, validation_split=0.2)
    y_pred_1=model1.predict(x_test)

    model2.fit(x_train, class_train, epochs=50, batch_size=8, verbose=1, validation_split=0.2)
    y_pred_2=model2.predict(x_test)


    y_pred=(y_pred_1 + y_pred_2)/2


    predicted_categories = np.argmax(y_pred, axis=1)
    actual_categories=np.argmax(class_test,axis=1)
    accuracy=accuracy_score(actual_categories, predicted_categories)
    auc=roc_auc_score(class_test, y_pred, average="weighted", multi_class="ovr")
    return accuracy, auc



skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=8)
skf.get_n_splits(images, labels)


Effmodel=EfficientNet((100,100,3), 2)
ResNet18=resnet((100,100,3), [2, 2, 2, 2], 2)
ResNet18.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), metrics = ['accuracy'])
ResNet34=resnet((100,100,3),  [3, 4, 6, 3], 2)
ResNet34.compile(optimizer='adam', loss=BinaryFocalLoss(gamma=2), metrics = ['accuracy'])

accuracy=[]
auc=[]
fold=1

for i, (train_index, test_index) in enumerate(skf.split(images, labels)):
    X_train, X_test, y_train, y_test = images[train_index], images[test_index], labels[train_index], labels[test_index]
    y_train=to_categorical(y_train, num_classes=2)
    y_test=to_categorical(y_test, num_classes=2)

    #Using a ResNet18 and ResNet34 ensemble...
    # if you use EfficientNet in ensemble, use Effmodel here
    acc, au = get_ensemble_score(ResNet34, ResNet18, X_train, X_test, y_train, y_test)
    accuracy.append(acc)
    auc.append(au)
    fold=fold+1

auc=np.array(auc)
accuracy=np.array(accuracy)

print("auc mean:" + str(np.mean(auc)))
print("auc std:" + str(np.std(auc)))
print("acc mean:" + str(np.mean(accuracy)))
print("acc std:" + str(np.std(accuracy)))

print(auc)


