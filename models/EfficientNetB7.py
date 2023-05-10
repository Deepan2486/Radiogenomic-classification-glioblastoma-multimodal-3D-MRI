
import tensorflow as tf
from tensorflow.keras.layers import InputLayer, GlobalAveragePooling2D, BatchNormalization, Dense, Dropout, Flatten, Conv2D, MaxPooling2D 
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from focal_loss import BinaryFocalLoss


def EfficientNetB7(input_size, n_classes):
 
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

  model = Model(inputs=model.inputs, outputs=class_outputs, name="EfficientNetB3")

  
  optimizer =  tf.keras.optimizers.Adam(learning_rate=1e-3)
  model.compile(optimizer=optimizer, loss=BinaryFocalLoss(gamma=2), metrics=["accuracy"])

  return model
