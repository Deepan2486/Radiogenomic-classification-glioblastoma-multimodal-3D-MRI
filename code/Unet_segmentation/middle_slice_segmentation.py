
import tensorflow as tf
import numpy as np
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.utils import class_weight
from keras.metrics import MeanIoU
import random


wt0, wt1, wt2, wt3 = 0.25,0.25,0.25,0.25
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = tf.keras.optimizers.Adam(LR)
TRAIN_DATASET_PATH = 'DATASET-FOLDER-PATH'
scaler = MinMaxScaler()

########################   DATA SAVING ###########################

t2_list = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data/BRATS_2021_niftii/*/*t2.nii.gz'))

slice=70

for img in range(0,595):
  temp_image_t2=nib.load(t2_list[img]).get_fdata()
  temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)
  temp_slice=temp_image_t2[:, :, slice]
  temp_slice=np.array(temp_slice)
  np.save('/content/drive/MyDrive/Colab Notebooks/data/middle_slice/t2/image_' +str(img)+ '.npy', temp_slice)

t1_npy=[]
t1_npy_list = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data/middle_slice/t1/*'))

for img in t1_npy_list:
    t1_npy.append(np.load(img))
    
t1_npy=np.array(t1_npy)


seg_list = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data/BRATS_2021_niftii/*/*seg.nii.gz'))
slice=70
for img in range(len(seg_list)):
  temp_image_mask=nib.load(seg_list[img]).get_fdata()
  temp_image_mask=temp_image_mask.astype(np.uint8)
  temp_image_mask[temp_image_mask==4] = 3 
  temp_slice=temp_image_mask[:, :, slice]
  temp_slice=np.array(temp_slice)
  np.save('/content/drive/MyDrive/Colab Notebooks/data/middle_slice/mask/mask_' +str(img)+ '.npy', temp_slice)

mask_npy_list = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data/middle_slice/mask/*'))

mask_npy=[]

for img in mask_npy_list:
    mask_npy.append(np.load(img))
    
mask_npy=np.array(mask_npy)

mask_npy=np.load('/content/drive/MyDrive/Colab Notebooks/data/middle_slice/mask_single_class_combined.npy')

mask_npy.shape

plt.imshow(mask_npy[120])

t1ce_list = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data/BRATS_2021_niftii/*/*t1ce.nii.gz'))
flair_list = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data/BRATS_2021_niftii/*/*flair.nii.gz'))

slice=70
for img in range(len(flair_list)):
  temp_image_flair=nib.load(flair_list[img]).get_fdata()
  temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)
  temp_slice=temp_image_flair[:, :, slice]
  temp_slice=np.array(temp_slice)
  np.save('/content/drive/MyDrive/Colab Notebooks/data/middle_slice/flair/img_' +str(img)+ '.npy', temp_slice)

flair_npy_list = sorted(glob.glob('/content/drive/MyDrive/Colab Notebooks/data/middle_slice/flair/*'))
flair_npy=[]
for img in flair_npy_list:
    flair_npy.append(np.load(img))
    
flair_npy=np.array(flair_npy)


#################################### MODEL TRAINING ###########################

def multi_unet_model(n_classes=2, IMG_HEIGHT=240, IMG_WIDTH=240, IMG_CHANNELS=1):
#Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
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
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    
    #Expansive path 
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
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
     
    outputs = Conv2D(n_classes, (1, 1), activation='softmax')(c9)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    
    #NOTE: Compile the model in the main program to make it easy to test with various loss functions
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    #model.summary()
    
    return model



#Resizing images, if needed
SIZE_X = 240 
SIZE_Y = 240
n_classes=4
train_images=flair_npy  #put whichever modality you are using
train_masks=mask_npy

labelencoder = LabelEncoder()
n, h, w = train_masks.shape
train_masks_reshaped = train_masks.reshape(-1,1)
train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
train_images = np.expand_dims(train_images, axis=3)
#train_images = normalize(train_images, axis=1)
train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)


#Create a subset of data for quick testing
#Picking 10% for testing and remaining for training

X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)
#Further split training data t a smaller subset for quick testing of models
X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.2, random_state = 0)
print("Class values in the dataset are ... ", np.unique(y_train))
print(X_train.shape)
print(y_train.shape)


train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))
test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))
print(train_masks_cat.shape)
print(y_train_cat.shape)


class_weights = class_weight.compute_class_weight(class_weight= 'balanced',
                                                 classes= np.unique(train_masks_reshaped_encoded),
                                                 y= train_masks_reshaped_encoded)
print("Class weights are...:", class_weights)


IMG_HEIGHT = X_train.shape[1]
IMG_WIDTH  = X_train.shape[2]
IMG_CHANNELS = X_train.shape[3]

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)

model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train_cat, 
                    batch_size = 8, 
                    verbose=1, 
                    epochs=50, 
                    validation_data=(X_test, y_test_cat), 
                    #class_weight=class_weights,
                    shuffle=False)

model.save('SAVE TRAINED MODEL HERE')



################################    MODEL PREDICTION TESTING   ################################
_, acc = model.evaluate(X_test, y_test_cat)
print("Accuracy is = ", (acc * 100.0), "%")

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Train and val loss (flair slice)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'y', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()



y_pred=model.predict(X_test)
y_pred_argmax=np.argmax(y_pred, axis=3)
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])
# class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[1,0])
# class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[0,1])
# print(class1_IoU)
# print(class2_IoU)

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)

plt.imshow(train_images[0, :,:,0], cmap='gray')
plt.imshow(train_masks[0], cmap='gray')


test_img_number = random.randint(0, len(X_test))
test_img = X_test[test_img_number]
ground_truth=y_test[test_img_number]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Ground Truth mask')
plt.imshow(ground_truth[:,:,0], cmap='plasma')
plt.subplot(233)
plt.title('Predicted mask')
plt.imshow(predicted_img, cmap='plasma')
plt.show()







