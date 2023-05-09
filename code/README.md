# Instructions for Source Code files

### Note: 

* Most of the experiments in this project were done in a phased manner. Since the dataset constitutes of 3D scans (large size), fitting the entire 3D scan into numpy arrays (/image generators) exhausts the RAM. We therefore first collect the slice(s) which are needed for the experiment, save all such slices individually to a separate folder. In the training/testing phase, these slices are imported into the model.
* The source code files contain multiple file locations. Users are required to fill in their own file locations (local/remote) into those instances and then run the models. 
* Packages (as mentioned here) should be installed beforehand, before importing them into the .py files. The main dataset is the [BraTS dataset](https://github.com/Deepan2486/Radiogenomic-classification-glioblastoma-multimodal-3D-MRI/tree/master/data/BraTS_2021).
* Assumptions, as and when taken, are mentioned sectionwise in this Readme file.


### File utility and How to use them:
(Click on the foldername to directly go to the folder directory)

- [**Data Exploration**](data_exploration/): This folder contains basic utility functions for dicom to Niftii conversion, to view the DCM files, use the existing dataset structure to extract the MRI scans etc. The functions are purely defined based on need. The user might or might not use it based on their initial requirements. Our code runs independent of these methods.

- [**UNet_segmentation**](Unet_segmentation/): This folder contains files which conducts semantic segmentation using UNet. Two variations are proposed: [middle_slice_segmentation.py](Unet_segmentation/middle_slice_segmentation.py) and [3_slice_classification.py](Unet_segmentation/3_slice_classification.py). In first method, the middle slices of each MRI scan are passed to the model, in the second case, we first sample non-overlapping stacks of 3 slices from the 155-sliced MRI scans (code for saving is in [3_slice_saving.py](Unet_segmentation/3_slice_saving.py)) and then the 3-stacks are passed to UNet to predict the mask of the middle slice.

  ```TRAIN_DATASET_PATH``` contains the root directory where the data is stored. ```t2_list``` and ```seg_list``` extracts the NifTI files from where the BraTS_2021 folder is located. Kindly change these paths to suit your need. Similarly in the saving phase, ```np.save()``` saved the sampled slices into a designated folder. Change the location of this folder.
  
  
- [**UNet_MTL**](UNet_MTL/): This folder has all the files responsible for creating, running and testing all variations of the Multi-task-learning model that performs joint segmentation+classification. The files:
  - [middle_slice_MTL.py](UNet_MTL/middle_slice_MTL.py) performs MTL on just the middle slice (the middle slice data was saved beforehand). The tumour masks are simplified to binary masks. Training phase involves taking the middle slice from each sample, testing phase involves testing on just the middle slice.
  - [10_slice_saving.py](UNet_MTL/10_slice_saving.py) and [10_slice_MTL.py](UNet_MTL/10_slice_MTL.py) trains the UNet model on 10 sampled slices on the training set. This makes the model robust to change in tumour shape. For testing, the middle slices of the test data samples are passed to the MTL model. 
  
  The sampled 10 slices are saved externally to a local folder (as shown in code) which should be changed in  ```np.save()```. Similarly, ```DATASET_PATH```, ``` train_images_list```, ``` train_mask_list```, ```class_labels``` should be changed to the saving locations.
  
  

- [**GLCM_SVM**](GLCM_SVM/): This uses the numerical feature representations of the tumour slices to train a traditional SVM classifier (there is no role of any CNN). The files are explained:
  - [best_slice_saving.py](GLCM_SVM/best_slice_saving.py) saves the best slice from the 3D scans (the slice with maximum tumour visibility). A simple mask percentage check is applied to achieve this.
  - [GLCM_training_testing.py](GLCM_SVM/GLCM_training_testing.py) trains the SVM by using the GLCM matrix. 

  The location site of saving the best slice can be altered in ```np.save()```. Similarly, lists like ``` train_images_list```, ``` train_mask_list``` should also be run with the actual file locations.

- [**Volumetric Projections**](volumetric_projections_classification/): Here we condense the 3D spatial information into a single slice, by taking projections along a particular axis. 
  - [AXIS_1_projection_saving.py](volumetric_projections_classification/AXIS_1_projection_saving.py) saves the projections along axis 1 (**axial axis**). Mean, stanadard deviation and maximum intensity projections are taken along axis 1. ```image_max= np.max(image_npy, axis=0)``` controls which axis the projection should be taken. Change ```np.save()``` location to where the calculated projections should be saved
  - [AXIS_1_classifier_projection_training_testing.py](volumetric_projections_classification/AXIS_1_classifier_projection_training_testing.py)  and [AXIS_1_MTL_projection_training_testing.py](volumetric_projections_classification/AXIS_1_MTL_projection_training_testing.py) performs **MTL** training/testing and **EfficientNet** classifier testing on the saved Standard deviations projections. It can be extended to other projections by changing the code in saving phase, and using appropriate location in ```train_images_list```.
  - [AXIS_2,3_projection_saving.py](volumetric_projections_classification/AXIS_2,3_projection_saving.py) saves the projections along axis 2 and axis 3 (**coronal and saggital axis**).
  - [AXIS_2,3_projection_training_testing.py](volumetric_projections_classification/AXIS_2,3_projection_training_testing.py) trains **MTL** classifier on the projections along axis 2 and axis 3. The code is similar to the one for axis 1.
  - [partitional_projection_saving.py](volumetric_projections_classification/partitional_projection_saving.py) divides the **240-slice 3D scan** (along axis 2, coronal axis) into 3 groups **(slice 0-80, 80-160 and 160-240)**. Then projections are taken individually along these 3 groups, stacked together to form a 3-stacked-sliced and saved.
  -  [partitional_projection_training_testing.py](volumetric_projections_classification/partitional_projection_training_testing.py) uses these stacks to get the final classification. Chnage the location of ```train_images_list``` and ```train_masks_list``` to the actual locations.

- [**2D Cropped Cascaded model**](2D_cropped_cascaded/): Here the files discuss the 2D cropped-mask cascaded model. This model trains separate UNet and classifier and uses a ```bounded-box-cropping``` to get the tumour ROI. The files:
  - [axis_1_cropped_saving.py](2D_cropped_cascaded/axis_1_cropped_saving.py) saves the cropped-tumour regions (slice=70 is fixed) from the tumour masks. We use ```cv2.boundingRect(<mask>)``` to gte the best rectangle that encloses the tumour ROI. Change the file location in ```np.save()``` to an appropriate folder.
  - [axis_1_cropped_TRAINING_TESTING.py](2D_cropped_cascaded/axis_1_cropped_TRAINING_TESTING.py) uses the saved tumour-cropped images (single slice) to train **EfficientNet** and **ResNet**. We use **5-fold-cross-validation** for improved training/testing (The functions are defined in the file itself but can also be found explicitly in [models](https://github.com/Deepan2486/Radiogenomic-classification-glioblastoma-multimodal-3D-MRI/tree/master/models/) . ```KFold(n_splits=5)``` is used to create the folds. 

- [**3D Cropped Cascaded model**](3D_cropped_cascaded/): Here we extend the 2D cropped model to a 3D adaptive version. The pipeline has the function of generating a 3D tumour voxel with 5 slices (with adaptive slice selectivity to choose the best slices). Separate Unet and classifiers are trained to achieve this task. This lengthy task is divided into 5 phases. Since we are opting for **5-fold-cross-validation**, training the Unet, saving the predicted tumour mask voxels, training the classifier on the voxels and testing on the particular fold cannot be done in a single iterative pass (due to data saving delays and conflicts). Hence we adapt a phase-by-phase approach. 
  - **PHASE 1 and PHASE 2**: [phase_1,2_saving_.py](3D_cropped_cascaded/phase_1,2_saving_.py) We explicitly use  ```StratifiedKFold(n_splits=5, shuffle=True, random_state=X)``` with a particular **random_state** in each phase to keep the foldwise division intact for the entire duration of the training/testing. In phase 1, we do 2 things: (a) sample 10 slices from each train data scan (for subsequent UNet training) (b) extract the best 3D volume by using ground truth masks along with classification true labels (for classifier training) In [phase1_segmentation_training.py](3D_cropped_cascaded/phase1_segmentation_training.py) we use the previously sampled slices to train **5 separate UNets** for each fold. The trained UNets are saved foldwise.
  - **PHASE 3**: [phase3_classification_training.py](3D_cropped_cascaded/phase3_classification_training.py) trains **separate classifiers (ResNet/ EfficientNet/ Ensembles)** on the tumour voxels saved from the ground truth masks. The trained classifiers are saved foldwise.
  - **PHASE 4**: [phase4_segmentation_testing+saving.py](3D_cropped_cascaded/phase4_segmentation_testing+saving.py) uses the previously trained foldwise UNets to pass the test data samples slice-wise, then gets the best predicted 3D tumour voxel. This is also saved foldwise to a separate folder. ```StratifiedKFold(n_splits=5, shuffle=True, random_state=X)``` with THE SAME **random_state** (used during phase 1) is used to get the current test datasets. 
  - **PHASE 5**: [phase5_finaltesting.py](3D_cropped_cascaded/phase5_finaltesting.py) does the final classification. Based on the predicted tumour voxels in phase 4, it feeds them into the foldwise saved classifiers to obtain the final resultant classification score. 

  ***Please note:** Several directory structures can be used to implemement this foldwise phased data-saving mechanism. The best way is to create separate folders called 'foldX' which will have the (a) sampled 10 slices for segmentation training (b) the ground truth 3D tumour voxels for classifier training (c) the saved Unet and saved classifier (d) the predicted tumour voxel foldwise. When we are currenly saving things in 'foldX', we are using all the other 4 folds as training data.*
  
  - [**3D cropped model (without segmentation training)**](3D_cropped_cascaded/testing_without_seg/): This folder is an addendum to the complex 3D cropped-cascaded model to reduce its pipeline complexity. It removes the Unet training and tumour prediction stage from the pipeline, rather uses the ground truth masks to obatin the 3D tumour voxel and train the classifier. This was mainly done for initial experimentative testing for the cropped model, since the pipeline implementation with segmentation is time consuming.
