# Instructions for Source Code files

### Note: 

* Most of the experiments in this project were done in a phased manner. Since the dataset constitutes of 3D scans (large size), fitting the entire 3D scan into numpy arrays (/image generators) exhausts the RAM. We therefore first collect the slice(s) which are needed for the experiment, save all such slices individually to a separate folder. In the training/testing phase, these slices are imported into the model.
* The source code files contain multiple file locations. Users are required to fill in their own file locations (local/remote) into those instances and then run the models. 
* Packages (as mentioned here) should be installed beforehand, before importing them into the .py files. The main dataset is the [BraTS dataset](https://github.com/Deepan2486/Radiogenomic-classification-glioblastoma-multimodal-3D-MRI/tree/master/data/BraTS_2021).
* Assumptions, as and when taken, are mentioned sectionwise in this Readme file.


### File descriptions and utility:

- (UNet_segmentation): 
