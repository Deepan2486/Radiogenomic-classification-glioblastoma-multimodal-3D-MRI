# Instructions for Source Code files

### A Note: 

* Most of the experiments in this project were done in a phased manner. Since the dataset constitutes of 3D scans (large size), fitting the entire 3D scan into numpy arrays (/image generators) exhausts the RAM. We therefore first collect the slice(s) which are needed for the experiment, save all such slices individually to a separate folder. In the training/testing phase, these slices are imported into the model.
* The source code files contain multiple file locations. Users are required to fill in their own file locations (local/remote) into those instances and then run the models. 

