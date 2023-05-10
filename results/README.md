# Instructions for Results

The experimentation results are arranged approach-wise. There is a direct correspondence to the source code arrangements in the [code](https://github.com/Deepan2486/Radiogenomic-classification-glioblastoma-multimodal-3D-MRI/tree/main/code/) folder.

The result files are explained below:
* [Segmentation Results](segmentation_results.pdf): This has the IoU scores (classwise) of the tumour segmentation task done in [Unet_segmentation](https://github.com/Deepan2486/Radiogenomic-classification-glioblastoma-multimodal-3D-MRI/tree/main/code/Unet_segmentation/). For the predicted tumour images, run the script in the respective segmentation file. 
* [MTL_results](MTL_results.pdf): This cumulatively has all the results for the MTL approach, done in  [MTL](https://github.com/Deepan2486/Radiogenomic-classification-glioblastoma-multimodal-3D-MRI/tree/main/code/UNet_MTL/). Users should run the below files to get the results:
  * T1-weighted and Flair middle slice results are performed in [middle_slice_MTL.py](https://github.com/Deepan2486/Radiogenomic-classification-glioblastoma-multimodal-3D-MRI/tree/main/code/UNet_MTL/middle_slice_MTL.py)
  * 10-slice sampled MTL results are performed in [10_slice_MTL.py](https://github.com/Deepan2486/Radiogenomic-classification-glioblastoma-multimodal-3D-MRI/tree/main/code/UNet_MTL/10_slice_MTL.py). 
