# The BraTS Dataset structure

- [classification_train_labels.csv](classification_train_labels.csv) has the true train labels of all the data samples included

- [Saved models](saved_models/) has some pre-trained models that might be used for quick testing purposes.

- The MRI scan directory:

  Each independent case has a dedicated folder identified by a five-digit number. Within each of these “case” folders, there are four sub-folders, each of them corresponding to each of the structural multi-parametric MRI (mpMRI) scans, in NifTI format. The exact mpMRI scans included are:

    - Fluid Attenuated Inversion Recovery (FLAIR)
    - T1-weighted pre-contrast (T1w)
    - T1-weighted post-contrast (T1Gd)
    - T2-weighted (T2)

   Exact folder structure:

  
