# Explainable and fully-automated clinical referral suggestion using multi-contrast magnetic resonance imaging for intra-axial mass-like lesions
Implementation of **[Explainable and fully-automated clinical referral suggestion using multi-contrast magnetic resonance imaging for intra-axial mass-like lesions]** which is submitted to Nature Communications. 

    TRAIN_FLDR
        ├── 0001.h5
        ├── 0002.h5
        ├── 0003.h5
        ├── ...
    
## Introduction
Official implementation of [Explainable and fully-automated clinical referral suggestion using multi-contrast magnetic resonance imaging for intra-axial mass-like lesions]. 

Our method consists of 3 phases: 
1. 3D Segmentation of mass-like lesions using contrast-enhanced T1-weighted (CE-T1w) and FLAIR volumes.
2. 2-stage classification of tumorous vs. non-tumorous condition, followed by clinical referral suggestion.
3. Interpretation of the model decision in discriminating tumorous vs. non-tumorous condition by layer-wise relevance propagation (LRP).
 
### Code Explanation

Before running the codes, Training data should be located in the TRAIN_FLDR, each subject formatted as HDF5 as follows: 

    TRAIN_FLDR
        ├── 0001.h5
        ├── 0002.h5
        ├── 0003.h5
        ├── ...
        
```
python Train.py --data-path path_to_TRAIN_FLDR --exp-dir path_to_CKPT_FLDR --gpu 0 --
```
