# RSNA22
RSNA 2022 - 3rd Place solution - Cervical Spine Fracture Detection

This is the source code for the 3rd place solution to the [RSNA 2022 Cervical Spine Fracture Detection](https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection).  
Video overview: [link](????)  
  
Sponsored by [RSNA](https://www.rsna.org/)
   
![](figs/study.gif) 

## Environment set up

For convenience of monitoring models we are using [neptune.ai](https://neptune.ai/home). 
In the configs, in the `config/` directory, I have set the neptune project to `light/kaggle-rsna2022`. 
You can switch this to your own neptune project, or else create a neptune project with that name.
   


## Data set up

Set up your [kaggle api](https://github.com/Kaggle/kaggle-api) for data download and run the below script. 
```
./bin/1_download.sh
```
The data is already split into folds and provided in the repo at `datamount/train_folded_v01.csv`.

## Model 1 - find study level bounding boxes. 

For this step run `./bin/2_bounding_box_train_infer.sh`. The details of the steps are below. 

### Bounding Box Label Creation

The first script, `_make_bbox_part1.py` uses the segmentations to create a bounding box for 
each slice in the study. We must align the z-axis direction of the slices to match slices for studies,
so we first load all dicom z-axis meta data to get the Study direction. We then load the segmentations, and 
use the outer C1-C7 segment map position to create a bounding box for each slice in the labelled studies. 
This outputs a file, `datamount/train_bbox_v01.csv.gz` in the format seen below. 


```
               StudyInstanceUID  slice_number   x0   y0   x1   y1  has_box  height  width  fold
110   1.2.826.0.1.3680043.10633           111  256  256  256  256        0       0      0     3
111   1.2.826.0.1.3680043.10633           112  256  256  256  256        0       0      0     3
112   1.2.826.0.1.3680043.10633           113  185  223  196  232        1      11      9     3
113   1.2.826.0.1.3680043.10633           114  185  223  196  232        1      11      9     3
114   1.2.826.0.1.3680043.10633           115  185  223  196  232        1      11      9     3
115   1.2.826.0.1.3680043.10633           116  182  215  202  235        1      20     20     3
116   1.2.826.0.1.3680043.10633           117  182  215  202  235        1      20     20     3
```

We also create a file `datamount/train_all_slices_v01.csv.gz` which is used for inference of boxes on all slices. 
This contains study name, slice number and fold for all slices. 

### Train a bounding box model

To train a bounding box model we will use config `cfg_loc_dh_01B`. This model loads random dicom slices
and learns for each, if there is a vertebrae bounding box, and if there is, what is the position of the 
box. Loss on the bounding box positions is only calculated where we have a bounding box, and loss for probability
of a box being present is calculated for all dicom slices. 

We run this model over each of 5 folds, and repeat it for 3 random seeds each. Obvously this model only,
trains on the slices we have segmentations for.  

### Run bounding box inference of all images. 

Here we take the weights of all trained models and run inference on all slices from found in the file 
`datamount/train_all_slices_v01.csv.gz`. 
You can check out the config file `cfg_loc_dh_01B_test` for details. 











