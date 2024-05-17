# Multiview species classification experiments
This repo contains the code to run the experiments reported in "Classifying geospatial objects from multiview aerial imagery using semantic meshes", available on [ArXiv](https://arxiv.org/abs/2405.09544).

# Instalation
This repository primarily contains scripts to run experiments and does not implement the core functionality. Instead, several independent repositories need to be installed. 

### [Geograypher](https://github.com/open-forest-observatory/geograypher)
This library determines the correspondences between individual images, meshes, and 2D geospatial data. 

### [MMSegmentation (fork)](https://github.com/open-forest-observatory/mmsegmentation)
MMSegmentation is a framework for semantic segmentation. The fork provides a few additional dataset definitions and improved inference flexibility.

### [segmentation_utils](https://github.com/open-forest-observatory/segmentation_utils)
This handles visualization and pre-processing of semantic segementation datasets.

### [data](https://osf.io/6snfq/)
The data for this project is hosted on Box and can be accessed through the Open Science Framework reference. This contains both input and intermediate data. The data should be downloaded into the `data` folder of this repository.

### Modifying `code/classification/constants.py`
There are several constants that you need to set based on how you install the above libraries. The first two, `SEGMENTATION_UTILS_PYTHON` and `MMSEG_PYTHONG` should be set to the location of the Python interpreter corresponding to the conda environments created in the earlier sections. This is used for subprocess calls because it is challenging to install all the dependencies all in one environment. The next two,  `SEGMENTATION_UTILS_FOLDER` and `MMSEGMENTATION_FOLDER`, correspond to the folders where those projects were installed. Scripts within these folder will be called by the subprocess. 


# Data structure 
The data in this project is taken from four sites, Chips, Delta, Lassic, and Valley. 

```
.
├── field_ref # Data collected from field surveys
│   │         # and combined with geometric tree detection
│   ├── crown_labels.gpkg
│   ├── crown_labels.gpkg
│   └── unlabeled_full_site_crowns
│       ├── chips.gpkg
│       ├── delta.gpkg
│       ├── lassic.gpkg
│       └── valley.gpkg
├── models # Trained segmentation models
│   └── multi_site
│       ├── MV-HN_chips_delta_lassic
│       ├── MV-LO_chips_delta_lassic
│       ├── ...
│       └── ortho_chips_delta_lassic
└── per_site_processing # Data from each site
    ├── chips
    │   ├── 01_images # The input images
    │   ├── 02_photogrammetry # Photogrammetry (geometric) products
    │   ├── 03_training_data # Formatted training data
    │   ├── 04_model_predictions # Segmentation predictions
    │   └── 05_processed_predictions # Geospatial predictions
    ├── delta
    ├── lassic
    └── valley
```
# Scripts
`01_MV_render_labels.py` Render labels corresponding each image

`01_ortho_create_training_chips.py` Create chips from the orthomosaic and create corresponding labels. 

`02_train_models.py` Preprocess the data and train the models

`03_predict_models.py` Generate predictions from the models and unlabeled data

`04_MV_aggregate_predictions.py` Aggregate the image-based predictions onto the mesh and then export into geospatial predictions

`04_ortho_aggregate.py` Aggregate the per-chip orthomosaic predictions into a single orthomosaic

`05_compute_accuracy.py` Compute the accuracy of the different predictions versus the groundtruth