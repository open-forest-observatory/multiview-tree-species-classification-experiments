import os
from pathlib import Path

# Python utilities
# TODO modify these if you are setting up your own project
# Because of the difficulty of having a shared set of dependencies, we use multiple different conda
# environments for different parts of the pipeline
# Python interpreter for the segmentation_utils conda repository
SEGMENTATION_UTILS_PYTHON = "/ofo-share/repos-david/conda/envs/mmseg-utils/bin/python"
# Python interpreter for the MMSegmentation conda repository
MMSEG_PYTHON = "/ofo-share/repos-david/conda/envs/openmmlab/bin/python"

# Location of the segmentation_utils folder
SEGMENTATION_UTILS_FOLDER = "/ofo-share/repos-david/mmsegmentation_utils"
# Location of the MMSegmentation folder
MMSEGMENTATION_FOLDER = "/ofo-share/repos-david/mmsegmentation"

# Scripts used for various tasks
FOLDER_TO_CITYSCAPES_SCRIPT = Path(
    SEGMENTATION_UTILS_FOLDER, "dev", "dataset_creation", "folder_to_cityscapes.py"
)
VIS_PREDS_SCRIPT = Path(
    SEGMENTATION_UTILS_FOLDER, "dev", "visualization", "visualize_semantic_labels.py"
)
TRAIN_SCRIPT = Path(MMSEGMENTATION_FOLDER, "tools", "train.py")
INFERENCE_SCRIPT = Path(MMSEGMENTATION_FOLDER, "tools", "inference.py")

# Important folders
# TODO make this work for notebooks as well if needed
# PROJECT_ROOT = Path(__file__, "..", "..", "..").resolve()
DEFAULT_INPUT_DATA_DIR = "/ofo-share/str-disp_drone-data-v2/"
DEFAULT_PREDICTION_DATA_DIR = "/ofo-share/str-disp_drone-data-v2/2_classification"

LABELS_COLUMN = "species_observed"

ALL_SITE_NAMES = ["chips", "delta", "lassic", "valley"]

TRAINING_IMGS_EXT = ".png"
INFERENCE_IMGS_EXT = ".png"
CHIP_SIZE = 3648
BATCH_SIZE = 2
TRAINING_STRIDE = int(CHIP_SIZE / 2)
INFERENCE_STRIDE = int(CHIP_SIZE / 2)

# Points less than this height (meters) above the DTM are considered ground
GROUND_HEIGHT_THRESHOLD = 2
# The image is downsampled to this fraction for accelerated rendering
RENDER_IMAGE_SCALE = 1
# Cameras within this distance of the traing data are used in the rendering process
BUFFER_RADIUS_METERS = 50
# Downsample target
DOWNSAMPLE_TARGET = 1
# Image downsample factor for aggregation
AGGREGATE_IMAGE_SCALE = 1
# Weight of ground when assigning polygons labels
GROUND_WEIGHT_POLYGON_LABELING = 0.01


# Step 1 functions
def convert_short_site_name_to_long(short_site_name):
    """"""
    return {
        "chips": "chips_20240621T0429",
        "delta": "delta_20240617T2314",
        "lassic": "lassic_20240621T0430",
        "valley": "valley_20240607T2022",
    }[short_site_name]


def get_labels_filename(input_data_dir, include_snag_class=True):
    """Path to the groundtruth labeling"""
    if include_snag_class:
        # Ground truth information
        labels_filename = Path(
            input_data_dir,
            "predicted-treecrowns-w-field-data",
            "predicted-treecrowns-w-field-data-cleaned_david.gpkg",
        )
    else:
        labels_filename = Path(
            input_data_dir,
            "predicted-treecrowns-w-field-data",
            "predicted-treecrowns-w-field-data-cleaned-snagsremoved_david.gpkg",
        )

    return labels_filename


def get_IDs_to_labels(include_snag_class=True):
    IDs_to_labels = {
        0: "ABCO",
        1: "CADE",
        2: "PILA",
        3: "PIPJ",
        4: "PSME",
    }
    if include_snag_class:
        IDs_to_labels[5] = "SNAG"

    return IDs_to_labels


def get_ortho_filename(site, input_data_dir):
    long_site_name = convert_short_site_name_to_long(site)

    return Path(
        input_data_dir,
        "photogrammetry",
        "outputs",
        f"{long_site_name}_ortho_dsm-mesh.tif",
    )


def get_ortho_training_data_folder(site, prediction_data_dir, append_vis=False):
    training_data_folder = Path(
        prediction_data_dir,
        "ortho_training_data",
        site,
    )

    if append_vis:
        return Path(training_data_folder, "vis")
    return training_data_folder


def get_subfolder_by_mission_type(folder, site_name, mission_type):
    raise NotImplementedError()
    subfolders = list(filter(os.path.isdir, list(folder.glob("*"))))
    if mission_type == "MV-LO":
        subfolders = list(filter(lambda x: "_120m" not in str(x), subfolders))
    elif mission_type == "MV-HN":
        subfolders = list(filter(lambda x: "_120m" in str(x), subfolders))
    else:
        raise ValueError(f"Mission type {mission_type} not valid")

    if len(subfolders) != 1:
        raise ValueError("Subfolders")

    return subfolders[0]


def get_image_folder(site_name, input_data_dir, mission_type=None):
    image_folder = Path(input_data_dir, "imagery-raw", "1_manually-cleaned")

    if mission_type is None:
        return image_folder
    return get_subfolder_by_mission_type(
        image_folder, site_name=site_name, mission_type=mission_type
    )


def get_mesh_filename(site_name, input_data_dir):

    long_site_name = convert_short_site_name_to_long(site_name)
    return Path(
        input_data_dir, "photogrammetry", "outputs", f"{long_site_name}_model_local.ply"
    )


def get_cameras_filename(site_name, input_data_dir):
    long_site_name = convert_short_site_name_to_long(site_name)
    return Path(
        input_data_dir, "photogrammetry", "outputs", f"{long_site_name}_cameras_manual_export.xml"
    )


def get_mesh_transform_filename(site_name, input_data_dir):
    return get_cameras_filename(site_name, input_data_dir=input_data_dir)


def get_DTM_filename(site_name, input_data_dir):
    long_site_name = convert_short_site_name_to_long(site_name)
    return Path(
        input_data_dir,
        "photogrammetry",
        "outputs",
        f"{long_site_name}_dtm-ptcloud.tif",
    )


def get_MV_training_folder(site, prediction_data_dir, append_vis=False):

    training_data_folder = Path(
        prediction_data_dir,
        "MV_training_data",
        site,
    )

    if append_vis:
        return Path(training_data_folder, "vis")
    return training_data_folder


def get_labeled_mesh_filename(site_name, prediction_data_dir, get_vis_filename=False):
    vis_file_stem = Path(
        prediction_data_dir,
        "labeled_mesh",
        f"{site_name}_labeled_mesh",
    )
    if get_vis_filename:
        return vis_file_stem.with_suffix(".png")
    else:
        return vis_file_stem.with_suffix(".ply")


# Step 2 functions
def get_training_sites_str(training_sites):
    return "_".join(training_sites)


def get_training_data_folder(prediction_data_dir, training_sites, mission_type):
    """Folder where model inputs and results go"""
    training_sites_str = get_training_sites_str(training_sites)
    return Path(
        prediction_data_dir,
        "training_data",
        f"{mission_type}_{training_sites_str}",
    )


def get_aggregated_labels_folder(prediction_data_dir, training_sites, mission_type):
    training_data_folder = get_training_data_folder(
        prediction_data_dir=prediction_data_dir,
        training_sites=training_sites,
        mission_type=mission_type,
    )
    return Path(training_data_folder, "labels")


def get_aggregated_images_folder(prediction_data_dir, training_sites, mission_type):
    training_data_folder = get_training_data_folder(
        prediction_data_dir=prediction_data_dir,
        training_sites=training_sites,
        mission_type=mission_type,
    )
    return Path(training_data_folder, "images")


def get_work_dir(prediction_data_dir, training_sites, mission_type, run_ID):
    """
    Where to train the model
    """
    training_data_folder = get_training_data_folder(
        prediction_data_dir=prediction_data_dir,
        training_sites=training_sites,
        mission_type=mission_type,
    )
    return Path(training_data_folder, "work_dir", run_ID)


def get_mmseg_style_training_folder(prediction_data_dir, training_sites, mission_type):
    training_data_folder = get_training_data_folder(
        prediction_data_dir=prediction_data_dir,
        training_sites=training_sites,
        mission_type=mission_type,
    )
    # Get the description (mission_<site names>) tag so the associated config will be named appropriately
    description = training_data_folder.parts[-1]
    return Path(training_data_folder, f"{description}_mmseg_formatted_data")


# Step 3 functions
def get_prediction_folder(
    prediction_site, training_sites, mission_type, run_ID, prediction_data_dir
):
    training_sites_str = get_training_sites_str(training_sites=training_sites)
    return Path(
        prediction_data_dir,
        "model_predictions",
        f"{training_sites_str}_{mission_type}_model",
        f"run_{run_ID}",
        prediction_site,
    )


def get_ortho_prediction_data_folder(site, prediction_data_dir, append_vis=False):
    training_data_folder = Path(
        prediction_data_dir,
        "ortho_prediction_data",
        site,
    )

    if append_vis:
        return Path(training_data_folder, "vis")
    return training_data_folder


# Step 4 functions
def get_predicted_output_base_file(
    prediction_site, training_sites, prediction_data_dir
):
    training_sites_str = get_training_sites_str(training_sites)
    return Path(
        prediction_data_dir,
        "accuracy_evaluation",
        f"{training_sites_str}_model",
        prediction_site,
    )


def get_figure_export_confusion_matrix_file(
    prediction_site,
    training_sites,
    mission_type,
    run_ID,
    prediction_data_dir,
):
    predicted_output_base_file = get_predicted_output_base_file(
        prediction_site, training_sites, prediction_data_dir=prediction_data_dir
    )

    return Path(
        predicted_output_base_file,
        mission_type,
        "cf_matrix",
        f"run_{run_ID}.svg",
    )


def get_npy_export_confusion_matrix_file(
    prediction_site,
    training_sites,
    mission_type,
    run_ID,
    prediction_data_dir,
):
    predicted_output_base_file = get_predicted_output_base_file(
        prediction_site, training_sites, prediction_data_dir=prediction_data_dir
    )

    return Path(
        predicted_output_base_file,
        mission_type,
        "cf_matrix",
        f"run_{run_ID}.npy",
    )


def get_aggregated_face_values_file(
    prediction_site, training_sites, mission_type, run_ID, prediction_data_dir
):
    predicted_output_base_file = get_predicted_output_base_file(
        prediction_site, training_sites, prediction_data_dir=prediction_data_dir
    )

    return Path(
        predicted_output_base_file,
        mission_type,
        "aggregated_face_values",
        f"run_{run_ID}.npy",
    )


def get_aggregated_raster_pred_file(
    training_sites, inference_site, run_ID, prediction_data_dir
):
    training_sites_str = get_training_sites_str(training_sites=training_sites)
    return Path(
        prediction_data_dir,
        "ortho_raster_predictions",
        f"{training_sites_str}_model_ortho_aggregated_raster",
        inference_site,
        f"run_{run_ID}.tif",
    )


# Step 5 functions


def get_unlabeled_crowns_file(site, data_dir):
    return Path(data_dir, "predicted-treecrowns", f"{site}.gpkg")


def get_oblique_images_folder(short_model_name):
    return {
        "chips": "/ofo-share/str-disp_drone-data-partial/str-disp_drone-data_imagery-missions/ChipsB/ChipsB_80m_2021_complete",
        "delta": "/ofo-share/str-disp_drone-data-partial/str-disp_drone-data_imagery-missions/DeltaB/DeltaB_80m",
        "valley": "/ofo-share/str-disp_drone-data-partial/str-disp_drone-data_imagery-missions/ValleyA/ValleyA_90m",
        "lassic": "/ofo-share/str-disp_drone-data-partial/str-disp_drone-data_imagery-missions/Lassic/Lassic_80m",
    }[short_model_name]


def get_labels_vis_folder(site_name, mission_type):
    return Path(
        VIS_ROOT,
        site_name,
        "rendered_labels_vis",
        mission_type,
    )


def get_inference_image_folder(site_name):
    return Path(
        DATA_ROOT,
        "per_site_processing",
        site_name,
        "03_training_data",
        "images_near_labels",
    )


def get_predicted_labeled_polygons_file(
    prediction_site, training_sites, mission_type, run_ID
):
    predicted_output_base_file = get_predicted_output_base_file(
        prediction_site, training_sites
    )

    return Path(
        predicted_output_base_file,
        mission_type,
        "predicted_labeled_polygons",
        f"run_{run_ID}.geojson",
    )


def get_predicted_vector_labels_filename(prediction_site, training_sites):
    return get_predicted_output_base_file(
        prediction_site=prediction_site, training_sites=training_sites
    ).with_suffix(".geojson")


def get_numpy_export_faces_texture_filename(prediction_site, training_sites):
    return get_predicted_output_base_file(
        prediction_site=prediction_site, training_sites=training_sites
    ).with_suffix(".npy")


def get_inference_raster_filename(inference_site):
    return Path(
        DATA_ROOT,
        "per_site_processing",
        inference_site,
        "02_photogrammetry",
        "exports",
        "orthos",
        f"{inference_site}.tif",
    )


def get_inference_chips_folder(inference_site):
    return Path(
        DATA_ROOT,
        "per_site_processing",
        inference_site,
        "04_model_predictions",
        "ortho_chipped_images",
    )
