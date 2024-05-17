import os
from pathlib import Path

# Python utilities
# TODO modify these if you are setting up your own project
# Because of the difficulty of having a shared set of dependencies, we use multiple different conda
# environments for different parts of the pipeline
# Python interpreter for the segmentation_utils conda repository
MMSEG_UTILS_PYTHON = "/ofo-share/repos-david/conda/envs/mmseg-utils/bin/python"
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
PROJECT_ROOT = Path(__file__, "..", "..", "..").resolve()
DATA_ROOT = Path(PROJECT_ROOT, "data")
VIS_ROOT = Path(PROJECT_ROOT, "vis")
# SCRATCH_ROOT = Path(Path.home(), "scratch", "organized_str_disp_MVMT_experiments")

# Ground truth information
LABELS_FILENAME = Path(DATA_ROOT, "field_ref", "crown_labels.gpkg")
# LABELS_FILENAME = Path(
#    "/ofo-share/scratch-derek/organized-str-disp-MVMT-experiments/field_ref/crowns_drone_w_field_data_updated_nosnags.gpkg"
# )
LABELS_COLUMN = "species_observed"

ALL_SITE_NAMES = ["chips", "delta", "lassic", "valley"]

# Conversion between short and long names
LONG_SITE_NAME_DICT = {
    "valley": "ValleyA",
    "chips": "ChipsB",
    "delta": "DeltaB",
    "lassic": "Lassic",
}


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


def get_IDs_to_labels(with_ground=False):
    IDs_to_labels = {
        0: "ABCO",
        1: "CADE",
        2: "PILA",
        3: "PIPJ",
        4: "PSME",
    }
    if with_ground:
        IDs_to_labels[5] = "ground"

    return IDs_to_labels


def get_unlabeled_crowns_file(site):
    return Path(DATA_ROOT, "field_ref", "unlabeled_full_site_crowns", f"{site}.gpkg")


def get_image_folder(site_name, mission_type=None):
    image_folder = Path(
        DATA_ROOT,
        "per_site_processing",
        site_name,
        "01_images",
    )
    if mission_type is None:
        return image_folder
    return get_subfolder_by_mission_type(image_folder, mission_type=mission_type)


def get_mesh_filename(site_name):
    return Path(
        DATA_ROOT,
        "per_site_processing",
        site_name,
        "02_photogrammetry",
        f"{site_name}_mesh.ply",
    )


def get_cameras_filename(site_name):
    # The camera file exported from Metashape
    return Path(
        DATA_ROOT,
        "per_site_processing",
        site_name,
        "02_photogrammetry",
        f"{site_name}_cameras.xml",
    )


def get_mesh_transform_filename(site_name):
    return get_cameras_filename(site_name)


def get_DTM_filename(site_name):
    return Path(
        DATA_ROOT,
        "per_site_processing",
        site_name,
        "02_photogrammetry",
        f"{site_name}_DTM.tif",
    )


def get_labeled_mesh_filename(site_name):
    return Path(
        DATA_ROOT,
        "per_site_processing",
        site_name,
        "03_training_data",
        "MV",
        "labeled_mesh.ply",
    )


def get_training_chips_folder(training_site):
    return Path(
        DATA_ROOT,
        "per_site_processing",
        training_site,
        "03_training_data",
        "ortho",
    )


def get_oblique_images_folder(short_model_name):
    return {
        "chips": "/ofo-share/str-disp_drone-data-partial/str-disp_drone-data_imagery-missions/ChipsB/ChipsB_80m_2021_complete",
        "delta": "/ofo-share/str-disp_drone-data-partial/str-disp_drone-data_imagery-missions/DeltaB/DeltaB_80m",
        "valley": "/ofo-share/str-disp_drone-data-partial/str-disp_drone-data_imagery-missions/ValleyA/ValleyA_90m",
        "lassic": "/ofo-share/str-disp_drone-data-partial/str-disp_drone-data_imagery-missions/Lassic/Lassic_80m",
    }[short_model_name]


def get_subfolder_by_mission_type(folder, mission_type):
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


def get_render_folder(site_name, mission_type=None):
    render_folder = Path(
        DATA_ROOT,
        "per_site_processing",
        site_name,
        "03_training_data",
        "MV",
        "rendered_labels",
    )
    if mission_type is None:
        return render_folder
    else:
        return get_subfolder_by_mission_type(render_folder, mission_type=mission_type)


def get_subset_images_folder(site_name, mission_type=None):
    subset_images_folder = Path(
        DATA_ROOT,
        "per_site_processing",
        site_name,
        "03_training_data",
        "MV",
        "images",
    )
    if mission_type is None:
        return subset_images_folder
    else:
        return get_subfolder_by_mission_type(
            subset_images_folder, mission_type=mission_type
        )


def get_mesh_vis_file(site_name):
    return Path(VIS_ROOT, site_name, "mesh_vis.png").resolve()


def get_labels_vis_folder(site_name, mission_type):
    return Path(
        VIS_ROOT,
        site_name,
        "rendered_labels_vis",
        mission_type,
    )


def get_training_sites_str(training_sites):
    return "_".join(training_sites)


def get_formatted_training_data_folder(training_sites, mission_type):
    training_sites_str = get_training_sites_str(training_sites)
    return Path(
        DATA_ROOT,
        "models",
        "multi_site",
        mission_type + "_" + training_sites_str,
        "formatted_training_data",
    )


def get_aggregated_labels_folder(training_sites, mission_type):
    training_data_folder = get_formatted_training_data_folder(
        training_sites,
        mission_type=mission_type,
    )
    return Path(training_data_folder, "labels")


def get_aggregated_images_folder(training_sites, mission_type):
    training_data_folder = get_formatted_training_data_folder(
        training_sites, mission_type=mission_type
    )
    return Path(training_data_folder, "images")


def get_work_dir(training_sites, mission_type, run_ID="00"):
    training_data_folder = get_formatted_training_data_folder(
        training_sites, mission_type=mission_type
    ).parent
    return Path(training_data_folder, f"work_dir_{run_ID}")


def get_mmseg_style_training_folder(training_sites, mission_type):
    training_data_folder = get_formatted_training_data_folder(
        training_sites, mission_type=mission_type
    )
    named_folder = training_data_folder.parts[-2]
    return Path(training_data_folder, f"{named_folder}_mmseg_style")


def get_inference_image_folder(site_name):
    return Path(
        DATA_ROOT,
        "per_site_processing",
        site_name,
        "03_training_data",
        "images_near_labels",
    )


def get_prediction_folder(prediction_site, training_sites, mission_type, run_ID):
    training_sites_str = get_training_sites_str(training_sites=training_sites)
    return Path(
        DATA_ROOT,
        "per_site_processing",
        prediction_site,
        "04_model_predictions",
        f"{training_sites_str}_{mission_type}_model",
        f"run_{run_ID}",
    )


def get_predicted_output_base_file(prediction_site, training_sites):
    training_sites_str = get_training_sites_str(training_sites)
    return Path(
        DATA_ROOT,
        "per_site_processing",
        prediction_site,
        "05_processed_predictions",
        f"{training_sites_str}_model",
    )


def get_aggregated_face_values_file(
    prediction_site, training_sites, mission_type, run_ID
):
    predicted_output_base_file = get_predicted_output_base_file(
        prediction_site, training_sites
    )

    return Path(
        predicted_output_base_file,
        mission_type,
        "aggregated_face_values",
        f"run_{run_ID}.npy",
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


def get_figure_export_confusion_matrix_file(
    prediction_site, training_sites, mission_type, run_ID
):
    predicted_output_base_file = get_predicted_output_base_file(
        prediction_site, training_sites
    )

    return Path(
        predicted_output_base_file,
        mission_type,
        "cf_matrix",
        f"run_{run_ID}.svg",
    )


def get_npy_export_confusion_matrix_file(
    prediction_site, training_sites, mission_type, run_ID
):
    predicted_output_base_file = get_predicted_output_base_file(
        prediction_site, training_sites
    )

    return Path(
        predicted_output_base_file,
        mission_type,
        "cf_matrix",
        f"run_{run_ID}.npy",
    )


def get_predicted_vector_labels_filename(prediction_site, training_sites):
    return get_predicted_output_base_file(
        prediction_site=prediction_site, training_sites=training_sites
    ).with_suffix(".geojson")


def get_numpy_export_faces_texture_filename(prediction_site, training_sites):
    return get_predicted_output_base_file(
        prediction_site=prediction_site, training_sites=training_sites
    ).with_suffix(".npy")


def get_aggregated_raster_pred_file(training_sites, inference_site, run_ID):
    training_sites_str = get_training_sites_str(training_sites=training_sites)
    return Path(
        DATA_ROOT,
        "per_site_processing",
        inference_site,
        "05_processed_predictions",
        f"{training_sites_str}_model_ortho_aggregated_raster",
        f"run_{run_ID}.tif",
    )


def get_training_raster_filename(training_site):
    return Path(
        DATA_ROOT,
        "per_site_processing",
        training_site,
        "02_photogrammetry",
        f"{training_site}_ortho.tif",
    )


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
