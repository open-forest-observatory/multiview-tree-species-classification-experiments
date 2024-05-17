import argparse

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from constants import (
    ALL_SITE_NAMES,
    LABELS_COLUMN,
    LABELS_FILENAME,
    get_aggregated_face_values_file,
    get_cameras_filename,
    get_DTM_filename,
    get_figure_export_confusion_matrix_file,
    get_IDs_to_labels,
    get_image_folder,
    get_mesh_filename,
    get_mesh_transform_filename,
    get_npy_export_confusion_matrix_file,
    get_predicted_labeled_polygons_file,
    get_prediction_folder,
    get_subfolder_by_mission_type,
    get_unlabeled_crowns_file,
)
from geograypher.constants import PRED_CLASS_ID_KEY
from geograypher.entrypoints import aggregate_images, label_polygons
from geograypher.utils.prediction_metrics import compute_and_show_cf

MISSION_TYPE = "MV-LO"
RUN_ID = "00"

CFs = []
accuracies = []
for site_name in ["valley"]:
    image_folder = get_image_folder(site_name)
    mesh_file = get_mesh_filename(site_name)
    cameras_file = get_cameras_filename(site_name)
    DTM_file = get_DTM_filename(site_name)
    mesh_transform_file = get_mesh_transform_filename(site_name)
    subset_images_folder = get_subfolder_by_mission_type(
        image_folder, mission_type=MISSION_TYPE
    )

    training_sites = ALL_SITE_NAMES
    ROI = get_unlabeled_crowns_file(site_name)

    prediction_folder = get_prediction_folder(
        prediction_site=site_name,
        training_sites=training_sites,
        mission_type=MISSION_TYPE,
        run_ID=RUN_ID,
    )
    aggregated_face_values_file = get_aggregated_face_values_file(
        site_name,
        training_sites=training_sites,
        mission_type=MISSION_TYPE,
        run_ID=RUN_ID,
    )
    predicted_labeled_polygons_file = get_predicted_labeled_polygons_file(
        site_name,
        training_sites=training_sites,
        mission_type=MISSION_TYPE,
        run_ID=RUN_ID,
    )
    figure_export_confusion_matrix_file = get_figure_export_confusion_matrix_file(
        site_name,
        training_sites=training_sites,
        mission_type=MISSION_TYPE,
        run_ID=RUN_ID,
    )
    npy_export_confusion_matrix_file = get_npy_export_confusion_matrix_file(
        site_name,
        training_sites=training_sites,
        mission_type=MISSION_TYPE,
        run_ID=RUN_ID,
    )

    polygon_preds = gpd.read_file(predicted_labeled_polygons_file)
    print(polygon_preds.keys())
    polygon_preds.plot(PRED_CLASS_ID_KEY, vmin=-0.5, vmax=9.5, legend=True)
    plt.show()
    # all_labels = np.unique(polygon_preds["species_observed"].tolist())
    # print(polygon_preds)
    # polygon_preds = polygon_preds.query("fire==@site_name")
    # polygon_preds.plot(PRED_CLASS_ID_KEY, vmin=-0.5, vmax=9.5, legend=True)

    # ret = compute_and_show_cf(
    #    pred_labels=polygon_preds[PRED_CLASS_ID_KEY].tolist(),
    #    gt_labels=polygon_preds["species_observed"],
    #    vis=False,
    #    labels=all_labels,
    # )
    # CFs.append(ret[0])
    # accuracies.append(ret[2])
total_CF = np.sum(CFs, axis=0)
print(np.sum(np.eye(5) * total_CF) / np.sum(total_CF))
print(accuracies)
