from geograypher.entrypoints import label_polygons
from geograypher.utils.prediction_metrics import compute_and_show_cf
from geograypher.constants import PRED_CLASS_ID_KEY

import geopandas as gpd
from constants import (
    get_IDs_to_labels,
    get_mesh_filename,
    get_mesh_transform_filename,
    get_DTM_filename,
    get_predicted_labeled_polygons_file,
    get_aggregated_face_values_file,
    get_figure_export_confusion_matrix_file,
    LABELS_FILENAME,
    LABELS_COLUMN,
)

SITE_NAMES = ["valley", "chips", "delta", "lassic"]

IDs_to_labels = get_IDs_to_labels()

for mission_type in ("MV-LO", "MV-HN"):
    for site_name in SITE_NAMES:
        for run_ID in ("00", "01", "02"):
            mesh_file = get_mesh_filename(site_name)
            DTM_file = get_DTM_filename(site_name)
            mesh_transform_file = get_mesh_transform_filename(site_name)

            training_sites = sorted(list(filter(lambda x: x != site_name, SITE_NAMES)))
            aggregated_face_values_file = get_aggregated_face_values_file(
                site_name,
                training_sites=training_sites,
                mission_type=mission_type,
                run_ID=run_ID,
            )
            predicted_labeled_polygons_file = get_predicted_labeled_polygons_file(
                site_name,
                training_sites=training_sites,
                mission_type=mission_type,
                run_ID=run_ID,
            )
            figure_export_confusion_matrix_file = (
                get_figure_export_confusion_matrix_file(
                    site_name,
                    training_sites=training_sites,
                    mission_type=mission_type,
                    run_ID=run_ID,
                )
            )

            # label_polygons(
            #    mesh_file=mesh_file,
            #    mesh_transform_file=mesh_transform_file,
            #    aggregated_face_values_file=aggregated_face_values_file,
            #    DTM_file=DTM_file,
            #    ROI=LABELS_FILENAME,
            #    IDs_to_labels=IDs_to_labels,
            #    geospatial_polygons_to_label=LABELS_FILENAME,
            #    geospatial_polygons_labeled_savefile=predicted_labeled_polygons_file,
            # )
            pred_polygons = gpd.read_file(predicted_labeled_polygons_file)
            pred_polygons = pred_polygons.query("fire==@site_name")
            pred_labels = pred_polygons[PRED_CLASS_ID_KEY].tolist()
            gt_labels = pred_polygons[LABELS_COLUMN].tolist()
            compute_and_show_cf(
                pred_labels=pred_labels,
                gt_labels=gt_labels,
                labels=list(IDs_to_labels.values()),
                savefile=figure_export_confusion_matrix_file,
            )
