import geopandas as gpd
from constants import (
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
    get_predicted_labeled_polygons_file,
    get_prediction_folder,
    get_subfolder_by_mission_type,
)
from geograypher.constants import PRED_CLASS_ID_KEY
from geograypher.entrypoints import aggregate_images, label_polygons
from geograypher.utils.prediction_metrics import compute_and_show_cf

SITE_NAMES = ["valley", "chips", "delta", "lassic"]

IDs_to_labels = get_IDs_to_labels()

for mission_type in ("MV-HN",):
    for site_name in ["lassic", "chips", "delta"]:
        for run_ID in ("00", "01", "02"):
            image_folder = get_image_folder(site_name)
            mesh_file = get_mesh_filename(site_name)
            cameras_file = get_cameras_filename(site_name)
            DTM_file = get_DTM_filename(site_name)
            mesh_transform_file = get_mesh_transform_filename(site_name)
            subset_images_folder = get_subfolder_by_mission_type(
                image_folder, mission_type=mission_type
            )

            training_sites = sorted(list(filter(lambda x: x != site_name, SITE_NAMES)))
            prediction_folder = get_prediction_folder(
                prediction_site=site_name,
                training_sites=training_sites,
                mission_type=mission_type,
                run_ID=run_ID,
            )
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

            aggregate_images(
                mesh_file=mesh_file,
                cameras_file=cameras_file,
                label_folder=prediction_folder,
                image_folder=image_folder,
                subset_images_folder=subset_images_folder,
                mesh_transform_file=mesh_transform_file,
                DTM_file=DTM_file,
                ROI=LABELS_FILENAME,
                IDs_to_labels=IDs_to_labels,
                aggregated_face_values_savefile=aggregated_face_values_file,
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
            # pred_polygons = gpd.read_file(predicted_labeled_polygons_file)
            # pred_polygons = pred_polygons.query("fire==@site_name")
            # pred_labels = pred_polygons[PRED_CLASS_ID_KEY].tolist()
            # gt_labels = pred_polygons[LABELS_COLUMN].tolist()
            # compute_and_show_cf(
            #    pred_labels=pred_labels,
            #    gt_labels=gt_labels,
            #    labels=list(IDs_to_labels.values()),
            #    savefile=figure_export_confusion_matrix_file,
            # )
