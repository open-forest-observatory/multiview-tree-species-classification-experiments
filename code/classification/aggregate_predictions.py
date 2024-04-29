import argparse
import geopandas as gpd
from constants import (
    LABELS_COLUMN,
    LABELS_FILENAME,
    ALL_SITE_NAMES,
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
    get_npy_export_confusion_matrix_file,
    get_subfolder_by_mission_type,
    get_unlabeled_crowns_file,
)
from geograypher.constants import PRED_CLASS_ID_KEY
from geograypher.entrypoints import aggregate_images, label_polygons
from geograypher.utils.prediction_metrics import compute_and_show_cf


IMAGE_DOWNSAMPLE = 0.25
N_AGGREGATION_CLUSTERS = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fullsite-pred", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--label-polygons", action="store_true")
    parser.add_argument("--compute-accuracy", action="store_true")
    parser.add_argument(
        "--site-names", nargs="+", default=["valley", "chips", "delta", "lassic"]
    )
    parser.add_argument("--run-IDs", nargs="+", default=("00",))
    parser.add_argument("--mission-types", nargs="+", default=("MV-LO",))
    args = parser.parse_args()
    return args


args = parse_args()

IDs_to_labels = get_IDs_to_labels()

for run_ID in args.run_IDs:
    for mission_type in args.mission_types:
        for site_name in args.site_names:
            image_folder = get_image_folder(site_name)
            mesh_file = get_mesh_filename(site_name)
            cameras_file = get_cameras_filename(site_name)
            DTM_file = get_DTM_filename(site_name)
            mesh_transform_file = get_mesh_transform_filename(site_name)
            subset_images_folder = get_subfolder_by_mission_type(
                image_folder, mission_type=mission_type
            )

            if args.fullsite_pred:
                training_sites = ALL_SITE_NAMES
                ROI = get_unlabeled_crowns_file(site_name)
            else:
                training_sites = sorted(
                    list(filter(lambda x: x != site_name, ALL_SITE_NAMES))
                )
                ROI = LABELS_FILENAME
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
            npy_export_confusion_matrix_file = get_npy_export_confusion_matrix_file(
                site_name,
                training_sites=training_sites,
                mission_type=mission_type,
                run_ID=run_ID,
            )
            if args.fullsite_pred:
                geospatial_polygons_to_label = get_unlabeled_crowns_file(site_name)
            else:
                geospatial_polygons_to_label = LABELS_FILENAME

            if args.aggregate:
                aggregate_images(
                    mesh_file=mesh_file,
                    cameras_file=cameras_file,
                    label_folder=prediction_folder,
                    image_folder=image_folder,
                    subset_images_folder=subset_images_folder,
                    mesh_transform_file=mesh_transform_file,
                    DTM_file=DTM_file,
                    ROI=ROI,
                    IDs_to_labels=IDs_to_labels,
                    aggregated_face_values_savefile=aggregated_face_values_file,
                    aggregate_image_scale=IMAGE_DOWNSAMPLE,
                    n_aggregation_clusters=N_AGGREGATION_CLUSTERS,
                )
            if args.label_polygons:
                label_polygons(
                    mesh_file=mesh_file,
                    mesh_transform_file=mesh_transform_file,
                    aggregated_face_values_file=aggregated_face_values_file,
                    DTM_file=DTM_file,
                    ROI=ROI,
                    IDs_to_labels=IDs_to_labels,
                    geospatial_polygons_to_label=geospatial_polygons_to_label,
                    geospatial_polygons_labeled_savefile=predicted_labeled_polygons_file,
                    vis_mesh=True,
                )

            if args.compute_accuracy:
                pred_polygons = gpd.read_file(predicted_labeled_polygons_file)

                if not args.fullsite_pred:
                    pred_polygons = pred_polygons.query("fire==@site_name")

                pred_labels = pred_polygons[PRED_CLASS_ID_KEY].tolist()
                gt_labels = pred_polygons[LABELS_COLUMN].tolist()
                compute_and_show_cf(
                    pred_labels=pred_labels,
                    gt_labels=gt_labels,
                    labels=list(IDs_to_labels.values()),
                    cf_plot_savefile=figure_export_confusion_matrix_file,
                    cf_np_savefile=npy_export_confusion_matrix_file,
                )
