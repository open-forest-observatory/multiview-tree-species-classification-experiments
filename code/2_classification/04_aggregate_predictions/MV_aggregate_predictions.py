import argparse
import sys
from pathlib import Path

from geograypher.entrypoints import aggregate_images, label_polygons

constants_dir = str(Path(Path(__file__).parent, "..").resolve())
sys.path.append(constants_dir)
from constants import (
    AGGREGATE_IMAGE_SCALE,
    ALL_SITE_NAMES,
    DEFAULT_INPUT_DATA_DIR,
    DEFAULT_PREDICTION_DATA_DIR,
    N_AGGREGATION_CLUSTERS,
    get_aggregated_face_values_file,
    get_cameras_filename,
    get_DTM_filename,
    get_IDs_to_labels,
    get_image_folder,
    get_labels_filename,
    get_mesh_filename,
    get_mesh_transform_filename,
    get_predicted_labeled_polygons_file,
    get_prediction_folder,
    get_unlabeled_crowns_file,
)


def main(
    site_name,
    fullsite_pred,
    include_snag_class,
    input_data_dir,
    prediction_data_dir,
    aggregate,
    classify_polygons,
    vis,
):
    # Get IDs to labels mapping
    IDs_to_labels = get_IDs_to_labels(include_snag_class=include_snag_class)

    # Get input folder and file paths
    image_folder = get_image_folder(site_name=site_name, input_data_dir=input_data_dir)
    subset_images_folder = get_image_folder(
        site_name=site_name, input_data_dir=input_data_dir, mission_type=mission_type
    )
    mesh_file = get_mesh_filename(site_name=site_name, input_data_dir=input_data_dir)
    cameras_file = get_cameras_filename(
        site_name=site_name, input_data_dir=input_data_dir
    )
    DTM_file = get_DTM_filename(site_name=site_name, input_data_dir=input_data_dir)
    mesh_transform_file = get_mesh_transform_filename(
        site_name=site_name, input_data_dir=input_data_dir
    )

    if fullsite_pred:
        # If we're using the full site, assume the model was trained on all input sites
        training_sites = ALL_SITE_NAMES
        # Set the ROI to be the bounds of the unlabled crowns
        unlabeled_polygons_file = get_unlabeled_crowns_file(
            site=site_name, input_data_dir=input_data_dir
        )
        ROI = unlabeled_polygons_file
        geospatial_polygons_to_label = unlabeled_polygons_file
    else:
        # If we're not doing a full site prediction, assume the model was trained on all other sites
        training_sites = sorted(list(filter(lambda x: x != site_name, ALL_SITE_NAMES)))
        # Set the ROI to be the bounds of just the labled crowns
        labeled_polygons_file = get_labels_filename(
            input_data_dir=input_data_dir, include_snag_class=include_snag_class
        )
        ROI = labeled_polygons_file
        geospatial_polygons_to_label = labeled_polygons_file

    # Get the folder where predictions are written to
    # Go one folder up since it needs to match the images structure
    # TODO see if this can be made more robust so predictions from another mission aren't added
    prediction_folder = get_prediction_folder(
        prediction_site=site_name,
        training_sites=training_sites,
        run_ID=run_ID,
        mission_type=mission_type,
        prediction_data_dir=prediction_data_dir,
    )

    aggregated_face_values_file = get_aggregated_face_values_file(
        site_name,
        training_sites=training_sites,
        mission_type=mission_type,
        run_ID=run_ID,
        prediction_data_dir=prediction_data_dir,
    )
    predicted_labeled_polygons_file = get_predicted_labeled_polygons_file(
        site_name,
        training_sites=training_sites,
        mission_type=mission_type,
        run_ID=run_ID,
        prediction_data_dir=prediction_data_dir,
    )

    if aggregate:
        # Aggregate predictions onto the mesh and save out the per-face result
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
            aggregate_image_scale=AGGREGATE_IMAGE_SCALE,
            n_aggregation_clusters=N_AGGREGATION_CLUSTERS,
            vis=vis,
        )

    if classify_polygons:
        # Label geospatial polygons based on the labeled faces of the mesh
        label_polygons(
            mesh_file=mesh_file,
            mesh_transform_file=mesh_transform_file,
            aggregated_face_values_file=aggregated_face_values_file,
            DTM_file=DTM_file,
            ROI=ROI,
            IDs_to_labels=IDs_to_labels,
            geospatial_polygons_to_label=geospatial_polygons_to_label,
            geospatial_polygons_labeled_savefile=predicted_labeled_polygons_file,
            vis_mesh=vis,
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fullsite-pred", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--classify-polygons", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--site-names", nargs="+", default=ALL_SITE_NAMES)
    parser.add_argument("--run-IDs", nargs="+", default=("00",))
    parser.add_argument("--mission-types", nargs="+", default=("MV-LO",))
    parser.add_argument(
        "--dont-include-snags", action="store_true", help="Include the snag class"
    )
    parser.add_argument(
        "--input-data-dir",
        default=DEFAULT_INPUT_DATA_DIR,
        help="Where the input data, such as photogrammetry produts and field reference is",
    )
    parser.add_argument(
        "--prediction-data-dir",
        default=DEFAULT_PREDICTION_DATA_DIR,
        help="where to write the output data",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    for run_ID in args.run_IDs:
        for mission_type in args.mission_types:
            for site_name in args.site_names:
                main(
                    site_name=site_name,
                    include_snag_class=not args.dont_include_snags,
                    fullsite_pred=args.fullsite_pred,
                    input_data_dir=args.input_data_dir,
                    prediction_data_dir=args.prediction_data_dir,
                    aggregate=args.aggregate,
                    classify_polygons=args.classify_polygons,
                    vis=args.vis,
                )
