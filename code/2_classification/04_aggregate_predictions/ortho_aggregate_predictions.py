import sys
from argparse import ArgumentParser
from pathlib import Path

import geopandas as gpd
import numpy as np
from geograypher.constants import PRED_CLASS_ID_KEY
from geograypher.predictors import assemble_tiled_predictions
from geograypher.utils.files import ensure_containing_folder
from geograypher.utils.geospatial import get_overlap_raster

# Imports from the constants
constants_dir = str(Path(Path(__file__).parent, "..").resolve())
sys.path.append(constants_dir)
from constants import (
    ALL_SITE_NAMES,
    DEFAULT_INPUT_DATA_DIR,
    DEFAULT_PREDICTION_DATA_DIR,
    get_aggregated_raster_pred_file,
    get_IDs_to_labels,
    get_labels_filename,
    get_ortho_filename,
    get_predicted_labeled_polygons_file,
    get_prediction_folder,
    get_unlabeled_crowns_file,
)


def main(
    input_data_dir,
    prediction_data_dir,
    test_site,
    run_ID,
    training_sites,
    full_site_prediction,
    aggregate_raster,
    label_polygons,
    include_snag_class,
):

    # Get input folder and file names
    ortho_filename = get_ortho_filename(test_site, input_data_dir=input_data_dir)
    # Where the predictions per chip are
    prediction_folder = get_prediction_folder(
        test_site,
        training_sites=training_sites,
        mission_type="ortho",
        run_ID=run_ID,
        prediction_data_dir=prediction_data_dir,
    )
    # Where to write the aggregated raster file
    aggregated_raster_pred_file = get_aggregated_raster_pred_file(
        training_sites,
        test_site,
        run_ID,
        prediction_data_dir=prediction_data_dir,
    )
    predicted_labeled_polygons_file = get_predicted_labeled_polygons_file(
        test_site,
        training_sites=training_sites,
        mission_type="ortho",
        run_ID=run_ID,
        prediction_data_dir=prediction_data_dir,
    )

    # Get the appropriate polygons to label
    if full_site_prediction:
        unlabeled_polygons_file = get_unlabeled_crowns_file(
            site=test_site, input_data_dir=input_data_dir
        )
    else:
        unlabeled_polygons_file = get_labels_filename(
            input_data_dir=input_data_dir, include_snag_class=include_snag_class
        )

    # Aggregate the individual predictions to one raster with the most common class per pixel
    if aggregate_raster:
        # Get the labels
        IDs_to_labels = get_IDs_to_labels(include_snag_class=include_snag_class)
        num_classes = len(IDs_to_labels)

        # List all the predicted files
        pred_files = list(prediction_folder.glob("*"))

        # Merge tiled predictions into one geospatial orthomosaic
        assemble_tiled_predictions(
            raster_input_file=ortho_filename,
            pred_files=pred_files,
            num_classes=num_classes,
            class_savefile=aggregated_raster_pred_file,
        )

    # Classify polygons
    if label_polygons:
        # Label the polygons using the predicted raster
        class_fractions, polygon_IDs_in_original = get_overlap_raster(
            unlabeled_df=unlabeled_polygons_file,
            classes_raster=aggregated_raster_pred_file,
            num_classes=num_classes,
        )
        # Get the highest weight class prediction per polygon
        pred_IDs = np.argmax(class_fractions, axis=1)

        # Turn this ID into a string label
        pred_classes = [IDs_to_labels[pred_ID] for pred_ID in pred_IDs]

        # Add the predicted classes to the dataframe
        polygons = gpd.read(unlabeled_polygons_file)
        # Create a new column for the predicted class and set all elements to unkownn
        polygons[PRED_CLASS_ID_KEY] = "unknown"
        # Set any polygons that we have a prediction for to the predicted class
        polygons[PRED_CLASS_ID_KEY][polygon_IDs_in_original] = pred_classes
        # Ensure that the export folder exists
        ensure_containing_folder(predicted_labeled_polygons_file)
        # Export the labeled polygons
        polygons.to_file(predicted_labeled_polygons_file)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--site-names",
        nargs="+",
        default=ALL_SITE_NAMES,
        help="Sites to generate prediction on",
    )
    parser.add_argument(
        "--run-IDs",
        nargs="+",
        default=("00",),
        help="The run IDs to generate predictions from",
    )
    parser.add_argument(
        "--full-site-prediction",
        action="store_true",
        help="Should predictions for the full site be generated",
    )
    parser.add_argument(
        "--aggregate-raster",
        action="store_true",
        help="Run the step to aggregate individual chipped predictions to a single raster",
    )
    parser.add_argument(
        "--label-polygons",
        action="store_true",
        help="Run the step to label polygons from the aggregated raster.",
    )
    parser.add_argument(
        "--input-data-dir",
        default=DEFAULT_INPUT_DATA_DIR,
        help="Where to find the input data, e.g. images, photogrammetry, field reference",
    )
    parser.add_argument(
        "--prediction-data-dir",
        default=DEFAULT_PREDICTION_DATA_DIR,
        help="Where to find and update ML prediction data",
    )
    parser.add_argument(
        "--include-snag-class",
        action="store_true",
        help="Whether to include the snag class in the labels",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    for run_ID in args.run_IDs:
        for test_site in args.site_names:
            main(
                input_data_dir=args.input_data_dir,
                prediction_data_dir=args.prediction_data_dir,
                test_site=test_site,
                training_sites=ALL_SITE_NAMES,  # TODO update
                run_ID=run_ID,
                full_site_prediction=args.full_site_prediction,
                aggregate_raster=args.aggregate_raster,
                label_polygons=args.label_polygons,
                include_snag_class=args.include_snag_class,
            )
