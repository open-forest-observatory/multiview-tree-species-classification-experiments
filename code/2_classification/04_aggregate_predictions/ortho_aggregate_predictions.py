import sys
from argparse import ArgumentParser
from pathlib import Path

import geopandas as gpd
import numpy as np
from geograypher.predictors import assemble_tiled_predictions
from geograypher.utils.geospatial import get_overlap_raster
from geograypher.utils.prediction_metrics import compute_and_show_cf

# Imports from the constants
constants_dir = str(Path(Path(__file__).parent, "..").resolve())
sys.path.append(constants_dir)
from constants import (
    ALL_SITE_NAMES,
    DEFAULT_INPUT_DATA_DIR,
    DEFAULT_PREDICTION_DATA_DIR,
    LABELS_COLUMN,
    get_aggregated_raster_pred_file,
    get_figure_export_confusion_matrix_file,
    get_IDs_to_labels,
    get_labels_filename,
    get_npy_export_confusion_matrix_file,
    get_ortho_filename,
    get_prediction_folder,
)


def main(
    input_data_dir,
    prediction_data_dir,
    test_site,
    run_ID,
    training_sites,
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
    # Where to save the confusion matrix figure
    figure_export_confusion_matrix_file = get_figure_export_confusion_matrix_file(
        test_site,
        training_sites=training_sites,
        mission_type="ortho",
        run_ID=run_ID,
        prediction_data_dir=prediction_data_dir,
    )
    # Where to save the numpy representation of the confusion matrix
    npy_export_confusion_matrix_file = get_npy_export_confusion_matrix_file(
        test_site,
        training_sites=training_sites,
        mission_type="ortho",
        run_ID=run_ID,
        prediction_data_dir=prediction_data_dir,
    )

    # Get the labels
    labels_filename = get_labels_filename(
        input_data_dir=input_data_dir, include_snag_class=include_snag_class
    )
    IDs_to_labels = get_IDs_to_labels(include_snag_class=include_snag_class)
    num_classes = len(IDs_to_labels)
    labels = list(IDs_to_labels.values())

    # List all the predicted files
    pred_files = list(prediction_folder.glob("*"))

    # Merge tiled predictions into one geospatial orthomosaic
    assemble_tiled_predictions(
        raster_input_file=ortho_filename,
        pred_files=pred_files,
        num_classes=num_classes,
        class_savefile=aggregated_raster_pred_file,
    )
    # Label the polygons using the predicted raster
    class_fractions, IDs_in_original = get_overlap_raster(
        unlabeled_df=labels_filename,
        classes_raster=aggregated_raster_pred_file,
        num_classes=num_classes,
    )
    # Get the highest weight class prediction per polygon
    pred_IDs = np.argmax(class_fractions, axis=1)

    # Turn this ID into a string label
    pred_classes = [IDs_to_labels[pred_ID] for pred_ID in pred_IDs]

    # Read the ground truth classes for the corresponding polygons
    gt_classes = gpd.read_file(labels_filename)[LABELS_COLUMN][IDs_in_original].tolist()
    # Compute and show the confusion matrix
    compute_and_show_cf(
        pred_labels=pred_classes,
        gt_labels=gt_classes,
        labels=labels,
        cf_plot_savefile=figure_export_confusion_matrix_file,
        cf_np_savefile=npy_export_confusion_matrix_file,
    )


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
                include_snag_class=args.include_snag_class,
            )
