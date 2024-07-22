import sys
from argparse import ArgumentParser
from pathlib import Path

import geopandas as gpd
from geograypher.constants import PRED_CLASS_ID_KEY
from geograypher.utils.prediction_metrics import compute_and_show_cf

# Imports from the constants
constants_dir = str(Path(Path(__file__).parent, "..").resolve())
sys.path.append(constants_dir)
from constants import (
    ALL_SITE_NAMES,
    DEFAULT_INPUT_DATA_DIR,
    DEFAULT_PREDICTION_DATA_DIR,
    LABELS_COLUMN,
    get_figure_export_confusion_matrix_file,
    get_IDs_to_labels,
    get_labels_filename,
    get_npy_export_confusion_matrix_file,
    get_predicted_labeled_polygons_file,
)


def main(
    site_name,
    training_sites,
    mission_type,
    run_ID,
    prediction_data_dir,
    input_data_dir,
    include_snag_class,
):
    # Get the filenames
    figure_export_confusion_matrix_file = get_figure_export_confusion_matrix_file(
        site_name,
        training_sites=training_sites,
        mission_type=mission_type,
        run_ID=run_ID,
        prediction_data_dir=prediction_data_dir,
    )
    npy_export_confusion_matrix_file = get_npy_export_confusion_matrix_file(
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
    ground_truth_polygons_file = get_labels_filename(
        input_data_dir=input_data_dir, include_snag_class=include_snag_class
    )

    # Read the ground truth polygons
    ground_truth_polygons = gpd.read_file(ground_truth_polygons_file)
    # Filter them to the appropriate site
    ground_truth_polygons = ground_truth_polygons.query("fire==@site_name")

    # Read the predicted polygons
    pred_polygons = gpd.read_file(predicted_labeled_polygons_file)
    # Convert the predicted polygons to the same CRS as the ground truth
    pred_polygons.to_crs(ground_truth_polygons.crs, inplace=True)

    ## Determine the most overlapping prediction for each gt tree

    # Do a spatial join of the ground truth and predictions
    gt_joined_with_preds = ground_truth_polygons.sjoin(pred_polygons)
    # Group by the groundtruth trees
    # modified from here https://www.reddit.com/r/learnpython/comments/pfy387/pandas_multiindex_item_and_date_select_most/
    grouped_by_gt_tree = gt_joined_with_preds.groupby(
        "observed_tree_id", sort=False, group_keys=True
    )
    # Get the predicted tree that best overlaps with the grountruth one
    gt_trees_with_best_overlapping_pred = grouped_by_gt_tree.apply(
        lambda x: x.iloc[x.area.argmax()], include_groups=False
    )

    # Get the groundtruth lables per polygon
    gt_labels = gt_trees_with_best_overlapping_pred[LABELS_COLUMN].tolist()
    # Get the predicted labels per polygon
    pred_labels = gt_trees_with_best_overlapping_pred[PRED_CLASS_ID_KEY].tolist()

    # Get the mapping between IDs and labels
    IDs_to_labels = get_IDs_to_labels(include_snag_class=include_snag_class)
    labels = list(IDs_to_labels.values()) + ["unknown"]

    # Compute a confusion matrix
    metrics = compute_and_show_cf(
        pred_labels=pred_labels,
        gt_labels=gt_labels,
        labels=labels,
        cf_plot_savefile=figure_export_confusion_matrix_file,
        cf_np_savefile=npy_export_confusion_matrix_file,
    )
    # print the metrics
    print(metrics)


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
        "--mission-types",
        nargs="+",
        default=("MV-LO", "MV-HN", "ortho"),
        help="Which missions to evaluate the accuracy of",
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

    for site_name in args.site_names:
        for run_ID in args.run_IDs:
            for mission_type in args.mission_types:
                print(site_name)
                main(
                    site_name=site_name,
                    training_sites=ALL_SITE_NAMES,  # TODO update
                    mission_type=mission_type,
                    run_ID=run_ID,
                    include_snag_class=args.include_snag_class,
                    input_data_dir=args.input_data_dir,
                    prediction_data_dir=args.prediction_data_dir,
                )
