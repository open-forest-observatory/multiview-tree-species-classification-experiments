import geopandas as gpd
import numpy as np
from geograypher.segmentation import assemble_tiled_predictions
from geograypher.utils.geospatial import get_overlap_raster
from geograypher.utils.prediction_metrics import compute_and_show_cf

from constants import (ALL_SITE_NAMES, LABELS_COLUMN, LABELS_FILENAME,
                       get_aggregated_raster_pred_file,
                       get_figure_export_confusion_matrix_file,
                       get_IDs_to_labels, get_npy_export_confusion_matrix_file,
                       get_prediction_folder, get_training_raster_filename)

IDs_to_labels = get_IDs_to_labels()
num_classes = len(IDs_to_labels)
labels = list(IDs_to_labels.values())

for run_ID in ("00", "01", "02"):
    for test_site in ALL_SITE_NAMES:
        ortho_filename = get_training_raster_filename(test_site)

        training_sites = list(filter(lambda x: x != test_site, ALL_SITE_NAMES))
        prediction_folder = get_prediction_folder(
            test_site,
            training_sites=training_sites,
            mission_type="ortho",
            run_ID=run_ID,
        )

        figure_export_confusion_matrix_file = get_figure_export_confusion_matrix_file(
            test_site,
            training_sites=training_sites,
            mission_type="ortho",
            run_ID=run_ID,
        )
        npy_export_confusion_matrix_file = get_npy_export_confusion_matrix_file(
            test_site,
            training_sites=training_sites,
            mission_type="ortho",
            run_ID=run_ID,
        )

        pred_files = list(prediction_folder.glob("*"))
        aggregated_raster_pred_file = get_aggregated_raster_pred_file(
            training_sites, test_site, run_ID
        )

        assemble_tiled_predictions(
            raster_input_file=ortho_filename,
            pred_files=pred_files,
            num_classes=num_classes,
            class_savefile=aggregated_raster_pred_file,
        )
        class_fractions, IDs_in_original = get_overlap_raster(
            unlabeled_df=LABELS_FILENAME,
            classes_raster=aggregated_raster_pred_file,
            num_classes=num_classes,
        )
        pred_IDs = np.argmax(class_fractions, axis=1)

        pred_classes = [IDs_to_labels[pred_ID] for pred_ID in pred_IDs]

        gt_classes = gpd.read_file(LABELS_FILENAME)[LABELS_COLUMN][
            IDs_in_original
        ].tolist()
        compute_and_show_cf(
            pred_labels=pred_classes,
            gt_labels=gt_classes,
            labels=labels,
            cf_plot_savefile=figure_export_confusion_matrix_file,
            cf_np_savefile=npy_export_confusion_matrix_file,
        )
