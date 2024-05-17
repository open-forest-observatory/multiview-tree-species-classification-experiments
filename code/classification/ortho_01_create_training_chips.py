from pathlib import Path
from argparse import ArgumentParser

import geopandas as gpd
import numpy as np
from constants import (
    ALL_SITE_NAMES,
    CHIP_SIZE,
    LABELS_COLUMN,
    LABELS_FILENAME,
    TRAINING_IMGS_EXT,
    TRAINING_STRIDE,
    get_IDs_to_labels,
    get_labels_vis_folder,
    get_training_chips_folder,
    get_training_raster_filename,
)
from geograypher.segmentation import write_chips
from geograypher.utils.visualization import show_segmentation_labels


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--site-names", nargs="+", default=ALL_SITE_NAMES)
    parser.add_argument("--just-vis", action="store_true")
    args = parser.parse_args()
    return args


args = parse_args()

# # Create training chips
IDs_to_labels = get_IDs_to_labels()
# Interchange the keys and values
label_remap = {v: k for k, v in IDs_to_labels.items()}
class_names = np.unique(gpd.read_file(LABELS_FILENAME)[LABELS_COLUMN]).tolist()

for training_site in args.site_names:
    training_raster_filename = get_training_raster_filename(training_site=training_site)
    training_chips_folder = get_training_chips_folder(training_site=training_site)
    labels_vis_folder = get_labels_vis_folder(training_site, mission_type="ortho")

    if not args.just_vis:
        print(f"Writing {training_site}")
        # create the paired ortho chips and associated labels
        write_chips(
            raster_file=training_raster_filename,
            output_folder=training_chips_folder,
            chip_size=CHIP_SIZE,
            chip_stride=TRAINING_STRIDE,
            label_vector_file=LABELS_FILENAME,
            label_column=LABELS_COLUMN,
            label_remap=label_remap,
            output_suffix=TRAINING_IMGS_EXT,
            ROI_file=LABELS_FILENAME,
        )
    print(f"Showing {training_site}")
    # Visualize this training data
    show_segmentation_labels(
        label_folder=Path(training_chips_folder, "anns"),
        image_folder=Path(training_chips_folder, "imgs"),
        label_suffix=".png",
        image_suffix=TRAINING_IMGS_EXT,
        savefolder=labels_vis_folder,
        IDs_to_labels=IDs_to_labels,
    )
