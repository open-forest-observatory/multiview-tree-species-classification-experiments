import sys
from argparse import ArgumentParser
from pathlib import Path

from geograypher.predictors import write_chips
from geograypher.utils.visualization import show_segmentation_labels

# Import from constants file
constants_dir = str(Path(Path(__file__).parent, "..").resolve())
sys.path.append(constants_dir)
from constants import (
    ALL_SITE_NAMES,
    CHIP_SIZE,
    DEFAULT_INPUT_DATA_DIR,
    DEFAULT_PREDICTION_DATA_DIR,
    LABELS_COLUMN,
    TRAINING_IMGS_EXT,
    TRAINING_STRIDE,
    get_IDs_to_labels,
    get_labels_filename,
    get_ortho_filename,
    get_ortho_training_data_folder,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--site-names", nargs="+", default=ALL_SITE_NAMES)
    parser.add_argument("--just-vis", action="store_true")
    parser.add_argument("--input-data-dir", default=DEFAULT_INPUT_DATA_DIR)
    parser.add_argument("--prediction-data-dir", default=DEFAULT_PREDICTION_DATA_DIR)
    parser.add_argument("--dont-include-snags", action="store_false")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # Get the labeled filename
    labels_filename = get_labels_filename(
        input_data_dir=args.input_data_dir,
        include_snag_class=not args.dont_include_snags,
    )
    print(f"Using labels from {labels_filename}")

    IDs_to_labels = get_IDs_to_labels()
    # Interchange the keys and values
    label_remap = {v: k for k, v in IDs_to_labels.items()}

    for site in args.site_names:
        # Get the filenames for each site
        ortho_filename = get_ortho_filename(
            site=site, input_data_dir=args.input_data_dir
        )
        labeled_chips_folder = get_ortho_training_data_folder(
            site=site, prediction_data_dir=args.prediction_data_dir
        )
        labeled_chips_vis_folder = get_ortho_training_data_folder(
            site=site,
            prediction_data_dir=args.prediction_data_dir,
            append_vis=True,
        )

        if not args.just_vis:
            print(f"Writing site: {site}")
            # create the paired ortho chips and associated labels
            write_chips(
                raster_file=ortho_filename,
                output_folder=labeled_chips_folder,
                chip_size=CHIP_SIZE,
                chip_stride=TRAINING_STRIDE,
                label_vector_file=labels_filename,
                label_column=LABELS_COLUMN,
                label_remap=label_remap,
                output_suffix=TRAINING_IMGS_EXT,
                ROI_file=labels_filename,
            )

        print(f"Showing {site}")
        # Visualize this training data
        show_segmentation_labels(
            label_folder=Path(labeled_chips_folder, "anns"),
            image_folder=Path(labeled_chips_folder, "imgs"),
            label_suffix=".png",
            image_suffix=TRAINING_IMGS_EXT,
            savefolder=labeled_chips_vis_folder,
            IDs_to_labels=IDs_to_labels,
        )
