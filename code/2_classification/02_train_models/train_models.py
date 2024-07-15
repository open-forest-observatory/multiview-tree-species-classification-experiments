# Standard library imports
import json
import os
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

# Imports from the constants
constants_dir = str(Path(Path(__file__).parent, "..").resolve())
sys.path.append(constants_dir)
from constants import (
    DEFAULT_INPUT_DATA_DIR,
    DEFAULT_PREDICTION_DATA_DIR,
    FOLDER_TO_CITYSCAPES_SCRIPT,
    MMSEG_PYTHON,
    SEGMENTATION_UTILS_PYTHON,
    TRAIN_SCRIPT,
    TRAINING_IMGS_EXT,
    get_aggregated_images_folder,
    get_aggregated_labels_folder,
    get_IDs_to_labels,
    get_mmseg_style_training_folder,
    get_ortho_training_data_folder,
    get_render_folder,
    get_subset_images_folder,
    get_work_dir,
)


def train_model(
    mission_type, training_sites, run_ID, input_data_dir, prediction_data_dir
):
    # Get folder path for agregated images and labels
    aggregated_images_folder = get_aggregated_images_folder(
        prediction_data_dir=prediction_data_dir,
        training_sites=training_sites,
        mission_type=mission_type,
    )
    aggregated_labels_folder = get_aggregated_labels_folder(
        prediction_data_dir=prediction_data_dir,
        training_sites=training_sites,
        mission_type=mission_type,
    )
    # Ensure that these folders exist
    Path(aggregated_images_folder).mkdir(exist_ok=True, parents=True)
    Path(aggregated_labels_folder).mkdir(exist_ok=True, parents=True)

    # Determine the IDs to labels
    IDs_to_labels = get_IDs_to_labels()
    class_names = list(IDs_to_labels.values())
    class_names_str = " ".join(class_names)

    # Symlink the relevant training sites into one folder
    for training_site in training_sites:
        if mission_type == "ortho":
            training_chips_folder = get_ortho_training_data_folder(
                site=training_site, prediction_data_dir=prediction_data_dir
            )
            render_folder = Path(training_chips_folder, "anns")
            image_folder = Path(training_chips_folder, "imgs")
            image_ext = TRAINING_IMGS_EXT
        else:
            render_folder = get_render_folder(
                site_name=training_site, mission_type=mission_type
            )
            image_folder = get_subset_images_folder(
                site_name=training_site, mission_type=mission_type
            )
            image_ext = ".JPG"

        # Determine the location to link to for each labels and images
        output_render_folder = Path(aggregated_labels_folder, training_site)
        output_image_folder = Path(aggregated_images_folder, training_site)

        # Remove the old data by either unlinking or actually removing
        if os.path.islink(output_render_folder):
            output_render_folder.unlink(missing_ok=True)
        else:
            shutil.rmtree(output_render_folder, ignore_errors=True)

        if os.path.islink(output_image_folder):
            output_image_folder.unlink(missing_ok=True)
        else:
            shutil.rmtree(output_image_folder, ignore_errors=True)

        # TODO could this just be a symlink?
        print(f"copying {render_folder} to {output_render_folder}")
        shutil.copytree(render_folder, output_render_folder)
        print(f"copying {image_folder} to {output_image_folder}")
        shutil.copytree(image_folder, output_image_folder)

    # Format the data according to how MMSEG expects
    mmseg_style_training_folder = get_mmseg_style_training_folder(
        prediction_data_dir=args.prediction_data_dir,
        training_sites=training_sites,
        mission_type=mission_type,
    )
    # Build the command string
    formatting_run_str = (
        f"{SEGMENTATION_UTILS_PYTHON} {FOLDER_TO_CITYSCAPES_SCRIPT} --images-folder {aggregated_images_folder}"
        + f" --labels-folder {aggregated_labels_folder} --output-folder {mmseg_style_training_folder}"
        + f" --image-ext {image_ext} --classes {class_names_str} --remove-old"
    )
    print(formatting_run_str)
    # Run it as a subprocess
    subprocess.run(
        formatting_run_str,
        shell=True,
    )

    # Get the work dir to save the model
    work_dir = get_work_dir(
        prediction_data_dir=args.prediction_data_dir,
        training_sites=training_sites,
        mission_type=mission_type,
        run_ID=run_ID,
    )
    # Identify the config file as the only python file in the
    config_file = list(Path(mmseg_style_training_folder).glob("*py"))[0]
    # Actually train the model
    subprocess.run(
        f"{MMSEG_PYTHON} {TRAIN_SCRIPT} {config_file} --work-dir {work_dir}",
        shell=True,
    )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--mission-types",
        nargs="+",
        default=("ortho", "MV-HN", "MV-LO"),
        help="Which type of data to use for the model",
    )
    parser.add_argument(
        "--run-IDs",
        nargs="+",
        default=("00",),
        help="Train a model with each of these tags for each configuration",
    )
    parser.add_argument(
        "--training-site-sets",
        nargs="+",
        default=[
            ["chips", "delta", "lassic", "valley"],
            ["chips", "delta", "lassic"],
            ["chips", "delta", "valley"],
            ["chips", "lassic", "valley"],
            ["delta", "lassic", "valley"],
        ],
        help="Train one model for each set of training sites in the list",
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

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    for run_ID in args.run_IDs:
        for training_site_set in args.training_site_sets:
            for mission_type in args.mission_types:
                train_model(
                    mission_type=mission_type,
                    training_sites=training_site_set,
                    run_ID=run_ID,
                    input_data_dir=args.input_data_dir,
                    prediction_data_dir=args.prediction_data_dir,
                )
