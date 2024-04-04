# Standard library imports
import subprocess
import json
import shutil
import sys
import os

# Contributed library imports
import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from pathlib import Path

# Imports from the constants
sys.path.append("../..")
from constants import (
    get_aggregated_images_folder,
    get_aggregated_labels_folder,
    get_subset_images_folder,
    get_render_folder,
    get_mmseg_style_training_folder,
    get_work_dir,
    get_training_chips_folder,
    get_IDs_to_labels,
    MMSEG_PYTHON,
    TRAIN_SCRIPT,
    TRAINING_IMGS_EXT,
    MMSEG_UTILS_PYTHON,
    FOLDER_TO_CITYSCAPES_SCRIPT,
)


def train_model(mission_type, training_sites, run_ID):
    aggregated_images_folder = get_aggregated_images_folder(
        training_sites=training_sites, mission_type=mission_type
    )
    aggregated_labels_folder = get_aggregated_labels_folder(
        training_sites=training_sites, mission_type=mission_type
    )

    # We need to merge all the imagery together
    # It should be as simple as creating the folders and symlinking the existing ones into it
    Path(aggregated_images_folder).mkdir(exist_ok=True, parents=True)
    Path(aggregated_labels_folder).mkdir(exist_ok=True, parents=True)

    if mission_type == "ortho":
        IDs_to_labels = get_IDs_to_labels()
    else:
        # Compute class names
        all_IDs_to_labels = []
        for training_site in training_sites:
            render_folder = get_render_folder(training_site)
            IDs_to_labels_file = Path(render_folder, "IDs_to_labels.json")
            with open(IDs_to_labels_file, "r") as infile:
                all_IDs_to_labels.append(json.load(infile))

        for i in range(len(all_IDs_to_labels) - 1):
            if all_IDs_to_labels[i] != all_IDs_to_labels[i + 1]:
                raise ValueError("Different IDs to labels")
        IDs_to_labels = all_IDs_to_labels[0]

    class_names = list(IDs_to_labels.values())
    class_names_str = " ".join(class_names)

    # Copy the relevant training sites into one folder
    for training_site in training_sites:
        if mission_type == "ortho":
            training_chips_folder = get_training_chips_folder(
                training_site=training_site
            )
            render_folder = Path(training_chips_folder, "anns")
            image_folder = Path(training_chips_folder, "imgs")
            image_ext = TRAINING_IMGS_EXT
        else:
            render_folder = get_render_folder(training_site, mission_type=mission_type)
            image_folder = get_subset_images_folder(
                training_site, mission_type=mission_type
            )
            image_ext = ".JPG"

        output_render_folder = Path(aggregated_labels_folder, training_site)
        output_image_folder = Path(aggregated_images_folder, training_site)

        if os.path.islink(output_render_folder):
            output_render_folder.unlink(missing_ok=True)
        else:
            shutil.rmtree(output_render_folder, ignore_errors=True)

        if os.path.islink(output_image_folder):
            output_image_folder.unlink(missing_ok=True)
        else:
            shutil.rmtree(output_image_folder, ignore_errors=True)

        print(f"copying {render_folder} to {output_render_folder}")
        shutil.copytree(render_folder, output_render_folder)
        print(f"copying {image_folder} to {output_image_folder}")
        shutil.copytree(image_folder, output_image_folder)

    # Format the data according to how MMSEG expects
    mmseg_style_training_folder = get_mmseg_style_training_folder(
        training_sites=training_sites, mission_type=mission_type
    )
    formatting_run_str = (
        f"{MMSEG_UTILS_PYTHON} {FOLDER_TO_CITYSCAPES_SCRIPT} --images-folder {aggregated_images_folder}"
        + f" --labels-folder {aggregated_labels_folder} --output-folder {mmseg_style_training_folder}"
        + f" --image-ext {image_ext} --classes {class_names_str} --remove-old"
    )
    print(formatting_run_str)
    subprocess.run(
        formatting_run_str,
        shell=True,
    )

    # Get the work dir to save the model
    work_dir = get_work_dir(
        training_sites=training_sites, mission_type=mission_type, run_ID=run_ID
    )
    # Identify the config file as the only python file in the
    config_file = list(Path(mmseg_style_training_folder).glob("*py"))[0]
    # Actually train the model
    subprocess.run(
        f"{MMSEG_PYTHON} {TRAIN_SCRIPT} {config_file} --work-dir {work_dir}",
        shell=True,
    )


# ALL_MISSION_TYPES = ("MV-HN", "MV-LO")
ALL_MISSION_TYPES = ("ortho",)
ALL_TRAINING_SITES = (
    ["chips", "delta", "lassic", "valley"],
    ["chips", "delta", "lassic"],
    ["chips", "delta", "valley"],
    ["chips", "lassic", "valley"],
    ["delta", "lassic", "valley"],
)
for run_ID in ("00", "01", "02"):
    for training_sites in ALL_TRAINING_SITES:
        for mission_type in ALL_MISSION_TYPES:
            train_model(
                mission_type=mission_type,
                training_sites=training_sites,
                run_ID=run_ID,
            )
