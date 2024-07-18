import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

from geograypher.cameras import MetashapeCameraSet
from geograypher.predictors.ortho_segmentor import write_chips
from geograypher.utils.visualization import show_segmentation_labels

constants_dir = str(Path(Path(__file__).parent, "..").resolve())
sys.path.append(constants_dir)
from constants import (
    ALL_SITE_NAMES,
    CHIP_SIZE,
    DEFAULT_INPUT_DATA_DIR,
    DEFAULT_PREDICTION_DATA_DIR,
    INFERENCE_SCRIPT,
    INFERENCE_STRIDE,
    MMSEG_PYTHON,
    get_cameras_filename,
    get_IDs_to_labels,
    get_image_folder,
    get_labels_filename,
    get_MV_images_subset_folder,
    get_ortho_filename,
    get_ortho_prediction_data_folder,
    get_prediction_folder,
    get_subfolder_by_mission_type,
    get_work_dir,
)


def predict_model(
    mission_type,
    training_sites,
    test_site,
    run_ID,
    batch_size,
    prediction_data_dir,
    input_data_dir,
    full_site=False,
):
    if mission_type == "ortho":
        ortho_filename = get_ortho_filename(
            site=test_site, input_data_dir=input_data_dir
        )
        input_images = get_ortho_prediction_data_folder(
            site=test_site, prediction_data_dir=prediction_data_dir
        )
        # The first step is to chip the images
        # If full site, chip the whole file
        # if not, we need to just get an ROI
        if full_site:
            # No ROI, just use everything
            ROI_file = None
        else:
            # including the snag class will give a slightly larger ROI
            ROI_file = get_labels_filename(
                input_data_dir=input_data_dir, include_snag_class=True
            )

        write_chips(
            raster_file=ortho_filename,
            output_folder=input_images,
            chip_size=CHIP_SIZE,
            chip_stride=INFERENCE_STRIDE,
            ROI_file=ROI_file,
            remove_old=True,
        )
    else:
        # Folder to all the raw images
        all_images_folder = get_image_folder(
            site_name=test_site,
            input_data_dir=input_data_dir,
        )
        # Path to the cameras file from metashape
        cameras_file = get_cameras_filename(
            site_name=test_site, input_data_dir=input_data_dir
        )
        # Create a cameras object
        camera_set = MetashapeCameraSet(
            camera_file=cameras_file, image_folder=all_images_folder
        )

        # If we don't want images from the full site, subset the camera set to only those within the ROI
        if not full_site:
            ROI_file = get_labels_filename(
                input_data_dir=input_data_dir, include_snag_class=True
            )
            # TODO make this radius configurable
            camera_set = camera_set.get_subset_ROI(ROI=ROI_file, buffer_radius=100)

        # Save out the subset of images
        subset_folder = get_MV_images_subset_folder(
            site=test_site, prediction_data_dir=prediction_data_dir
        )
        camera_set.save_images(subset_folder)

        # Append the mission subfolder
        input_images = Path(
            subset_folder,
            get_subfolder_by_mission_type(
                site_name=test_site, mission_type=mission_type
            ),
        )

    # Get folder to write predictions to
    prediction_folder = get_prediction_folder(
        prediction_site=test_site,
        training_sites=training_sites,
        mission_type=mission_type,
        run_ID=run_ID,
        prediction_data_dir=prediction_data_dir,
    )
    # If the prediction folder exists alread, remove it
    if prediction_folder.is_dir():
        shutil.rmtree(prediction_folder)
    # Make the prediction folder
    prediction_folder.mkdir(parents=True, exist_ok=True)

    # Get folder where model is
    work_dir = get_work_dir(
        prediction_data_dir=prediction_data_dir,
        training_sites=training_sites,
        mission_type=mission_type,
        run_ID=run_ID,
    )
    # Get the config file
    config_file = list(Path(work_dir).glob("*py"))[0]
    # TODO this assumes we're training for 10k iters, we should make this more flexible
    checkpoint_file = Path(work_dir, "iter_10000.pth")
    # Extension of the images we're going to do inference on
    # TODO we should make this more robust
    extension = ".JPG"

    # Create the command that we're going to run
    run_str = (
        f"{MMSEG_PYTHON} {INFERENCE_SCRIPT} {config_file} "
        + f"{checkpoint_file} {input_images} {prediction_folder} --batch-size {batch_size}"
        + f" --extension {extension}"
    )
    print(run_str)
    subprocess.run(
        run_str,
        shell=True,
    )

    # Visualize a subset of the predictions
    vis_folder = Path(prediction_folder.parent, prediction_folder.parts[-1] + "_vis")
    IDs_to_labels = get_IDs_to_labels()
    show_segmentation_labels(
        label_folder=prediction_folder,
        image_folder=input_images,
        savefolder=vis_folder,
        IDs_to_labels=IDs_to_labels,
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
        "--mission-types",
        nargs="+",
        default=("ortho", "MV-HN", "MV-LO"),
        help="The mission types to generate predictions on",
    )
    parser.add_argument(
        "--batch-size",
        default=2,
        type=int,
        help="The number of images to run inference on at once",
    )
    parser.add_argument(
        "--full-site",
        action="store_true",
        help="Whether to run inference on the full site or just the region near the labeled polygons",
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
        for test_site in args.site_names:
            for mission_type in args.mission_types:
                predict_model(
                    mission_type=mission_type,
                    training_sites=ALL_SITE_NAMES,
                    test_site=test_site,
                    run_ID=run_ID,
                    batch_size=args.batch_size,
                    full_site=args.full_site,
                    prediction_data_dir=args.prediction_data_dir,
                    input_data_dir=args.input_data_dir,
                )
