import subprocess
from argparse import ArgumentParser
from pathlib import Path

from constants import (
    ALL_SITE_NAMES,
    INFERENCE_SCRIPT,
    MMSEG_PYTHON,
    get_image_folder,
    get_prediction_folder,
    get_subset_images_folder,
    get_training_chips_folder,
    get_work_dir,
)


def predict_model(
    mission_type, training_sites, test_site, run_ID, batch_size, full_site=False
):
    prediction_folder = get_prediction_folder(
        prediction_site=test_site,
        training_sites=training_sites,
        mission_type=mission_type,
        run_ID=run_ID,
    )
    if mission_type == "ortho":
        input_images = Path(get_training_chips_folder(training_site=test_site), "imgs")
        extension = ".png"
    elif full_site:
        input_images = get_image_folder(test_site, mission_type)
        extension = ".JPG"
    else:
        input_images = get_subset_images_folder(test_site, mission_type=mission_type)
        # Append the last part of the input images folder so the name matches
        # TODO figure out whether this is entirely required
        prediction_folder = Path(prediction_folder, input_images.parts[-1])
        extension = ".JPG"

    work_dir = get_work_dir(
        training_sites=training_sites, mission_type=mission_type, run_ID=run_ID
    )
    config_file = list(Path(work_dir).glob("*py"))[0]
    checkpoint_file = Path(work_dir, "iter_10000.pth")
    prediction_folder.mkdir(parents=True, exist_ok=True)
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
        default=("00", "01", "02"),
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
                )
