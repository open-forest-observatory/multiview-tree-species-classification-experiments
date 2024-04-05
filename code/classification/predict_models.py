from constants import (
    get_prediction_folder,
    get_subset_images_folder,
    get_image_folder,
    get_work_dir,
    get_training_chips_folder,
    MMSEG_PYTHON,
    INFERENCE_SCRIPT,
)
from pathlib import Path
import subprocess

BATCH_SIZE = 2


def predict_model(mission_type, training_sites, test_site, run_ID, full_site=False):
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
        + f"{checkpoint_file} {input_images} {prediction_folder} --batch-size {BATCH_SIZE}"
        + f" --extension {extension}"
    )
    print(run_str)
    subprocess.run(
        run_str,
        shell=True,
    )


ALL_SITES = ["chips", "delta", "lassic", "valley"]

# ALL_MISSION_TYPES = ("MV-HN", "MV-LO")
ALL_MISSION_TYPES = ("ortho",)

# for run_ID in ("02",):
#    for test_site in ALL_SITES:
#        for mission_type in ALL_MISSION_TYPES:
#            training_sites = list(filter(lambda x: x != test_site, ALL_SITES))
#            predict_model(
#                mission_type=mission_type,
#                training_sites=training_sites,
#                test_site=test_site,
#                run_ID=run_ID,
#            )

for run_ID in ("00",):
    for test_site in ALL_SITES:
        for mission_type in ("MV-HN", "MV-LO"):
            predict_model(
                mission_type=mission_type,
                training_sites=ALL_SITES,
                test_site=test_site,
                run_ID=run_ID,
                full_site=True,
            )
