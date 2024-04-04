from constants import (
    get_prediction_folder,
    get_subset_images_folder,
    get_work_dir,
    MMSEG_PYTHON,
    INFERENCE_SCRIPT,
)
from pathlib import Path
import subprocess

BATCH_SIZE = 2


def predict_model(mission_type, training_sites, test_site, run_ID):
    subset_images_folder = get_subset_images_folder(
        test_site, mission_type=mission_type
    )
    prediction_folder = get_prediction_folder(
        prediction_site=test_site,
        training_sites=training_sites,
        mission_type=mission_type,
        run_ID=run_ID,
    )

    prediction_folder = Path(prediction_folder, subset_images_folder.parts[-1])
    work_dir = get_work_dir(
        training_sites=training_sites, mission_type=mission_type, run_ID=run_ID
    )
    config_file = list(Path(work_dir).glob("*py"))[0]
    checkpoint_file = Path(work_dir, "iter_10000.pth")
    prediction_folder.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        f"{MMSEG_PYTHON} {INFERENCE_SCRIPT} {config_file} "
        + f"{checkpoint_file} {subset_images_folder} {prediction_folder} --batch-size {BATCH_SIZE}",
        shell=True,
    )


ALL_SITES = ["chips", "delta", "lassic", "valley"]

ALL_MISSION_TYPES = ("MV-HN", "MV-LO")
ALL_TRAINING_SITES = (
    ["chips", "delta", "lassic", "valley"],
    ["chips", "delta", "lassic"],
    ["chips", "delta", "valley"],
    ["chips", "lassic", "valley"],
    ["delta", "lassic", "valley"],
)

for run_ID in ("00", "01", "02"):
    for test_site in ALL_SITES:
        for mission_type in ALL_MISSION_TYPES:
            training_sites = list(filter(lambda x: x != test_site, ALL_SITES))
            predict_model(
                mission_type=mission_type,
                training_sites=training_sites,
                test_site=test_site,
                run_ID=run_ID,
            )
