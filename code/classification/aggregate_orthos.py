from geograypher.segmentation import assemble_tiled_predictions

from constants import (
    get_training_raster_filename,
    get_prediction_folder,
    ALL_SITE_NAMES,
    get_IDs_to_labels,
    get_aggregated_raster_pred_file,
)


ALL_TRAINING_SITES = (
    ["chips", "delta", "lassic", "valley"],
    ["chips", "delta", "lassic"],
    ["chips", "delta", "valley"],
    ["chips", "lassic", "valley"],
    ["delta", "lassic", "valley"],
)

num_classes = len(get_IDs_to_labels())

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
