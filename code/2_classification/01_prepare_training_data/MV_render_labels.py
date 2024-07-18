import sys
from argparse import ArgumentParser
from pathlib import Path

from geograypher.entrypoints import render_labels
from geograypher.utils.visualization import show_segmentation_labels

# Import from constants file
constants_dir = str(Path(Path(__file__).parent, "..").resolve())
sys.path.append(constants_dir)
from constants import (
    ALL_SITE_NAMES,
    DEFAULT_INPUT_DATA_DIR,
    DEFAULT_PREDICTION_DATA_DIR,
    GROUND_HEIGHT_THRESHOLD,
    LABELS_COLUMN,
    get_cameras_filename,
    get_DTM_filename,
    get_IDs_to_labels,
    get_image_folder,
    get_labeled_mesh_filename,
    get_labels_filename,
    get_mesh_filename,
    get_mesh_transform_filename,
    get_MV_training_folder,
)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--site-names", nargs="+", default=ALL_SITE_NAMES)
    parser.add_argument("--just-vis", action="store_true")
    parser.add_argument("--input-data-dir", default=DEFAULT_INPUT_DATA_DIR)
    parser.add_argument("--prediction-data-dir", default=DEFAULT_PREDICTION_DATA_DIR)
    parser.add_argument("--dont-include-snags", action="store_true")
    args = parser.parse_args()
    return args


def main(site_name, include_snag_class, input_data_dir, prediction_data_dir, just_vis):
    IDs_to_labels = get_IDs_to_labels(include_snag_class=include_snag_class)

    labels_filename = get_labels_filename(
        input_data_dir=input_data_dir, include_snag_class=include_snag_class
    )
    image_folder = get_image_folder(site_name, input_data_dir=input_data_dir)
    mesh_file = get_mesh_filename(site_name, input_data_dir=input_data_dir)
    cameras_file = get_cameras_filename(site_name, input_data_dir=input_data_dir)
    DTM_file = get_DTM_filename(site_name, input_data_dir=input_data_dir)
    mesh_transform_file = get_mesh_transform_filename(site_name, input_data_dir)
    labeled_mesh_filename = get_labeled_mesh_filename(
        site_name=site_name, prediction_data_dir=prediction_data_dir
    )
    MV_training_folder = get_MV_training_folder(
        site_name, prediction_data_dir=prediction_data_dir
    )
    mesh_vis_file = get_labeled_mesh_filename(
        site_name, prediction_data_dir=prediction_data_dir, get_vis_filename=True
    )

    MV_training_imgs_folder = Path(MV_training_folder, "imgs")
    MV_training_anns_folder = Path(MV_training_folder, "anns")
    MV_training_vis_folder = Path(MV_training_folder, "vis")

    # Only visualize data if that's all that is requested
    if just_vis:
        print(f"About to show {site_name}")
        show_segmentation_labels(
            label_folder=MV_training_anns_folder, image_folder=MV_training_imgs_folder
        )
    else:
        print(f"About to render {site_name}")
        render_labels(
            mesh_file=mesh_file,
            cameras_file=cameras_file,
            image_folder=image_folder,
            DTM_file=DTM_file,
            ground_height_threshold=GROUND_HEIGHT_THRESHOLD,
            texture=labels_filename,
            texture_column_name=LABELS_COLUMN,
            transform_file=mesh_transform_file,
            subset_images_savefolder=MV_training_imgs_folder,
            render_savefolder=MV_training_anns_folder,
            textured_mesh_savefile=labeled_mesh_filename,
            mesh_vis_file=mesh_vis_file,
            labels_vis_folder=MV_training_vis_folder,
            IDs_to_labels=IDs_to_labels,
            cameras_ROI_buffer_radius_meters=100,
        )


if __name__ == "__main__":
    args = parse_args()
    for site_name in args.site_names:
        main(
            site_name=site_name,
            include_snag_class=not args.dont_include_snags,
            input_data_dir=args.input_data_dir,
            prediction_data_dir=args.prediction_data_dir,
            just_vis=args.just_vis,
        )
