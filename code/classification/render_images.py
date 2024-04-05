from constants import (
    GROUND_HEIGHT_THRESHOLD,
    LABELS_COLUMN,
    LABELS_FILENAME,
    get_cameras_filename,
    get_DTM_filename,
    get_IDs_to_labels,
    get_image_folder,
    get_labeled_mesh_filename,
    get_labels_vis_folder,
    get_mesh_filename,
    get_mesh_transform_filename,
    get_mesh_vis_file,
    get_render_folder,
    get_subset_images_folder,
)
from geograypher.entrypoints import render_labels
from geograypher.utils.visualization import show_segmentation_labels

SITE_NAMES = ["chips", "delta", "lassic", "valley"][2:]

IDs_to_labels = get_IDs_to_labels()

for site_name in SITE_NAMES:
    image_folder = get_image_folder(site_name)
    mesh_file = get_mesh_filename(site_name)
    cameras_file = get_cameras_filename(site_name)
    DTM_file = get_DTM_filename(site_name)
    mesh_transform_file = get_mesh_transform_filename(site_name)
    render_folder = get_render_folder(site_name)
    subset_images_savefolder = get_subset_images_folder(site_name)
    labeled_mesh_filename = get_labeled_mesh_filename(site_name)
    mesh_vis_file = get_mesh_vis_file(site_name)
    labels_vis_folder = get_labels_vis_folder(site_name)

    print(f"About to render {site_name}")
    render_labels(
        mesh_file=mesh_file,
        cameras_file=cameras_file,
        image_folder=image_folder,
        DTM_file=DTM_file,
        ground_height_threshold=GROUND_HEIGHT_THRESHOLD,
        texture=LABELS_FILENAME,
        texture_column_name=LABELS_COLUMN,
        transform_file=mesh_transform_file,
        subset_images_savefolder=subset_images_savefolder,
        render_savefolder=render_folder,
        textured_mesh_savefile=labeled_mesh_filename,
        mesh_vis_file=mesh_vis_file,
        labels_vis_folder=labels_vis_folder,
    )
