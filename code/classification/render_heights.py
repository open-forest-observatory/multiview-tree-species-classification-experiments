from pathlib import Path

from constants import (
    ALL_SITE_NAMES,
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
from geograypher.cameras import MetashapeCameraSet
from geograypher.meshes import TexturedPhotogrammetryMesh
from geograypher.utils.visualization import show_segmentation_labels

IDs_to_labels = get_IDs_to_labels()

for site_name in ALL_SITE_NAMES:
    image_folder = get_image_folder(site_name)
    mesh_file = get_mesh_filename(site_name)
    cameras_file = get_cameras_filename(site_name)
    DTM_file = get_DTM_filename(site_name)
    mesh_transform_file = get_mesh_transform_filename(site_name)
    render_folder = get_render_folder(site_name)
    subset_images_savefolder = get_subset_images_folder(site_name)
    labeled_mesh_filename = get_labeled_mesh_filename(site_name)
    mesh_vis_file = get_mesh_vis_file(site_name)
    labels_vis_folder = get_labels_vis_folder(site_name, "MV")

    render_folder = Path(render_folder.parent, "heights_above_ground")

    # print(f"About to render {site_name}")
    # mesh = TexturedPhotogrammetryMesh(
    #    mesh_file,
    #    transform_filename=cameras_file,
    #    ROI=LABELS_FILENAME,
    #    ROI_buffer_meters=100,
    # )
    # height_above_ground = mesh.get_height_above_ground(DTM_file=DTM_file)
    # height_above_ground_mesh = TexturedPhotogrammetryMesh(
    #    mesh.pyvista_mesh,
    #    transform_filename=cameras_file,
    #    texture=height_above_ground,
    # )
    # camera_set = MetashapeCameraSet(camera_file=cameras_file, image_folder=image_folder)

    # training_camera_set = camera_set.get_subset_ROI(
    #    ROI=LABELS_FILENAME, buffer_radius_meters=100
    # )
    ##height_above_ground_mesh.vis(camera_set=training_camera_set)
    # height_above_ground_mesh.save_renders_pytorch3d(
    #    camera_set=training_camera_set,
    #    output_folder=render_folder,
    #    make_composites=False,
    # )
    labels_vis_folder = Path(labels_vis_folder.parent, "height_vis")
    show_segmentation_labels(
        render_folder,
        image_folder,
        savefolder=labels_vis_folder,
        label_suffix=".npy",
    )
