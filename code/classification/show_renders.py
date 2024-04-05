from constants import (
    get_render_folder,
    get_subset_images_savefolder,
    get_labels_vis_folder,
)
from geograypher.utils.visualization import show_segmentation_labels

site = "lassic"

render_folder = get_render_folder(site)
subset_folder = get_subset_images_savefolder(site)
labels_vis_folder = get_labels_vis_folder(site)


show_segmentation_labels(render_folder, subset_folder, savefolder=labels_vis_folder)
