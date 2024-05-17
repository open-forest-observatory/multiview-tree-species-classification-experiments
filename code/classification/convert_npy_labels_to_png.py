from pathlib import Path

import numpy as np
from geograypher.utils.files import ensure_containing_folder
from imageio import imwrite
from tqdm import tqdm

from constants import ALL_SITE_NAMES, get_render_folder

for site_name in ALL_SITE_NAMES:
    render_folder = get_render_folder(site_name)
    render_folder_npy = Path(render_folder.parent, render_folder.name + "_npy")

    npy_files = list(render_folder_npy.rglob("*npy"))
    for npy_file in tqdm(npy_files):
        rel_path = npy_file.relative_to(render_folder_npy)
        output_path = Path(render_folder, rel_path).with_suffix(".png")
        ensure_containing_folder(output_path)

        label_data = np.load(npy_file)

        if np.any(np.logical_not(np.isfinite(label_data))):
            breakpoint()

        label_data = label_data.astype(np.uint8)

        imwrite(output_path, np.squeeze(label_data))
