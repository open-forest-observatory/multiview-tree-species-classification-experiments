import os
from collections import defaultdict
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from constants import (
    ALL_SITE_NAMES,
    get_IDs_to_labels,
    get_npy_export_confusion_matrix_file,
)
from geograypher.utils.prediction_metrics import compute_comprehensive_metrics
from sklearn.metrics import ConfusionMatrixDisplay

VIS_FOLDER = Path(
    "/ofo-share/scratch-david/multiview-tree-species-classification-experiments-earthvision/vis/total_accuracy"
)

IDs_to_labels = get_IDs_to_labels()

labels = list(IDs_to_labels.values())

all_metrics = {}
all_cfs = {}

MISSION_TYPES = ("ortho", "MV-HN", "MV-LO")
RUN_IDS = ("00", "01", "02")

# Read in all the saved confusion matrices
for mission_type in MISSION_TYPES:
    for site_name in ALL_SITE_NAMES:
        for run_ID in RUN_IDS:
            # Determine the training sites as the ones that aren't the inference site
            training_sites = [x for x in ALL_SITE_NAMES if x != site_name]
            cf_matrix_np_path = get_npy_export_confusion_matrix_file(
                site_name, training_sites, mission_type=mission_type, run_ID=run_ID
            )
            cf_matrix = np.load(cf_matrix_np_path)
            key = (site_name, mission_type, run_ID)
            all_cfs[key] = cf_matrix
            all_metrics[key] = compute_comprehensive_metrics(
                cf_matrix=cf_matrix, class_names=labels
            )

# Sort into a list for each mission
cfs_by_mission = {
    mission_type: list(filter(lambda x: x[0][1] == mission_type, all_cfs.items()))
    for mission_type in MISSION_TYPES
}
# Extract the cf matrix since it was a (key, value) tuple
cfs_by_mission = {k: [x[1] for x in v] for k, v in cfs_by_mission.items()}
# Sum all the CF matrices
cfs_by_mission = {k: np.sum(v, axis=0) for k, v in cfs_by_mission.items()}

# Print per-site accuracy and save the confusion matrices
for k, v in cfs_by_mission.items():
    print(f"Accuracy for {k} is {np.sum(v * np.eye(5)) / np.sum(v)*100:.02f}%")
    cf_disp = ConfusionMatrixDisplay(confusion_matrix=v, display_labels=labels)
    cf_disp.plot()

    plt.savefig(Path(VIS_FOLDER, f"{k}.png"))


cfs_by_mission_run = {
    mission_run: list(
        filter(
            lambda x: x[0][1] == mission_run[0] and x[0][2] == mission_run[1],
            all_cfs.items(),
        )
    )
    for mission_run in product(MISSION_TYPES, RUN_IDS)
}
cfs_by_mission_run = {
    k: np.sum([x[1] for x in v], axis=0) for k, v in cfs_by_mission_run.items()
}
metrics_by_mission_run = {
    k: compute_comprehensive_metrics(v, class_names=labels)
    for k, v in cfs_by_mission_run.items()
}

acc_values_per_site = defaultdict(list)
rec_values_per_site = defaultdict(list)
prec_values_per_site = defaultdict(list)

for k, v in all_metrics.items():
    acc_values_per_site[k[:2]].append(v["accuracy"])
    rec_values_per_site[k[:2]].append(v["class_averaged_recall"])
    prec_values_per_site[k[:2]].append(v["class_averaged_precision"])

for k, v in metrics_by_mission_run.items():
    key = ("all", k[0])
    acc_values_per_site[key].append(v["accuracy"])
    rec_values_per_site[key].append(v["class_averaged_recall"])
    prec_values_per_site[key].append(v["class_averaged_precision"])

site_missions = list(product(ALL_SITE_NAMES + ["all"], MISSION_TYPES))

# Print the table of per-site accuracies

# Print accuracies
print("\n\n\nAcc.", end="")
for sm in site_missions:
    print(f" & {np.mean(acc_values_per_site[sm]):.2f}", end="")
print("\\\\")
# And their std
print("std", end="")
for sm in site_missions:
    print(f" & {np.std(acc_values_per_site[sm]):.2f}", end="")
print("\\\\ \\hline")

print("Rec.", end="")
for sm in site_missions:
    print(f" & {np.mean(rec_values_per_site[sm]):.2f}", end="")
print("\\\\")
# And their std
print("std", end="")
for sm in site_missions:
    print(f" & {np.std(rec_values_per_site[sm]):.2f}", end="")
print("\\\\ \\hline")

print("Prec.", end="")
for sm in site_missions:
    print(f" & {np.mean(prec_values_per_site[sm]):.2f}", end="")
print("\\\\")
# And their std
print("std", end="")
for sm in site_missions:
    print(f" & {np.std(prec_values_per_site[sm]):.2f}", end="")
print("\\\\ \\hline\n\n\n")

# Aggregate across runs
cfs_by_mission_site = {
    (mission, site): list(
        filter(
            lambda x: x[0][0] == site and x[0][1] == mission,
            all_cfs.items(),
        )
    )
    for site, mission in product(ALL_SITE_NAMES, MISSION_TYPES)
}
cfs_by_mission_site = {
    k: np.sum([x[1] for x in v], axis=0) for k, v in cfs_by_mission_site.items()
}
accuracy_by_mission_site = {
    k: compute_comprehensive_metrics(v, class_names=labels)["accuracy"]
    for k, v in cfs_by_mission_site.items()
}

site_mission_cf_savefolder = Path(VIS_FOLDER, "site-mission-cfs")
site_mission_cf_savefolder.mkdir(exist_ok=True, parents=True)

# Save out the cf matrices
for key in cfs_by_mission_site.keys():
    cf = cfs_by_mission_site[key]
    accuracy = accuracy_by_mission_site[key]
    print(f"{key[1]} {key[0]}. Acc: {accuracy:.03f}")

    cf_disp = ConfusionMatrixDisplay(confusion_matrix=cf, display_labels=labels)
    cf_disp.plot()

    plt.savefig(Path(site_mission_cf_savefolder, f"{key[1]}_{key[0]}.png"))
