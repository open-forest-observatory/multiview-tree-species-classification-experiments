## Take the drone-detected tree crowns, which have been attributed with field data from matching field
# trees, filter them to drop trees with < 50% green crown volume, drop trees of very rare species,
# and merge PIPO and PIJE into PIPJ

library(tidyverse)
library(sf)

data_dir = readLines("data_dir.txt", n = 1)

crowns = st_read(file.path(data_dir, "out_crowns-w-field-labels", "crowns_drone_w_field_data.gpkg"))

crowns = crowns |>
  filter(percent_green_observed >= 50) |>
  filter(!(species_observed %in% c("QUEV"))) |>
  mutate(species_observed = recode(species_observed,
                                   "PIPO" = "PIPJ",
                                   "PIJE" = "PIPJ"))

st_write(crowns, file.path(data_dir, "out_crowns-w-field-labels", "crowns_drone_w_field_data_filtered.gpkg"), delete_dsn = TRUE)
