## Take the drone-detected tree crowns, which have been attributed with field data from matching field
# trees, filter them to drop trees with < 50% green crown volume, drop trees of very rare species,
# and merge PIPO and PIJE into PIPJ

PREDICTED_TREECROWNS_W_FIELD_DATA_FILE = "/ofo-share/str-disp_drone-data-v2/predicted-treecrowns-w-field-data/predicted-treecrowns-w-field-data.gpkg"
PREDICTED_TREECROWNS_W_FIELD_DATA_FILTERED_FILE = "/ofo-share/str-disp_drone-data-v2/predicted-treecrowns-w-field-data/predicted-treecrowns-w-field-data-filtered.gpkg"


library(tidyverse)
library(sf)

crowns = st_read(PREDICTED_TREECROWNS_W_FIELD_DATA_FILE)

crowns = crowns |>
  filter(percent_green_observed >= 50) |>
  filter(!(species_observed %in% c("QUEV"))) |>
  mutate(species_observed = recode(species_observed,
                                   "PIPO" = "PIPJ",
                                   "PIJE" = "PIPJ"))

st_write(crowns, PREDICTED_TREECROWNS_W_FIELD_DATA_FILTERED_FILE, delete_dsn = TRUE)
