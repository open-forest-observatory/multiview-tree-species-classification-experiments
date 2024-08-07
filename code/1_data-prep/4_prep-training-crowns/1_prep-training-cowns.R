## Take the drone-detected tree crowns, which have been attributed with field data from matching field
# trees, filter them to drop trees with < 50% green crown volume, drop trees of very rare species,
# and merge PIPO and PIJE into PIPJ

PREDICTED_TREECROWNS_W_FIELD_DATA_FILE = "/ofo-share/str-disp_drone-data-v2/predicted-treecrowns-w-field-data/predicted-treecrowns-w-field-data.gpkg"
PREDICTED_TREECROWNS_W_FIELD_DATA_CLEANED_FILE = "/ofo-share/str-disp_drone-data-v2/predicted-treecrowns-w-field-data/predicted-treecrowns-w-field-data-cleaned.gpkg"
PREDICTED_TREECROWNS_W_FIELD_DATA_FILTERED_FILE = "/ofo-share/str-disp_drone-data-v2/predicted-treecrowns-w-field-data/predicted-treecrowns-w-field-data-cleaned-snagsremoved.gpkg"


library(tidyverse)
library(sf)

crowns = st_read(PREDICTED_TREECROWNS_W_FIELD_DATA_FILE)

# merge species that are functionally the same, and drop species that are very rare
crowns = crowns |>
  filter(!(species_observed %in% c("QUEV", "QUCC"))) |>
  mutate(species_observed = recode(species_observed,
                                   "PIPO" = "PIPJ",
                                   "PIJE" = "PIPJ")) |>
  mutate(species_observed = ifelse(percent_green_observed < 10, "SNAG", species_observed))

st_write(crowns, PREDICTED_TREECROWNS_W_FIELD_DATA_CLEANED_FILE, delete_dsn = TRUE)

# drop snags (actually any trees with < 50% green crown volume)
crowns = crowns |>
  filter(percent_green_observed >= 50)

st_write(crowns, PREDICTED_TREECROWNS_W_FIELD_DATA_FILTERED_FILE, delete_dsn = TRUE)
