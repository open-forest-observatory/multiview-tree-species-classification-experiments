## Match drone trees to field trees and join field data to drone-detected tree crowns

library(tidyverse)
library(sf)
library(units)


### Setup

source("code/1_data-prep/3_join-predicted-and-observed-trees/lib/match-trees.R")

FIELD_STEMMAP_FILE = "/ofo-share/str-disp_drone-data-v2/field-reference/stems_v4.gpkg"
FIELD_STEMMAP_BOUNDS_FILE = "/ofo-share/str-disp_drone-data-v2/field-reference/plot_bounds_v4.gpkg"
PREDICTED_TREETOPS_DIR = "/ofo-share/str-disp_drone-data-v2/predicted-treetops/"
PREDICTED_TREECROWNS_DIR = "/ofo-share/str-disp_drone-data-v2/predicted-treecrowns/"
PREDICTED_TREECROWNS_W_FIELD_DATA_FILE = "/ofo-share/str-disp_drone-data-v2/predicted-treecrowns-w-field-data/predicted-treecrowns-w-field-data.gpkg"

# Load field trees
trees_field = st_read(FIELD_STEMMAP_FILE) |>
  st_transform(3310)

# Load field perims
perims_field = st_read(FIELD_STEMMAP_BOUNDS_FILE) |>
  st_transform(3310)


## Get the field tree dataset into the format expected by the matching algorithm (column names, etc)
trees_field$Height = trees_field$ht_top

# If there's "snag" in the species, make the spacies "SNAG" and make percent green 0 (sometimes it
# was omitted in the field because it was obvious it was 0)
trees_field = trees_field |>
  mutate(species = toupper(species)) |>
  mutate(species = ifelse(str_detect(species, regex("snag", ignore_case = TRUE)), "SNAG", species),
        pct_current_green = ifelse(str_detect(species, regex("snag", ignore_case = TRUE)), 0, pct_current_green)) |>
  # assume that if percent current green is NA, it's 0
  mutate(pct_current_green = ifelse(is.na(pct_current_green) | pct_current_green == "NA", 0, pct_current_green))


# Make a data frame of all the field stem maps we want, so we can loop through it
stemmaps = data.frame(stem_map_name = c("Chips_1", "Chips_1_ABCO", "Chips_2", "Delta_1", "Delta_2", "Delta_3", "Valley_1", "Lassic_1", "Lassic_2"),
                      fire_name =     c("chips",   "chips",        "chips",   "delta",   "delta",    "delta", "valley",    "lassic",    "lassic"))

crowns_drone_w_field_data = data.frame()

for (i in 1:nrow(stemmaps)) {

  stem_map_name_foc = stemmaps[i, ]$stem_map_name
  fire_name_foc = stemmaps[i, ]$fire_name

  # Load field trees
  trees_field_foc = trees_field |>
      filter(stem_map_name == stem_map_name_foc) |>
      mutate(observed_tree_id = tree_id)

  # Load field perim
  perim_field_foc = perims_field |>
      filter(stem_map_name == stem_map_name_foc)

  # Load drone trees (points and crowns) and crop to focal area around field reference trees
  treetops_file = file.path(PREDICTED_TREETOPS_DIR, paste0(fire_name_foc, ".gpkg"))
  trees_drone = st_read(treetops_file) |>
    st_transform(3310)

  ## Get the drone tree dataset into the expected format (column names, etc)
  trees_drone = trees_drone |>
    select(predicted_tree_id = treeID,
           height = Z)

  # Load drone crowns
  crowns_file = file.path(PREDICTED_TREECROWNS_DIR, paste0(fire_name_foc, ".gpkg"))
  crowns_drone = st_read(crowns_file) |>
    st_transform(3310) |>
    select(predicted_tree_id = treeID)

  # Designate area beyond the field stem map perimeter to allow field trees to match to drone trees
  perim_buff = st_buffer(perim_field_foc, 10)

  ## Get drone trees and crowns within the buffered field plot
  trees_drone_foc = trees_drone |>
    st_intersection(perim_buff)

  crowns_drone_foc_idxs = crowns_drone |>
    st_intersects(perim_buff, sparse = FALSE)

  crowns_drone_foc = crowns_drone[crowns_drone_foc_idxs, ]

  # Run matching and filter to only matched trees
  matches = match_trees_singlestratum(trees_field_foc,
                                      trees_drone_foc,
                                      search_height_proportion = 0.5,
                                      search_distance_fun_slope = 0.1,
                                      search_distance_fun_intercept = 1)

  matches = matches |>
    filter(!is.na(final_predicted_tree_match_id))

  ## Take the crown polygons and look up the species of the matched field tree
  # First get only the columns we need from the field tree data
  trees_field_foc_simp = matches |>
    st_drop_geometry() |>
    select(observed_tree_id,
           species_observed = species,
           height_observed = ht_top,
           percent_green_observed = pct_current_green,
           stem_map_name,
           predicted_tree_id = final_predicted_tree_match_id) |>
    mutate(live_observed = as.numeric(percent_green_observed) > 0,
           percent_green_observed = as.numeric(percent_green_observed),
           fire = fire_name_foc)

  #  Join the field tree data to the drone crown polygons, also pull in the photogrammetry tree height (from the treetop points)
  crowns_drone_foc_w_field_data = crowns_drone_foc |>
    inner_join(trees_field_foc_simp, by = "predicted_tree_id") |>
    left_join(trees_drone_foc |> st_drop_geometry(), by = join_by(predicted_tree_id, stem_map_name)) |>
    rename(height_chm = height)

  # Bind onto running data frame
  crowns_drone_w_field_data = rbind(crowns_drone_w_field_data, crowns_drone_foc_w_field_data)

}

st_write(crowns_drone_w_field_data, PREDICTED_TREECROWNS_W_FIELD_DATA_FILE, delete_dsn = TRUE)
