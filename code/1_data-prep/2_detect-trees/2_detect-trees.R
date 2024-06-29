## Takes a CHM and makes a map of treetops

library(sf)
library(terra)
library(here)
library(tidyverse)
library(lidR)

### Setup ####

FIELD_BOUNDARIES_DIR = "/ofo-share/str-disp_drone-data-v2/field-site-boundaries/"
CHM_DIR = "/ofo-share/str-disp_drone-data-v2/chms/"
PREDICTED_TREETOPS_DIR = "/ofo-share/str-disp_drone-data-v2/predicted-treetops/"


#### Processing ####

sites = c("lassic", "valley", "chips", "delta")

for (site in sites) {

  ## Layers to process

  focal_area_file = file.path(FIELD_BOUNDARIES_DIR, paste0(site, ".gpkg"))
  chm_file = file.path(CHM_DIR, paste0(site, ".tif"))
  treetop_out_file = file.path(PREDICTED_TREETOPS_DIR, paste0(site, ".gpkg"))

  focal_area = st_read(focal_area_file) |> st_transform(3310)
  focal_area_inner = st_buffer(focal_area, -10) # to remove the edge trees

  # find the chm file
  chm = rast(chm_file)

  cat("Fix extraneous vals")
  chm[chm < 0] = -0.1

  chm_res = res(chm) |>  mean()

  #resample coarser
  chm_coarse = project(chm, y = "epsg:3310", res = 0.25)

  # skip smoothing
  chm_smooth = chm_coarse
  # # apply smooth
  # smooth_size = 7
  # weights = matrix(1, nrow = smooth_size, ncol = smooth_size)
  # chm_smooth = focal(chm_coarse, weights, fun=  mean)

  
  
  cat("Detecting trees\n")
  
  lin <- function(x){
    win = x*0.11 + 0
    win[win < 0.5] = 0.5
    win[win > 100] = 100
    return(win)
  } # window filter function to use in next step

  treetops <- locate_trees(chm_smooth, lmf(ws = lin, shape = "circular", hmin = 5))

  treetops = treetops |>
    rename(coarse_smoothed_chm_height = Z)

  # crop to the inner buffer
  treetops = st_intersection(treetops, focal_area_inner |> st_transform(st_crs(treetops)))

  # pull the height from the highres unsmoothed CHM
  height = terra::extract(chm, treetops)[, 2]
  treetops$highres_chm_height = height

  # pull the height from the coarse unsmoothed CHM
  height = terra::extract(chm_coarse, treetops)[,2]
  treetops$coarse_unsmoothed_chm_height = height


  # get the overall height as the max of the three
  treetops$Z = pmax(treetops$coarse_unsmoothed_chm_height, treetops$coarse_smoothed_chm_height, treetops$highres_chm_height)

  ## Save treetops
  st_write(treetops, treetop_out_file, delete_dsn = TRUE, quiet = TRUE)
}
