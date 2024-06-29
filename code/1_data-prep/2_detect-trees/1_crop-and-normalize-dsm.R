## Takes a DSM, converts to CHM, resamples to 0.12 m, and saves

library(sf)
library(purrr)
library(tidyverse)
library(terra)

#### Setup ####

PHOTOGRAMMETRY_OUTPUTS_DIR = "/ofo-share/str-disp_drone-data-v2/photogrammetry/outputs/"
CHM_DIR = "/ofo-share/str-disp_drone-data-v2/chms/"
FIELD_BOUNDARIES_DIR = "/ofo-share/str-disp_drone-data-v2/field-site-boundaries/"

#### Processing ####

site = "lassic"


# load DTM
dtm_file = list.files(PHOTOGRAMMETRY_OUTPUTS_DIR, pattern = paste0(site, "_\\d{8}T\\d{4}_dtm-ptcloud\\.tif"), full.names = TRUE)
if (length(dtm_file) != 1) stop(paste0("DTM file absent or multiple matches for site: ", site))


## get DSM layer from metashape output
dsm_file = list.files(PHOTOGRAMMETRY_OUTPUTS_DIR, pattern = paste0(site, "_\\d{8}T\\d{4}_dsm-mesh\\.tif"), full.names = TRUE)
if (length(dsm_file) != 1) stop(paste0("Mesh-based DSM file absent or multiple matches for site: ", site))


# get site boundary
boundary_file = file.path(FIELD_BOUNDARIES_DIR, paste0(site, ".gpkg"))
boundary = st_read(boundary_file)

# Crop to study area boundary
dsm = rast(dsm_file)
dsm = crop(dsm, boundary |> st_transform(st_crs(dsm)))
dsm = mask(dsm, boundary |> st_transform(st_crs(dsm)))

dtm = rast(dtm_file)
dtm = crop(dtm, boundary |> st_transform(st_crs(dtm)))
dtm = mask(dtm, boundary |> st_transform(st_crs(dtm)))


## upscale to 0.12 m
dsm_upscale = project(dsm, y = "EPSG:3310", res = 0.12, method = "bilinear")


## interpolate the DTM to the res, extent, etc of the DSM
dtm_interp = project(dtm, dsm_upscale, method = "bilinear")


#### Calculate canopy height model ####
#### and save to tif

## calculate canopy height model
chm = dsm_upscale - dtm_interp


## create dir if doesn't exist, then write# file to write
chm_file = file.path(CHM_DIR, paste0(site, ".tif"))
writeRaster(chm, chm_file, overwrite = TRUE)

gc()
