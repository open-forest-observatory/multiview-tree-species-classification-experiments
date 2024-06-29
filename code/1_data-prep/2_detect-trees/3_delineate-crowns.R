## Takes ttops and a CHM and makes a map of tree crowns ("treecrowns")

library(sf)
library(terra)
library(tidyverse)
library(lidR)
library(nngeo)
library(smoothr)

CHM_DIR = "/ofo-share/str-disp_drone-data-v2/chms/"
PREDICTED_TREETOPS_DIR = "/ofo-share/str-disp_drone-data-v2/predicted-treetops/"
PREDICTED_TREECROWNS_DIR = "/ofo-share/str-disp_drone-data-v2/predicted-treecrowns/"

sites = c("delta", "chips", "valley", "lassic")

for (site in sites) {

  chm_file = file.path(CHM_DIR, paste0(site, ".tif"))
  treetop_file = file.path(PREDICTED_TREETOPS_DIR, paste0(site, ".gpkg"))
  treecrown_out_file = file.path(PREDICTED_TREECROWNS_DIR, paste0(site, ".gpkg"))

  chm = rast(chm_file)
  ttops = st_read(treetop_file)


  # Force raster to load into memory
  chm = chm * 1

  treecrowns = silva2016(chm, ttops, max_cr_factor = 0.24, exclusion = 0.1)()


  treecrowns <- as.polygons(treecrowns)
  treecrowns <- st_as_sf(treecrowns)
  treecrowns <- st_cast(treecrowns, "MULTIPOLYGON")
  treecrowns <- st_cast(treecrowns, "POLYGON")
  treecrowns <- st_remove_holes(treecrowns)
  treecrowns <- st_make_valid(treecrowns)
  treecrowns <- smooth(treecrowns, method = "ksmooth", smoothness = 3)
  treecrowns <- st_simplify(treecrowns, preserveTopology = TRUE, dTolerance = 0.1)

  # assign treecrowns the treetop height and remove those that have no treetops in them
  treecrowns = st_join(treecrowns, ttops)
  treecrowns = treecrowns[, -1]
  treecrowns = treecrowns[!is.na(treecrowns$Z),]

  st_write(treecrowns, treecrown_out_file, delete_dsn = TRUE)

}
