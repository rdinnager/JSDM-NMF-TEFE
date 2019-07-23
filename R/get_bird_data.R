library(letsR)
library(readr)
library(sp)
library(sf)
library(raster)
library(fasterize)
library(pbapply)
library(dplyr)

load("//rsb.anu.edu.au/data/largedata/EEG/Cardillo Lab/BirdLife/all_spp_ranges.rData")

ranges <- ranges[[1]]

test <- st_as_sf(ranges[[1]])
test2 <- st_as_sf(ranges[[2]])
test3 <- rbind(test, test2)

test_list <- list(test, test2)
test_sf <- test_list %>%
  do.call(rbind, .)

## get world raster from worldclim
bioclim <- getData("worldclim", var = "bio", res = 10)
bioclim
plot(bioclim[[1]])

test_fast <- fasterize(test3, bioclim[[1]], by = "SCINAME", background = 0)
test_fast <- mask(test_fast, bioclim[[1]])
plot(test_fast)

ranges <- pblapply(ranges, st_as_sf) %>%
  do.call(rbind, .)

write_rds(ranges, "data/all_bird_ranges_sf.rds")
writeRaster(bioclim, "data/bird_matching_bioclim.grd")

all_bird_pa <- fasterize(ranges, bioclim[[1]], by = "SCINAME", background = 0)
all_bird_pa <- mask(all_bird_pa, bioclim[[1]])

writeRaster(all_bird_pa, "data/all_bird_pa.grd")


bird_pa <- lets.presab.birds("//rsb.anu.edu.au/data/largedata/EEG/Cardillo Lab/Dinnage/Bird-Biomes/data/ranges",
                             count = TRUE)