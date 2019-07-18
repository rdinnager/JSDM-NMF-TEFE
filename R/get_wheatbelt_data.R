library(readr)
library(dplyr)
library(tidyr)

spp_dat <- read_csv("data/wb_sppsites_APG.csv")
plot_data <- read_csv("data/wb_plots.csv")

pa_dat <- spp_dat %>%
  dplyr::select(SiteID, Name) %>%
  mutate(Pres = 1) %>%
  group_by(SiteID, Name) %>%
  summarise(count = sum(Pres)) %>%
  tidyr::spread(SiteID, count, fill = 0) %>%
  mutate_at(vars(-Name), ~ ifelse(. > 0, 1, 0))

trait_dat <- spp_dat %>%
  group_by(Name) %>%
  summarise(Family = Family[1], Genus = Genus[1], Lifeform = lifeform[1])

env_dat <- plot_data %>%
  dplyr::select(SiteID = Quadrat, Lat, Long,
                Elev, MTAnn, Pann, pH, OrgC, NTOT,
                PTOT, K, Clay, Silt, Sand, Sdep)

write_csv(pa_dat, "data/wb_pa_data.csv")
write_csv(trait_dat, "data/wb_trait_data.csv")
write_csv(env_dat, "data/wb_env_data.csv")