library(tidyverse)
library(cowplot)

# set directory containing output
exp <- "20180518"
filedir <- paste0("~/Box Sync/ZamanianLab/Data/WormvizOutput/Bpahangi/", exp)
setwd(filedir)

# drug(s) of interest
drug = "DOP"

# get plate design
plate1 <- read.csv("plate_design.csv", header = FALSE, sep = ",")
colnames(plate1) <- c("row","0001","0002","0003","0004","0005","0006","0007","0008","0009","0010","0011","0012")
plate1 <- plate1 %>% slice(2:9)
plate1.m <- plate1 %>% 
  gather(col, Treatment, 2:13) %>% 
  mutate(Well = paste0(row, col)) %>%
  select(-row, -col) %>%
  separate(Treatment,c("Drug", "Dose", "Rep"), sep = "_", remove = FALSE) %>%
  filter(Drug == drug) %>%
  mutate(Exp = exp)

#Import data
data1 <- read.csv(paste0(exp, ".csv"), header = TRUE) %>%
  select(Well, Motility = Normalized.Motility) 

# second directory containing DOP files
exp <- "20180316"
filedir <- paste0("~/Box Sync/ZamanianLab/Data/WormvizOutput/Bpahangi/", exp)
setwd(filedir)

#Get plate design
plate2 <- read.csv("plate_design.csv", header = FALSE, sep = ",")
colnames(plate2) <- c("row","0001","0002","0003","0004","0005","0006","0007","0008","0009","0010","0011","0012")
plate2 <- plate2 %>% slice(2:9)
plate2.m <- plate2 %>% 
  gather(col, Treatment, 2:13) %>% 
  mutate(Well = paste0(row, col)) %>%
  select(-row, -col) %>%
  separate(Treatment,c("Drug", "Dose", "Rep"), sep = "_", remove = FALSE) %>%
  filter(Drug == drug) %>%
  mutate(Exp = exp)

data2 <- read.csv(paste0(drug, "_", exp, ".csv"), header = TRUE) %>%
  select(Well, Motility = Normalized.Motility) 

all_wells <- rbind(plate1.m, plate2.m)
all_data <- rbind(data1, data2)

lookup <- data.frame(Dose = c("10-3", "5-3", "2.5-3", "1.25-3", "10-4", "6.25-4", "3.125-4", "10-5", "10-6", "10-7", "10-8"),
                     DEC = c(1000, 500, 250, 125, 100, 62.5, 31.25, 10, 1, 0.1, 0.01))

dop_data <- all_wells %>%
  left_join(., lookup) %>%
  left_join(., all_data)

dop_plot <- ggplot(dop_data, aes(x = DEC, y = Motility)) + 
  geom_boxplot(aes(group = DEC)) +
  geom_point(aes(color = Rep)) +
  geom_smooth(method = loess, se = FALSE) +
  scale_x_log10(limits = c(0.01, 1000), 
                breaks = c(0.01, 0.1, 1, 10, 100, 1000),
                labels = c("0.01", "0.1", "1", "10", "100", "1000")) +
  # scale_y_continuous(limits = c(0, 1.25), breaks = c(0, 0.25, 0.5, 0.75, 1.0)) +
  labs(x = "Drug Concentration (uM)", y = "Normalized Optical Flow") +
  theme_bw(base_size = 16, base_family = "Helvetica") +
  theme(legend.position = "none")
dop_plot

save_plot(paste0(exp, "_DOP", ".pdf"), dop_plot, base_height = 10, base_aspect_ratio = 1)
