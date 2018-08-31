library(tidyverse)
library(cowplot)

# set directory containing output
exp <- "20180316"
filedir <- paste0("~/Box Sync/ZamanianLab/Data/WormvizOutput/Bpahangi/", exp)
setwd(filedir)

#Get plate design
plate1 <- read.csv("plate_design.csv", header = FALSE, sep = ",")
colnames(plate1) <- c("row","0001","0002","0003","0004","0005","0006","0007","0008","0009","0010","0011","0012")
plate1.m <- plate1 %>%
  slice(2:9) %>%
  gather(col, Treatment, 2:13) %>% 
  mutate(Well = paste0(row, col)) %>%
  select(-row, -col) %>%
  separate(Treatment ,c("Drug", "Dose", "Rep"), sep = "_", remove = FALSE) %>%
  filter(Drug != "NA")

lookup <- data.frame(Dose = c("10-3", "10-4", "10-5", "10-6", "10-7", "10-8"),
                     DEC = c(1000, 100, 10, 1, 0.1, 0.01))

#Import data
# MISSING E0012
data1 <- read.csv(paste0(exp, ".csv"), header = TRUE) %>%
  select(Well, Motility = Normalized.Motility) %>%
  left_join(., plate1.m) %>%
  group_by(Drug, Dose)

#mean data / plot
data1_mean <- summarise(data1, Mean = mean(Motility))

#normalize to control
data1 <- mutate(data1, Normalized.Motility = Motility / data1_mean$Mean[2]) %>%
  filter(Drug != "H2O")

data1 <- data1 %>%
  left_join(., lookup)

plot <- ggplot(data1, aes(x = DEC, y = Normalized.Motility)) + 
  geom_boxplot(aes(group = DEC), alpha = 0.6) +
  geom_point() +
  geom_smooth(method = loess, se = FALSE) +
  scale_x_log10(limits = c(0.004, 3000), 
                breaks = c(0.01, 0.1, 1, 10, 100, 1000),
                labels = c("0.01", "0.1", "1", "10", "100", "1000")) +
  # scale_y_continuous(limits = c(0, 1.25), breaks = c(0, 0.25, 0.5, 0.75, 1.0)) +
  labs(x = "Drug Concentration (uM)", y = "Normalized Optical Flow (Percent of Control)") +
  facet_grid(Drug ~ .) +
  theme_bw(base_size = 16, base_family = "Helvetica") +
  theme(legend.position = "none",
        panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank())
plot

save_plot(paste0("../plots/", exp, ".pdf"), plot, base_height = 10, base_aspect_ratio = 1)

###################################################################################################

#                                          Merge DOP experiments

###################################################################################################

# set directory containing output
exp <- "20180518"
filedir <- paste0("~/Box Sync/ZamanianLab/Data/WormvizOutput/Bpahangi/", exp)
setwd(filedir)

# experiment parameters
drug = "DOP"
control = "H2O"

#Get plate design
plate2 <- read.csv("plate_design.csv", header = FALSE, sep = ",")
colnames(plate2) <- c("row","0001","0002","0003","0004","0005","0006","0007","0008","0009","0010","0011","0012")
plate2.m <- plate2 %>%
  slice(2:9) %>%
  gather(col, Treatment, 2:13) %>% 
  mutate(Well = paste0(row, col)) %>%
  select(-row,-col) %>% 
  separate(Treatment, c("Drug", "Dose", "Rep"), sep = "_", remove = FALSE) %>%
  filter(Drug == drug | Drug == control)

# remove mis-segmented video
plate2.m <- filter(plate2.m, Well != "G0001")

data2 <- read.csv(paste0(exp, ".csv"), header = TRUE) %>%
  select(Well, Motility = Normalized.Motility) %>%
  left_join(., plate2.m) %>%
  filter(!is.na(Drug)) %>%
  group_by(Drug, Dose) %>%
  mutate(Date = "20180518")

#mean data / plot
data2_mean <- summarise(data2, Mean = mean(Motility))

#normalize to control
data2 <- mutate(data2, Normalized.Motility = Motility / data2_mean$Mean[7]) %>%
  filter(Drug != "H2O")

lookup <- data.frame(Dose = c("10-3", "5-3", "2.5-3", "1.25-3", "10-4", "6.25-4", "3.125-4", "10-5", "10-6", "10-7", "10-8"),
                     DEC = c(1000, 500, 250, 125, 100, 62.5, 31.25, 10, 1, 0.1, 0.01))

dop_data <- ungroup(data1) %>%
  filter(Drug == drug) %>%
  select(-DEC) %>%
  mutate(Date = "20180316") %>%
  filter(Drug != "H2O") %>%
  select(Well:Rep, Date, Normalized.Motility) %>% #reorder columns
  bind_rows(., data2) %>%
  left_join(., lookup)

dop_plot <- ggplot(dop_data, aes(x = DEC, y = Normalized.Motility)) + 
  geom_boxplot(aes(group = DEC), alpha = 0.6) +
  geom_point(aes(color = Date)) +
  geom_smooth(method = loess, se = FALSE) +
  scale_x_log10(limits = c(0.004, 3000), 
                breaks = c(0.01, 0.1, 1, 10, 100, 1000),
                labels = c("0.01", "0.1", "1", "10", "100", "1000")) +
  # scale_y_continuous(limits = c(0, 1.25), breaks = c(0, 0.25, 0.5, 0.75, 1.0)) +
  labs(x = "DOP Concentration (uM)", y = "Normalized Optical Flow (Percent of Control)") +
  # facet_grid(Drug ~ .) +
  theme_bw(base_size = 16, base_family = "Helvetica") +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank())
dop_plot

save_plot("../plots/all_dop.pdf", dop_plot, base_height = 10, base_aspect_ratio = 1.2)


###################################################################################################

#                                         IVM experiment

###################################################################################################

# set directory containing output
exp <- "20180518"
filedir <- paste0("~/Box Sync/ZamanianLab/Data/WormvizOutput/Bpahangi/", exp)
setwd(filedir)

# experiment parameters
drug = "IVM"
control = "DMSO"

#Get plate design
plate3 <- read.csv("plate_design.csv", header = FALSE, sep = ",")
colnames(plate3) <- c("row","0001","0002","0003","0004","0005","0006","0007","0008","0009","0010","0011","0012")
plate3.m <- plate3 %>%
  slice(2:9) %>%
  gather(col, Treatment, 2:13) %>% 
  mutate(Well = paste0(row, col)) %>%
  select(-row, -col) %>% 
  separate(Treatment, c("Drug", "Dose", "Rep"), sep = "_", remove = FALSE) %>%
  filter(Drug == drug | Drug == control)

data3 <- read.csv(paste0(exp, ".csv"), header = TRUE) %>%
  select(Well, Motility = Normalized.Motility) %>%
  left_join(., plate3.m) %>%
  filter(!is.na(Drug)) %>%
  group_by(Drug, Dose)

#mean data / plot
data3_mean <- summarise(data3, Mean = mean(Motility))

#normalize to control
data3 <- mutate(data3, Normalized.Motility = Motility / data3_mean$Mean[1]) %>%
  filter(Drug != control)

lookup <- data.frame(Dose = c("10-4", "10-5", "10-6", "10-7", "10-8", "10-9"),
                     DEC = c(100, 10, 1, 0.1, 0.01, 0.001))

data3 <- data3 %>%
  left_join(., lookup)

ivm_plot <- ggplot(data3, aes(x = DEC, y = Normalized.Motility)) + 
  geom_boxplot(aes(group = DEC), alpha = 0.6) +
  geom_point(aes(color = Rep)) +
  geom_smooth(method = loess, se = FALSE) +
  scale_x_log10(limits = c(0.0004, 300), 
                breaks = c(0.001, 0.01, 0.1, 1, 10, 100),
                labels = c("0.001", "0.01", "0.1", "1", "10", "100")) +
  # scale_y_continuous(limits = c(0, 1.25), breaks = c(0, 0.25, 0.5, 0.75, 1.0)) +
  labs(x = "IVM Concentration (uM)", y = "Normalized Optical Flow (Percent of Control)") +
  # facet_grid(Drug ~ .) +
  theme_bw(base_size = 16, base_family = "Helvetica") +
  theme(panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank())
ivm_plot

save_plot("../plots/ivm.pdf", ivm_plot, base_height = 10, base_aspect_ratio = 1.2)
