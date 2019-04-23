
library(latex2exp)
library(dplyr)
library(ggplot2)
library(RColorBrewer)


# load spreading data

# each dataset will have a column called k for the number of reinforcing signals
# and another column called ratio_k for the ratio of adoptions at k to adoptions at k-1
# any additional column is dropped


MODEL_1 = "(0.05,1)"
MODEL_2 = "(0.025,0.5)"
MODEL_3 = "(0.05,1(0.05,0.5))"
MODEL_4 = "(ORG-0.05,1)"
MODEL_5 = "REL(0.05,1)"
MODEL_6 = "(0.001,1)"

default_intervention_size = 10

intervention_name_map <- c(
  "none" = "original",
  "random_addition" = "random addition",
  "triad_addition" = "triadic addition",
  "rewired" = "rewired"
)

network_group_name_map <- c(
  "banerjee_combined_edgelist_" = "Banerjee et. al.\n (2013)",
  "cai_edgelist_" = "Cai et. al. (2015)",
  "chami_advice_edgelist_" = "Chami et. al. (2017) \n Advice Network",
  "chami_friendship_edgelist_" = "Chami et. al. (2017) \n Friendship Network",
  "fb100_edgelist_" = "Traud et. al. (2012)"
)


# load and summarize each set data set:

# cai:

cai_data <- read.csv(
  "data/cai-data/output/cai_edgelist_spreading_data_dump.csv",
  stringsAsFactors = FALSE
)

cai_filtered_data <- cai_data %>%
  filter(network_size > 10) %>%
  filter(model == MODEL_1) %>% 
  select(network_group,intervention_type,time_to_spread)%>%
  mutate(
    intervention = intervention_name_map[intervention_type]
  )%>%
  mutate(
    network_group = network_group_name_map[network_group]
  ) %>% select(network_group,intervention,time_to_spread) %>% 
  mutate(intervention = as.factor(intervention))%>%
  mutate(intervention = factor(intervention,levels(intervention)[c(1,3,4,2)]))

cai_summary_data <- cai_filtered_data %>%
  group_by(network_group,intervention) %>%
  summarise(
    time_to_spread_mean = mean(time_to_spread),
    time_to_spread_sd = sd(time_to_spread)
  )

# chami advice:

chami_advice_data <- read.csv(
  "data/chami-advice-data/output/chami_advice_edgelist_spreading_data_dump.csv",
  stringsAsFactors = FALSE
)

chami_advice_filtered_data <- chami_advice_data %>%
  filter(network_size > 10) %>%
  filter(model == MODEL_1) %>% 
  select(network_group,intervention_type,time_to_spread)%>%
  mutate(
    intervention = intervention_name_map[intervention_type]
  )%>%
  mutate(
    network_group = network_group_name_map[network_group]
  ) %>% select(network_group,intervention,time_to_spread) %>% 
  mutate(intervention = as.factor(intervention))%>%
  mutate(intervention = factor(intervention,levels(intervention)[c(1,3,4,2)]))


chami_advice_summary_data <- chami_advice_filtered_data %>%
  group_by(network_group,intervention) %>%
  summarise(
    time_to_spread_mean = mean(time_to_spread),
    time_to_spread_sd = sd(time_to_spread)
  )

# chami friendship:

chami_friendship_data <- read.csv(
  "data/chami-friendship-data/output/chami_friendship_edgelist_spreading_data_dump.csv",
  stringsAsFactors = FALSE
)

chami_friendship_filtered_data <- chami_friendship_data %>%
  filter(network_size > 10) %>%
  filter(model == MODEL_1) %>% 
  select(network_group,intervention_type,time_to_spread)%>%
  mutate(
    intervention = intervention_name_map[intervention_type]
  )%>%
  mutate(
    network_group = network_group_name_map[network_group]
  ) %>% select(network_group,intervention,time_to_spread)%>% 
  mutate(intervention = as.factor(intervention))%>%
  mutate(intervention = factor(intervention,levels(intervention)[c(1,3,4,2)]))

chami_friendship_summary_data <- chami_friendship_filtered_data %>%
  group_by(network_group,intervention) %>%
  summarise(
    time_to_spread_mean = mean(time_to_spread),
    time_to_spread_sd = sd(time_to_spread)
  )

# banerjee combined:

banerjee_combined_data <- read.csv(
  "data/banerjee-combined-data/output/banerjee_combined_edgelist_spreading_data_dump.csv",
  stringsAsFactors = FALSE
)

banerjee_combined_filtered_data <- banerjee_combined_data %>%
  filter(network_size > 10) %>%
  filter(model == MODEL_1) %>% 
  select(network_group,intervention_type,time_to_spread)%>%
  mutate(
    intervention = intervention_name_map[intervention_type]
  )%>%
  mutate(
    network_group = network_group_name_map[network_group]
  ) %>% select(network_group,intervention,time_to_spread)%>% 
  mutate(intervention = as.factor(intervention))%>%
  mutate(intervention = factor(intervention,levels(intervention)[c(1,3,4,2)]))

banerjee_combined_summary_data <- banerjee_combined_filtered_data %>%
  group_by(network_group,intervention) %>%
  summarise(
    time_to_spread_mean = mean(time_to_spread),
    time_to_spread_sd = sd(time_to_spread)
  )

# fb40:

fb_data <- read.csv(
  "data/fb100-data/output/fb100_edgelist_spreading_data_dump.csv",
  stringsAsFactors = FALSE
)

smallest.40 <- fb_data %>% group_by(network_id) %>%
  summarise(
    network_size = first(network_size)
  ) %>%
  mutate(
    network_size_rev_rank = rank(-network_size)
  ) %>%
  filter(
    network_size_rev_rank <= 40
  )


fb_data <- fb_data %>%
  #filter(sample_id < 100) %>%
  semi_join(smallest.40)


fb_filtered_data <- fb_data %>%
  filter(network_size > 10) %>%
  filter(model == MODEL_1) %>% 
  select(network_group,intervention_type,time_to_spread)%>%
  mutate(
    intervention = intervention_name_map[intervention_type]
  )%>%
  mutate(
    network_group = network_group_name_map[network_group]
  ) %>% select(network_group,intervention,time_to_spread)%>% 
  mutate(intervention = as.factor(intervention))%>%
  mutate(intervention = factor(intervention,levels(intervention)[c(1,3,4,2)]))

fb_summary_data <- fb_filtered_data %>%
  group_by(network_group,intervention) %>%
  summarise(
    time_to_spread_mean = mean(time_to_spread),
    time_to_spread_sd = sd(time_to_spread)
  )

all_summaries = rbind(cai_summary_data, 
                      chami_advice_summary_data ,
                      chami_friendship_summary_data,
                      banerjee_combined_summary_data,
                      fb_summary_data)


all_filtered_data = rbind(cai_filtered_data,
                          chami_advice_filtered_data ,
                          chami_friendship_filtered_data,
                          banerjee_combined_filtered_data,
                          fb_filtered_data)

write.csv(all_summaries,"data/all-spreading-time-summaries/all_summaries.csv")

write.csv(all_filtered_data,"data/all-spreading-time-summaries/all_filtered_data.csv")



# ploting

theme_set(theme_bw())
theme_update(
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),
  axis.title.y=element_blank()
)

intervention_colors <- c(
  "original" = "black",
  "random addition" = brewer.pal(8, "Set1")[1],
  "triadic addition" = brewer.pal(8, "Set1")[2],
  "rewired" = brewer.pal(8, "Set1")[5]
)

all_summaries_plot <- ggplot(
  aes(x = network_group, y=time_to_spread_mean, color=intervention),
  data = all_summaries, #%>%filter(group != "bakshy_role_no_feed"),
  ylab='') +
  xlab("time to spread")+
  geom_line()+
  geom_point()+
  scale_color_manual(values = intervention_colors)
  #scale_color_discrete(name = "Dataset/study")+
  #ylab("p(k)/p(k-1)") +
  #xlab("k") +
  #theme(legend.position = c(0.75, 0.7))

all_summaries_plot

ggsave('figures/spreading_time_summaries/all_summaries_plot.pdf',
       all_summaries_plot
       , width = 12, height = 10)

# compute lower and upper whiskers
#xlim1 = boxplot.stats(all_filtered_data$time_to_spread)$stats[c(1, 5)]

all_summaries_box_plot <- 
  ggplot(data = all_filtered_data, aes(x=network_group, y=time_to_spread,color=intervention),
         xlab='',ylim = c(1,40))+#+ 
  #coord_cartesian(ylim = c(1,40))+
  ylab("time to spread")+
  scale_color_manual(values = intervention_colors) + 
  #geom_boxplot(aes(color=intervention),outlier.shape=NA,coef=0) +
  stat_summary(fun.y=mean, 
               aes(color=intervention), 
               geom="point", 
               position=position_dodge(width=0.75), 
               shape=13, 
               size=5,
               show_guide = TRUE)+ coord_cartesian(xlim = c(1,30)) + 
  theme(legend.justification=c(1,1), legend.position=c(0.95,0.95))+ 
  scale_y_log10(breaks = c(3,4,6,10,18,34)) + 
  coord_flip(ylim = c(3,21))
  

all_summaries_box_plot

ggsave('figures/spreading_time_summaries/all_summaries_box_plot.pdf',
       all_summaries_box_plot
       , width = 12, height = 10)

