
library(latex2exp)
library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(Hmisc)
library(plotrix)

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


#cwd <- dirname(rstudioapi::getSourceEditorContext()$path)
cwd <- "."

intervention_name_map <- c(
  "none" = "original",
  "random_addition" = "random addition",
  "triad_addition" = "triadic addition",
  "rewired" = "rewired"
)

network_group_name_map <- c(
  "banerjee_combined_edgelist_" = "Banerjee et al.\n (2013)",
  "cai_edgelist_" = "Cai et al. (2015)",
  "chami_advice_edgelist_" = "Chami et al. (2017) \n advice network",
  "chami_friendship_edgelist_" = "Chami et al. (2017) \n friendship network",
  "fb100_edgelist_" = "Traud et al. (2012)"
)


# load and summarize each set data set:

# cai:

cai_data <- read.csv(
  paste(cwd,"/data/cai-data/output/cai_edgelist_spreading_data_dump.csv",sep=""),
  stringsAsFactors = FALSE
)

cai_filtered_data <- cai_data %>%
  filter(network_size > 10) %>%
  filter(model == MODEL_1) %>% 
  filter(intervention_size %in% c(0, default_intervention_size)) %>% 
  select(network_group,intervention_type,time_to_spread,network_id)%>%
  mutate(
    intervention = intervention_name_map[intervention_type]
  )%>%
  mutate(
    network_group = network_group_name_map[network_group]
  ) %>% 
  mutate(intervention = as.factor(intervention))%>%
  mutate(intervention = factor(intervention,levels(intervention)[c(1,3,4,2)]))

# chami advice:

chami_advice_data <- read.csv(
  paste(cwd,"/data/chami-advice-data/output/chami_advice_edgelist_spreading_data_dump.csv",sep=""),
  stringsAsFactors = FALSE
)

chami_advice_filtered_data <- chami_advice_data %>%
  filter(network_size > 10) %>%
  filter(model == MODEL_1) %>% 
  filter(intervention_size %in% c(0, default_intervention_size)) %>% 
  select(network_group,intervention_type,time_to_spread,network_id)%>%
  mutate(
    intervention = intervention_name_map[intervention_type]
  )%>%
  mutate(
    network_group = network_group_name_map[network_group]
  ) %>% 
  mutate(intervention = as.factor(intervention))%>%
  mutate(intervention = factor(intervention,levels(intervention)[c(1,3,4,2)]))

# chami friendship:

chami_friendship_data <- read.csv(
  paste(cwd,"/data/chami-friendship-data/output/chami_friendship_edgelist_spreading_data_dump.csv",sep=""),  
  stringsAsFactors = FALSE
)

chami_friendship_filtered_data <- chami_friendship_data %>%
  filter(network_size > 10) %>%
  filter(model == MODEL_1) %>% 
  filter(intervention_size %in% c(0, default_intervention_size)) %>% 
  select(network_group,intervention_type,time_to_spread,network_id)%>%
  mutate(
    intervention = intervention_name_map[intervention_type]
  )%>%
  mutate(
    network_group = network_group_name_map[network_group]
  ) %>% 
  mutate(intervention = as.factor(intervention))%>%
  mutate(intervention = factor(intervention,levels(intervention)[c(1,3,4,2)]))

# banerjee combined:

banerjee_combined_data <- read.csv(
  paste(cwd,"/data/banerjee-combined-data/output/banerjee_combined_edgelist_spreading_data_dump.csv",sep=""),
  stringsAsFactors = FALSE
)

banerjee_combined_filtered_data <- banerjee_combined_data %>%
  filter(network_size > 10) %>%
  filter(model == MODEL_1) %>% 
  filter(intervention_size %in% c(0, default_intervention_size)) %>% 
  select(network_group,intervention_type,time_to_spread,network_id)%>%
  mutate(
    intervention = intervention_name_map[intervention_type]
  )%>%
  mutate(
    network_group = network_group_name_map[network_group]
  ) %>%  
  mutate(intervention = as.factor(intervention))%>%
  mutate(intervention = factor(intervention,levels(intervention)[c(1,3,4,2)]))

# fb40:

fb_data <- read.csv(
  paste(cwd,"/data/fb100-data/output/fb100_edgelist_spreading_data_dump.csv",sep=""),
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
  filter(intervention_size %in% c(0, default_intervention_size)) %>% 
  select(network_group,intervention_type,time_to_spread,network_id)%>%
  mutate(
    intervention = intervention_name_map[intervention_type]
  )%>%
  mutate(
    network_group = network_group_name_map[network_group]
  ) %>%  
  mutate(intervention = as.factor(intervention))%>%
  mutate(intervention = factor(intervention,levels(intervention)[c(1,3,4,2)]))

all_filtered_data = rbind(
  cai_filtered_data,
  chami_advice_filtered_data ,
  chami_friendship_filtered_data,
  banerjee_combined_filtered_data,
  fb_filtered_data
) %>% 
  mutate(network_group = as.factor(network_group))%>%
  mutate(network_group = factor(network_group, levels(network_group)[c(2,1,5,4,3)]))



q <- qnorm(1 - .05/2)
all_summaries <- all_filtered_data %>%
  group_by(network_group, network_id) %>%
  mutate(
    time_to_spread_diff = time_to_spread - time_to_spread[intervention == "original"]
  ) %>%
  group_by(network_group, intervention) %>%
  summarise(
    time_to_spread_mean = mean(time_to_spread),
    time_to_spread_mean_diff = mean(time_to_spread_diff),
    time_to_spread_se = std.error(time_to_spread),
    time_to_spread_diff_se = std.error(time_to_spread_diff),
    time_to_spread_ub = time_to_spread_mean + q * time_to_spread_se,
    time_to_spread_lb = time_to_spread_mean - q * time_to_spread_se,
    time_to_spread_ub_diff = time_to_spread_mean + q * time_to_spread_diff_se,
    time_to_spread_lb_diff = time_to_spread_mean - q * time_to_spread_diff_se
    )

all_summaries_group_by_id <- all_filtered_data %>%
  group_by(network_group, network_id, intervention) %>%
  summarise(
    time_to_spread = mean(time_to_spread)
    ) %>%
  group_by(network_group, network_id) %>%
  mutate(
    time_to_spread_diff = time_to_spread - time_to_spread[intervention == "original"]
  ) %>%
  group_by(network_group, intervention) %>%
  summarise(
    time_to_spread_mean = mean(time_to_spread),
    time_to_spread_mean_diff = mean(time_to_spread_diff),
    time_to_spread_se = std.error(time_to_spread),
    time_to_spread_diff_se = std.error(time_to_spread_diff),
    time_to_spread_ub = time_to_spread_mean + q * time_to_spread_se,
    time_to_spread_lb = time_to_spread_mean - q * time_to_spread_se,
    time_to_spread_ub_diff = time_to_spread_mean + q * time_to_spread_diff_se,
    time_to_spread_lb_diff = time_to_spread_mean - q * time_to_spread_diff_se
    )


write.csv(all_summaries,
          paste(cwd,"/data/all-spreading-time-summaries/all_summaries.csv",sep=""))


write.csv(all_summaries_group_by_id,
          paste(cwd,"/data/all-spreading-time-summaries/all_summaries_group_by_id.csv",sep=""))

write.csv(all_filtered_data,
          paste(cwd,"/data/all-spreading-time-summaries/all_filtered_data.csv",sep=""))

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

intervention_shapes <- c(
  "original" = 21,
  "random addition" = 22,
  "triadic addition" = 23,
  "rewired" = 24
)

all_summaries_plot <- ggplot(
  aes(x = network_group, y=time_to_spread_mean, color=intervention),
  data = all_summaries, #%>%filter(group != "bakshy_role_no_feed"),
  xlab='',ylim = c(1,40))+ 
  ylab("time to spread")+
  scale_color_manual(values = intervention_colors) + 
  scale_fill_manual(values = intervention_colors) +
  scale_shape_manual(values = intervention_shapes) +
  geom_pointrange(aes(ymin=time_to_spread_lb, ymax=time_to_spread_ub,shape=intervention),
                  position=position_dodge(width=0.75))+
  coord_cartesian(xlim = c(1,30)) + 
  theme(legend.justification=c(1,1), legend.position=c(0.95,0.7)) + 
  scale_y_log10(breaks = c(3,4,6,10,18,34)) + 
  coord_flip(ylim = c(3,25))

all_summaries_plot

ggsave('figures/spreading_time_summaries/all_summaries_plot.pdf',
       all_summaries_plot,
       width = 6, height = 5)

all_summaries_plot_diff_ci <- all_summaries_group_by_id %>%
  ungroup() %>%
  mutate(
    network_group = factor(network_group, levels = sort(levels(network_group), T)),
    ) %>%
  ggplot(
  aes(
    x = network_group,
    y = time_to_spread_mean,
    ymin = time_to_spread_lb_diff,
    ymax = time_to_spread_ub_diff,
    color = intervention, shape = intervention, fill = intervention
  )
) +
  ylab("time to spread") +
  scale_color_manual(values = intervention_colors) + 
  scale_fill_manual(values = intervention_colors) +
  scale_shape_manual(values = intervention_shapes) +
  geom_point(
    position=position_dodge2(width=0.7, reverse = TRUE),
    size = 2
  ) +
  geom_linerange(
    position=position_dodge2(width=0.7, reverse = TRUE),
    show.legend = F
  ) +
  coord_cartesian(xlim = c(1,30)) + 
  theme(
    legend.justification=c(1, 1),
    legend.position=c(0.95, 0.3),
    legend.title = element_blank(),
    legend.key = element_rect(size = 1),
    legend.key.size = unit(.9, 'lines')
  ) + 
  scale_y_log10(breaks = 2^(2:5)) +
  coord_flip()
all_summaries_plot_diff_ci

ggsave('figures/spreading_time_summaries/all_summaries_plot_diff_ci.pdf',
       all_summaries_plot_diff_ci,
       width = 5, height = 3.5)

all_summaries_group_by_id_plot <- ggplot(
  aes(x = network_group, y=time_to_spread_mean, color=intervention),
  data = all_summaries_group_by_id, #%>%filter(group != "bakshy_role_no_feed"),
  xlab='',ylim = c(1,40))+ 
  ylab("time to spread")+
  scale_color_manual(values = intervention_colors) + 
  scale_fill_manual(values = intervention_colors) +
  scale_shape_manual(values = intervention_shapes) +
  geom_pointrange(aes(ymin=time_to_spread_lb, ymax=time_to_spread_ub,shape=intervention),
                  position=position_dodge(width=0.75))+
  coord_cartesian(xlim = c(1,30)) + 
  theme(legend.justification=c(1,1), legend.position=c(0.95,0.7)) + 
  scale_y_log10(breaks = c(3,4,6,10,18,34)) + 
  coord_flip(ylim = c(3,25))

all_summaries_group_by_id_plot

ggsave(paste(cwd,"/figures/spreading_time_summaries/all_summaries_group_by_id_plot.pdf",sep=""),
       all_summaries_group_by_id_plot
       , width = 5, height = 4)

# compute lower and upper whiskers
#xlim1 = boxplot.stats(all_filtered_data$time_to_spread)$stats[c(1, 5)]

#all_filtered_data$network_group <- c("Banerjee et al.\n (2013)", 
#                                     "Cai et al. (2015)", 
#                                     "Chami et al. (2017) \n Advice Network",
#                                     "Chami et al. (2017) \n Friendship Network",
#                                     "Traud et al. (2012)")

all_summaries_box_plot <- 
  ggplot(data = all_filtered_data, aes(x=network_group, y=time_to_spread,color=intervention),
         xlab='',ylim = c(1,40))+#+ 
  #coord_cartesian(ylim = c(1,40))+
  ylab("time to spread")+
  scale_color_manual(values = intervention_colors) + 
  scale_fill_manual(values = intervention_colors) + 
  scale_shape_manual(values = intervention_shapes) + 
  #geom_boxplot(aes(color=intervention),outlier.shape=NA,coef=0) +
  #stat_summary(fun.y=mean, 
  #             aes(color=intervention,fill=intervention), 
  #             geom="point", 
  #             position=position_dodge(width=0.75), 
  #             shape=16, 
  #             size=5,
  #             show_guide = TRUE) + 
  stat_summary(fun.data = mean_cl_boot, 
               aes(color=intervention), geom = "errorbar", 
               size=6,width=0,
               position=position_dodge(width=0.75), show_guide = TRUE)+ 
  coord_cartesian(xlim = c(1,30)) + 
  theme(legend.justification=c(1,1), legend.position=c(0.95,0.7)) + 
  scale_y_log10(breaks = c(3,4,6,10,18,34)) + 
  coord_flip(ylim = c(3,21))

all_summaries_box_plot

ggsave(paste(cwd,"/figures/spreading_time_summaries/all_summaries_bar_plot_se.pdf",sep=""),
       all_summaries_box_plot, width = 6, height = 5)

