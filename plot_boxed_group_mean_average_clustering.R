library(latex2exp)
library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(Hmisc)
library(plotrix)

MODEL_1 = "(0.05,1)"
MODEL_2 = "(0.025,0.5)"
MODEL_3 = "(0.05,1(0.05,0.5))"
MODEL_4 = "(ORG-0.05,1)"
MODEL_5 = "REL(0.05,1)"
MODEL_6 = "(0.001,1)"

default_intervention_size = 10


cwd <- dirname(rstudioapi::getSourceEditorContext()$path)
#cwd <- "."

intervention_name_map <- c(
  "none" = "original",
  "random_addition" = "random addition",
  "triad_addition" = "triadic addition",
  "rewired" = "rewired",
  "rewiring" = "rewired"
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

cai_properties_data <- read.csv(
  paste(cwd,"/data/cai-data/output/cai_edgelist_properties_data_dump.csv",sep=""),
  stringsAsFactors = FALSE
)
cai_filtered_properties_data <- cai_properties_data %>%
  filter(network_size > 10) %>%
  filter(intervention_size %in% c(0, default_intervention_size)) %>% 
  select(network_group,intervention_type,avg_clustering,network_id)%>%
  mutate(
    intervention = intervention_name_map[intervention_type]
  )%>%
  mutate(
    network_group = network_group_name_map[network_group]
  ) %>% 
  mutate(intervention = as.factor(intervention))%>%
  mutate(intervention = factor(intervention,levels(intervention)[c(1,3,4,2)]))

# chami advice:

chami_advice_properties_data <- read.csv(
  paste(cwd,"/data/chami-advice-data/output/chami_advice_edgelist_properties_data_dump.csv",sep=""),
  stringsAsFactors = FALSE
)
chami_advice_filtered_properties_data <- chami_advice_properties_data %>%
  filter(network_size > 10) %>%
  filter(intervention_size %in% c(0, default_intervention_size)) %>% 
  select(network_group,intervention_type, avg_clustering,network_id)%>%
  mutate(
    intervention = intervention_name_map[intervention_type]
  )%>%
  mutate(
    network_group = network_group_name_map[network_group]
  ) %>% 
  mutate(intervention = as.factor(intervention))%>%
  mutate(intervention = factor(intervention,levels(intervention)[c(1,3,4,2)]))

# chami friendship:

chami_friendship_properties_data <- read.csv(
  paste(cwd,"/data/chami-friendship-data/output/chami_friendship_edgelist_properties_data_dump.csv",sep=""),  
  stringsAsFactors = FALSE
)

chami_friendship_filtered_properties_data <- chami_friendship_properties_data %>%
  filter(network_size > 10) %>%
  filter(intervention_size %in% c(0, default_intervention_size)) %>% 
  select(network_group,intervention_type,avg_clustering,network_id)%>%
  mutate(
    intervention = intervention_name_map[intervention_type]
  )%>%
  mutate(
    network_group = network_group_name_map[network_group]
  ) %>% 
  mutate(intervention = as.factor(intervention))%>%
  mutate(intervention = factor(intervention,levels(intervention)[c(1,3,4,2)]))

# banerjee combined:

banerjee_combined_properties_data <- read.csv(
  paste(cwd,"/data/banerjee-combined-data/output/banerjee_combined_edgelist_properties_data_dump.csv",sep=""),
  stringsAsFactors = FALSE
)

banerjee_combined_filtered_properties_data <- banerjee_combined_properties_data %>%
  filter(network_size > 10) %>%
  filter(intervention_size %in% c(0, default_intervention_size)) %>% 
  select(network_group,intervention_type,avg_clustering,network_id)%>%
  mutate(
    intervention = intervention_name_map[intervention_type]
  )%>%
  mutate(
    network_group = network_group_name_map[network_group]
  ) %>%  
  mutate(intervention = as.factor(intervention))%>%
  mutate(intervention = factor(intervention,levels(intervention)[c(1,3,4,2)]))

# fb-40 combined:

fb40_combined_properties_data <- read.csv(
  paste(cwd,"/data/fb100-data/output/fb100_edgelist_properties_data_dump.csv",sep=""),
  stringsAsFactors = FALSE
)

fb40_combined_filtered_properties_data <- fb40_combined_properties_data %>%
  filter(network_size > 10) %>%
  filter(intervention_size %in% c(0, default_intervention_size)) %>% 
  select(network_group,intervention_type,avg_clustering,network_id)%>%
  mutate(
    intervention = intervention_name_map[intervention_type]
  )%>%
  mutate(
    network_group = network_group_name_map[network_group]
  ) %>%  
  mutate(intervention = as.factor(intervention))%>%
  mutate(intervention = factor(intervention,levels(intervention)[c(1,3,4,2)]))

all_filtered_properties_data = rbind(
  cai_filtered_properties_data,
  chami_advice_filtered_properties_data ,
  chami_friendship_filtered_properties_data,
  banerjee_combined_filtered_properties_data,
  fb40_combined_filtered_properties_data
) %>% 
  mutate(network_group = as.factor(network_group))%>%
  mutate(network_group = factor(network_group, levels(network_group)[c(5,2,1,4,3)]))



q <- qnorm(1 - .05/2)

all_properties_summaries <- all_filtered_properties_data %>%
  group_by(network_group, network_id) %>%
  mutate(
    avg_clustering_diff = avg_clustering - avg_clustering[intervention == "original"]
  ) %>%
  group_by(network_group, intervention) %>%
  summarise(
    avg_clustering_mean = mean(avg_clustering),
    avg_clustering_mean_diff = mean(avg_clustering_diff),
    avg_clustering_se = std.error(avg_clustering),
    avg_clustering_diff_se = std.error(avg_clustering_diff),
    avg_clustering_ub = avg_clustering_mean + q * avg_clustering_se,
    avg_clustering_lb = avg_clustering_mean - q * avg_clustering_se,
    avg_clustering_ub_diff = avg_clustering_mean + q * avg_clustering_diff_se,
    avg_clustering_lb_diff = avg_clustering_mean - q * avg_clustering_diff_se
  )

all_properties_summaries_group_by_id <- all_filtered_properties_data %>%
  group_by(network_group, network_id, intervention) %>%
  summarise(
    avg_clustering = mean(avg_clustering)
  ) %>%
  group_by(network_group, network_id) %>%
  mutate(
    avg_clustering_diff = avg_clustering - avg_clustering[intervention == "original"]
  ) %>%
  group_by(network_group, intervention) %>%
  summarise(
    avg_clustering_mean = mean(avg_clustering),
    avg_clustering_mean_diff = mean(avg_clustering_diff),
    avg_clustering_se = std.error(avg_clustering),
    avg_clustering_diff_se = std.error(avg_clustering_diff),
    avg_clustering_ub = avg_clustering_mean + q * avg_clustering_se,
    avg_clustering_lb = avg_clustering_mean - q * avg_clustering_se,
    avg_clustering_ub_diff = avg_clustering_mean + q * avg_clustering_diff_se,
    avg_clustering_lb_diff = avg_clustering_mean - q * avg_clustering_diff_se
  )


write.csv(all_properties_summaries,
          paste(cwd,"/data/all-spreading-time-summaries/all_properties_summaries.csv",sep=""))


write.csv(all_properties_summaries_group_by_id,
          paste(cwd,"/data/all-spreading-time-summaries/all_properties_summaries_group_by_id.csv",sep=""))

write.csv(all_filtered_properties_data,
          paste(cwd,"/data/all-spreading-time-summaries/all_filtered_properties_data.csv",sep=""))

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

all_properties_summaries_plot <- ggplot(
  aes(x = avg_clustering_mean, y=network_group, color=intervention),
  data = all_properties_summaries)+
  xlab("average clustering")+ylab('')+
  scale_color_manual(values = intervention_colors) + 
  scale_fill_manual(values = intervention_colors) +
  scale_shape_manual(values = intervention_shapes) +
  geom_pointrange(aes(xmin=avg_clustering_lb, xmax=avg_clustering_ub,shape=intervention),
                  position=position_dodge(width=0.75))+
  theme(legend.justification=c(1,1), legend.position=c(0.97,0.9)) + 
  scale_x_log10(breaks = c(0.1,0.16,0.2,0.23,0.26))

all_properties_summaries_plot

ggsave(paste(cwd,"/figures/spreading_time_summaries/all_properties_summaries_plot.pdf",sep=""),
       all_properties_summaries_plot,
       width = 10, height = 5)

all_properties_summaries_plot_diff_ci <- all_properties_summaries_group_by_id %>%
  ungroup() %>%
  mutate(
    network_group = factor(network_group, levels = sort(levels(network_group), T)),
  ) %>%
  ggplot(
    aes(
      x = network_group,
      y = avg_clustering_mean,
      ymin = avg_clustering_lb_diff,
      ymax = avg_clustering_ub_diff,
      color = intervention, shape = intervention, fill = intervention
    )
  ) +
  ylab("average clustering") +xlab('')+
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
    legend.position=c(0.4, 0.98),
    legend.title = element_blank(),
    legend.key = element_rect(size = 1),
    legend.key.size = unit(.9, 'lines')
  ) + 
  coord_flip(ylim=c(0.1,0.3))
all_properties_summaries_plot_diff_ci

ggsave(paste(cwd,"/figures/spreading_time_summaries/all_properties_summaries_plot_diff_ci.pdf",sep=""),
       all_properties_summaries_plot_diff_ci,
       width = 5, height = 3.5)

all_properties_summaries_group_by_id_plot <- ggplot(
  aes(x = network_group, y=avg_clustering_mean, color=intervention),
  data = all_properties_summaries_group_by_id,
  xlab='',ylim = c(1,40))+ 
  ylab("average clustering")+xlab('')+
  scale_color_manual(values = intervention_colors) + 
  scale_fill_manual(values = intervention_colors) +
  scale_shape_manual(values = intervention_shapes) +
  geom_pointrange(aes(ymin=avg_clustering_lb, ymax=avg_clustering_ub,shape=intervention),
                  position=position_dodge(width=0.75))+
  coord_cartesian(xlim = c(1,30)) + 
  theme(legend.justification=c(1,1), legend.position=c(0.4,0.4)) + 
  coord_flip(ylim = c(0,0.3))

all_properties_summaries_group_by_id_plot

ggsave(paste(cwd,"/figures/spreading_time_summaries/all_properties_summaries_group_by_id_plot.pdf",sep=""),
       all_properties_summaries_group_by_id_plot
       , width = 5, height = 4)

