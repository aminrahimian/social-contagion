# plotting spread times versus network size for the empirical networks
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

cwd <- dirname(rstudioapi::getSourceEditorContext()$path)
#cwd <- "."

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
  "chami_union_edgelist_" = "Chami et al. (2017)",
  "fb100_edgelist_" = "Traud et al. (2012)"
)

# load and summarize each set data set:

# cai:

cai_data <- read.csv(
  paste(cwd,"/data/cai-data/output/cai_edgelist_spreading_data_dump.csv",sep=""),
  stringsAsFactors = FALSE
)

# chami advice:

chami_advice_data <- read.csv(
  paste(cwd,"/data/chami-advice-data/output/chami_advice_edgelist_spreading_data_dump.csv",sep=""),
  stringsAsFactors = FALSE
)

# chami friendship:

chami_friendship_data <- read.csv(
  paste(cwd,"/data/chami-friendship-data/output/chami_friendship_edgelist_spreading_data_dump.csv",sep=""),  
  stringsAsFactors = FALSE
)

# chami union:

chami_union_data <- read.csv(
  paste(cwd,"/data/chami-union-data/output/chami_union_edgelist_spreading_data_dump.csv",sep=""),  
  stringsAsFactors = FALSE
)

if("size_of_spread" %in% colnames(chami_union_data))
{
  chami_union_data = subset(chami_union_data, select = -c(size_of_spread) )
}

# banerjee combined:

banerjee_combined_data <- read.csv(
  paste(cwd,"/data/banerjee-combined-data/output/banerjee_combined_edgelist_spreading_data_dump.csv",sep=""),
  stringsAsFactors = FALSE
)

# Fb100:

fb_100_data <- read.csv(
  paste(cwd,"/data/fb100-data/output/fb100_edgelist_spreading_data_dump.csv",sep=""),
  stringsAsFactors = FALSE
)

if("size_of_spread" %in% colnames(fb_100_data))
{
  fb_100_data = subset(fb_100_data, select = -c(size_of_spread) )
}

all_data <-rbind(
  cai_data,
  chami_advice_data ,
  chami_friendship_data,
  banerjee_combined_data,
  fb_100_data
)

q <- qnorm(1 - .05/2)

all_filtered_data_network_size <-all_data%>%
  filter(intervention_size %in% c(0, 10)) %>%
  filter(model == MODEL_1) %>%
  select(network_group,intervention_type,time_to_spread,network_size,network_id)%>%
  mutate(
    intervention = intervention_name_map[intervention_type]
  )%>%
  mutate(
    network_group = network_group_name_map[network_group]
  ) %>%  
  mutate(intervention = as.factor(intervention))%>%
  mutate(intervention = factor(intervention,levels(intervention)[c(1,3,4,2)]))%>% 
  mutate(network_group = as.factor(network_group))%>%
  mutate(network_group = factor(network_group,levels = sort(levels(network_group), T))) 

all_summaries_network_size <- all_filtered_data_network_size %>%
  group_by(network_group, network_id) %>%
  mutate(
    time_to_spread_diff = time_to_spread - 
      time_to_spread[intervention == "original"]
  ) %>% group_by(network_group, intervention) %>%
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

#test for the mean calculation
# cai_data_group_9<-cai_data%>%
#   filter(network_id == 9) %>%
#   filter(intervention_type=='none') %>%
#   filter(model == MODEL_1)
# mean(cai_data_group_9$time_to_spread)

all_summaries_group_by_id_network_size <- all_filtered_data_network_size %>%
  group_by(network_group, network_id) %>%
  mutate(
    time_to_spread_diff = time_to_spread
    - time_to_spread[intervention == "original"]
  ) %>%
  group_by(network_group, network_id,intervention, network_size) %>%
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
# ploting

theme_set(theme_bw())
theme_update(
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank()
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


# Use 95% confidence interval instead of SEM
all_summaries_network_size_plot_line_error_bar <- ggplot(
  aes(x = network_size, 
      y=time_to_spread_mean, 
      color=intervention, 
      shape=intervention,
      fill=intervention),
  data = all_summaries_group_by_id_network_size%>%filter(network_group=="Cai et al. (2015)")) +
  scale_color_manual(name = "Cai et al. (2015)", values = intervention_colors) + 
  scale_fill_manual(name = "Cai et al. (2015)", values = intervention_colors) +
  scale_shape_manual(name = "Cai et al. (2015)", values = intervention_shapes) +
  geom_errorbar(aes(ymin=time_to_spread_lb, ymax=time_to_spread_ub), 
                width=.1, position=position_dodge(width=0.75)) +
  geom_line(position=position_dodge(width=0.75)) +
  geom_point(position=position_dodge(width=0.75),size=2) +
  theme(
    legend.justification=c(1, 1),
    legend.position=c(0.95, 0.98),
    legend.key = element_rect(size = 1),
    legend.key.size = unit(0.3, 'lines')
  ) + xlab("network size") + ylab("mean time to spread")

all_summaries_network_size_plot_line_error_bar

ggsave(paste(cwd,'/figures/spreading_time_summaries/all_summaries_network_size_plot_line_error_bar_cai.pdf',sep=""),
       all_summaries_network_size_plot_line_error_bar,
       width = 7, height = 4)

all_summaries_network_size_plot_line_error_bar <- ggplot(
  aes(x = network_size, 
      y=time_to_spread_mean, 
      color=intervention, 
      shape=intervention,
      fill=intervention),
  data = all_summaries_group_by_id_network_size%>%filter(network_group=="Banerjee et al.\n (2013)")) +
  scale_color_manual(name = "Banerjee et al. (2013)", values = intervention_colors) + 
  scale_fill_manual(name = "Banerjee et al. (2013)", values = intervention_colors) +
  scale_shape_manual(name = "Banerjee et al. (2013)", values = intervention_shapes) +
  geom_errorbar(aes(ymin=time_to_spread_lb, ymax=time_to_spread_ub), 
                width=.1, position=position_dodge(width=0.75)) +
  geom_line(position=position_dodge(width=0.75)) +
  geom_point(position=position_dodge(width=0.75),size=2) + 
  theme(
    legend.justification=c(1, 1),
    legend.position=c(0.99, 0.98),
    legend.key = element_rect(size = 1),
    legend.key.size = unit(2, 'lines')
  )   + xlab("network size") + ylab("mean time to spread")

all_summaries_network_size_plot_line_error_bar

ggsave(paste(cwd,'/figures/spreading_time_summaries/all_summaries_network_size_plot_line_error_bar_banerjee.pdf',sep=""),
       all_summaries_network_size_plot_line_error_bar,
       width = 7, height = 4)

all_summaries_network_size_plot_line_error_bar <- ggplot(
  aes(x = network_size, 
      y=time_to_spread_mean, 
      color=intervention, 
      shape=intervention,
      fill=intervention),
  data = all_summaries_group_by_id_network_size
  %>%filter(network_group=="Chami et al. (2017) \n advice network")) +
  scale_color_manual(name = "Chami et al. (2017) \n advice network", values = intervention_colors) + 
  scale_fill_manual(name = "Chami et al. (2017) \n advice network",values = intervention_colors) +
  scale_shape_manual(name = "Chami et al. (2017) \n advice network",values = intervention_shapes) +
  geom_errorbar(aes(ymin=time_to_spread_lb, ymax=time_to_spread_ub), 
                width=.1, position=position_dodge(width=0.75)) +
  geom_line(position=position_dodge(width=0.75)) +
  geom_point(position=position_dodge(width=0.75),size=2) + 
  theme(
    legend.justification=c(1, 1),
    legend.position=c(0.98, 0.98),
    legend.key = element_rect(size = 1),
    legend.key.size = unit(2, 'lines')
  )   + xlab("network size") + ylab("mean time to spread")

all_summaries_network_size_plot_line_error_bar

ggsave(paste(cwd,'/figures/spreading_time_summaries/all_summaries_network_size_plot_line_error_bar_chami_advice.pdf',sep=""),
       all_summaries_network_size_plot_line_error_bar,
       width = 7, height = 4)

all_summaries_network_size_plot_line_error_bar <- ggplot(
  aes(x = network_size, 
      y=time_to_spread_mean, 
      color=intervention, 
      shape=intervention,
      fill=intervention),
  data = all_summaries_group_by_id_network_size%>%filter(network_group=="Chami et al. (2017) \n friendship network")) +
  scale_color_manual(name = "Chami et al. (2017) \n friendship network", values = intervention_colors) + 
  scale_fill_manual(name = "Chami et al. (2017) \n friendship network", values = intervention_colors) +
  scale_shape_manual(name = "Chami et al. (2017) \n friendship network", values = intervention_shapes) +
  geom_errorbar(aes(ymin=time_to_spread_lb, ymax=time_to_spread_ub), 
                width=.1, position=position_dodge(width=0.75)) +
  geom_line(position=position_dodge(width=0.75)) +
  geom_point(position=position_dodge(width=0.75),size=2) +
  theme(
    legend.justification=c(1, 1),
    legend.position=c(0.9, 0.98),
    legend.key = element_rect(size = 1),
    legend.key.size = unit(2, 'lines')
  )  + xlab("network size") + ylab("mean time to spread")   

all_summaries_network_size_plot_line_error_bar

ggsave(paste(cwd,'/figures/spreading_time_summaries/all_summaries_network_size_plot_line_error_bar_chami_friendship.pdf',sep=""),
       all_summaries_network_size_plot_line_error_bar,
       width = 7, height = 4)

all_summaries_network_size_plot_line_error_bar <- ggplot(
  aes(x = network_size, 
      y=time_to_spread_mean, 
      color=intervention, 
      shape=intervention,
      fill=intervention),
  data = all_summaries_group_by_id_network_size%>%filter(network_group=="Chami et al. (2017) \n friendship network")) +
  scale_color_manual(name = "Chami et al. (2017)", values = intervention_colors) + 
  scale_fill_manual(name = "Chami et al. (2017)", values = intervention_colors) +
  scale_shape_manual(name = "Chami et al. (2017)", values = intervention_shapes) +
  geom_errorbar(aes(ymin=time_to_spread_lb, ymax=time_to_spread_ub), 
                width=.1, position=position_dodge(width=0.75)) +
  geom_line(position=position_dodge(width=0.75)) +
  geom_point(position=position_dodge(width=0.75),size=2) +
  theme(
    legend.justification=c(1, 1),
    legend.position=c(0.95, 0.95),
    legend.key = element_rect(size = 1),
    legend.key.size = unit(2, 'lines')
  )  + xlab("network size") + ylab("mean time to spread")   

all_summaries_network_size_plot_line_error_bar

ggsave(paste(cwd,'/figures/spreading_time_summaries/all_summaries_network_size_plot_line_error_bar_chami_union.pdf',sep=""),
       all_summaries_network_size_plot_line_error_bar,
       width = 7, height = 4)

all_summaries_network_size_plot_line_error_bar <- ggplot(
  aes(x = network_size, 
      y=time_to_spread_mean, 
      color=intervention, 
      shape=intervention,
      fill=intervention),
  data = all_summaries_group_by_id_network_size%>%filter(network_group=="Traud et al. (2012)")) +
  scale_color_manual(name = "Traud et al. (2012)", values = intervention_colors) + 
  scale_fill_manual(name = "Traud et al. (2012)", values = intervention_colors) +
  scale_shape_manual(name = "Traud et al. (2012)", values = intervention_shapes) +
  geom_errorbar(aes(ymin=time_to_spread_lb, ymax=time_to_spread_ub), 
                width=.1, position=position_dodge(width=0.75)) +
  geom_line(position=position_dodge(width=0.75)) +
  geom_point(position=position_dodge(width=0.75),size=2) +
  theme(
    legend.justification=c(1, 1),
    legend.position=c(0.3, 0.99),
    legend.key = element_rect(size = 1),
    legend.key.size = unit(1,2, 'lines')
  )  + xlab("network size") + ylab("mean time to spread") + xlim(c(500,7900))   

all_summaries_network_size_plot_line_error_bar

ggsave(paste(cwd,'/figures/spreading_time_summaries/all_summaries_network_size_plot_line_error_bar_fb100.pdf',sep=""),
       all_summaries_network_size_plot_line_error_bar,
       width = 7, height = 4)


#plot the spread_time vs network_size group by network id
all_summaries_network_size_plot_line_error_bar <- ggplot(
  aes(x = network_size, 
      y=time_to_spread_mean,
      shape=intervention,
      fill=intervention,
      group=network_id),
  data = all_summaries_group_by_id_network_size%>%filter(network_group=="Cai et al. (2015)")) +
  scale_color_manual(name = "Cai et al. (2015)", values = intervention_colors) + 
  scale_fill_manual(name = "Cai et al. (2015)", values = intervention_colors) +
  scale_shape_manual(name = "Cai et al. (2015)", values = intervention_shapes) +
  geom_errorbar(aes(ymin=time_to_spread_lb, ymax=time_to_spread_ub), 
                width=.1) +
  geom_line(colour = "grey") +
  geom_point(size=2) +
  theme(
    legend.justification=c(1, 1),
    legend.position=c(0.85, 0.98),
    legend.key = element_rect(size = 1),
    legend.key.size = unit(0.3, 'lines')
  ) + xlab("network size") + ylab("mean time to spread")

all_summaries_network_size_plot_line_error_bar

ggsave(paste(cwd,'/figures/spreading_time_summaries/all_summaries_network_size_plot_line_error_bar_cai_groupby_id.pdf',sep=""),
       all_summaries_network_size_plot_line_error_bar,
       width = 7, height = 4)

all_summaries_network_size_plot_line_error_bar <- ggplot(
  aes(x = network_size, 
      y=time_to_spread_mean, 
      shape=intervention,
      fill=intervention,
      group=network_id),
  data = all_summaries_group_by_id_network_size%>%filter(network_group=="Banerjee et al.\n (2013)")) +
  scale_color_manual(name = "Banerjee et al. (2013)", values = intervention_colors) + 
  scale_fill_manual(name = "Banerjee et al. (2013)", values = intervention_colors) +
  scale_shape_manual(name = "Banerjee et al. (2013)", values = intervention_shapes) +
  geom_errorbar(aes(ymin=time_to_spread_lb, ymax=time_to_spread_ub), 
                width=.1) +
  geom_point(size=2) + 
  geom_line(colour = "grey") +
  theme(
    legend.justification=c(1, 1),
    legend.position=c(0.99, 0.98),
    legend.key = element_rect(size = 1),
    legend.key.size = unit(2, 'lines')
  )   + xlab("network size") + ylab("mean time to spread")

all_summaries_network_size_plot_line_error_bar

ggsave(paste(cwd,'/figures/spreading_time_summaries/all_summaries_network_size_plot_line_error_bar_banerjee_groupby_id.pdf',sep=""),
       all_summaries_network_size_plot_line_error_bar,
       width = 7, height = 4)

all_summaries_network_size_plot_line_error_bar <- ggplot(
  aes(x = network_size, 
      y=time_to_spread_mean,
      shape=intervention,
      fill=intervention,
      group=network_id),
  data = all_summaries_group_by_id_network_size
  %>%filter(network_group=="Chami et al. (2017) \n advice network")) +
  scale_color_manual(name = "Chami et al. (2017) \n advice network", values = intervention_colors) + 
  scale_fill_manual(name = "Chami et al. (2017) \n advice network",values = intervention_colors) +
  scale_shape_manual(name = "Chami et al. (2017) \n advice network",values = intervention_shapes) +
  geom_errorbar(aes(ymin=time_to_spread_lb, ymax=time_to_spread_ub), 
                width=.1) +
  geom_point(size=2) + 
  geom_line(colour = "grey") +
  theme(
    legend.justification=c(1, 1),
    legend.position=c(0.98, 0.98),
    legend.key = element_rect(size = 1),
    legend.key.size = unit(2, 'lines')
  )   + xlab("network size") + ylab("mean time to spread")

all_summaries_network_size_plot_line_error_bar

ggsave(paste(cwd,'/figures/spreading_time_summaries/all_summaries_network_size_plot_line_error_bar_chami_advice_groupby_id.pdf',sep=""),
       all_summaries_network_size_plot_line_error_bar,
       width = 7, height = 4)

all_summaries_network_size_plot_line_error_bar <- ggplot(
  aes(x = network_size, 
      y=time_to_spread_mean,
      shape=intervention,
      fill=intervention,
      group=network_id),
  data = all_summaries_group_by_id_network_size%>%filter(network_group=="Chami et al. (2017) \n friendship network")) +
  scale_color_manual(name = "Chami et al. (2017) \n friendship network", values = intervention_colors) + 
  scale_fill_manual(name = "Chami et al. (2017) \n friendship network", values = intervention_colors) +
  scale_shape_manual(name = "Chami et al. (2017) \n friendship network", values = intervention_shapes) +
  geom_errorbar(aes(ymin=time_to_spread_lb, ymax=time_to_spread_ub), 
                width=.1) +
  geom_point() +
  geom_line(colour = "grey") +
  theme(
    legend.justification=c(1, 1),
    legend.position=c(0.9, 0.98),
    legend.key = element_rect(size = 1),
    legend.key.size = unit(2, 'lines')
  )  + xlab("network size") + ylab("mean time to spread")   

all_summaries_network_size_plot_line_error_bar

ggsave(paste(cwd,'/figures/spreading_time_summaries/all_summaries_network_size_plot_line_error_bar_chami_friendship_groupby_id.pdf',sep=""),
       all_summaries_network_size_plot_line_error_bar,
       width = 7, height = 4)

all_summaries_network_size_plot_line_error_bar <- ggplot(
  aes(x = network_size, 
      y=time_to_spread_mean,
      shape=intervention,
      fill=intervention,
      group=network_id),
  data = all_summaries_group_by_id_network_size%>%filter(network_group=="Chami et al. (2017) \n friendship network")) +
  scale_color_manual(name = "Chami et al. (2017)", values = intervention_colors) + 
  scale_fill_manual(name = "Chami et al. (2017)", values = intervention_colors) +
  scale_shape_manual(name = "Chami et al. (2017)", values = intervention_shapes) +
  geom_errorbar(aes(ymin=time_to_spread_lb, ymax=time_to_spread_ub), 
                width=.1) +
  geom_line() +
  geom_point(size=2) +
  geom_line(colour = "grey") +
  theme(
    legend.justification=c(1, 1),
    legend.position=c(0.95, 0.95),
    legend.key = element_rect(size = 1),
    legend.key.size = unit(2, 'lines')
  )  + xlab("network size") + ylab("mean time to spread")   

all_summaries_network_size_plot_line_error_bar

ggsave(paste(cwd,'/figures/spreading_time_summaries/all_summaries_network_size_plot_line_error_bar_chami_union_groupby_id.pdf',sep=""),
       all_summaries_network_size_plot_line_error_bar,
       width = 7, height = 4)

all_summaries_network_size_plot_line_error_bar <- ggplot(
  aes(x = network_size, 
      y=time_to_spread_mean,
      shape=intervention,
      fill=intervention,
      group=network_id),
  data = all_summaries_group_by_id_network_size%>%filter(network_group=="Traud et al. (2012)")) +
  scale_color_manual(name = "Traud et al. (2012)", values = intervention_colors) + 
  scale_fill_manual(name = "Traud et al. (2012)", values = intervention_colors) +
  scale_shape_manual(name = "Traud et al. (2012)", values = intervention_shapes) +
  geom_errorbar(aes(ymin=time_to_spread_lb, ymax=time_to_spread_ub), 
                width=.1) +
  geom_point(size=2) +
  geom_line(colour = "grey") +
  theme(
    legend.justification=c(1, 1),
    legend.position=c(0.3, 0.99),
    legend.key = element_rect(size = 1),
    legend.key.size = unit(1,2, 'lines')
  )  + xlab("network size") + ylab("mean time to spread") + xlim(c(500,7900))   

all_summaries_network_size_plot_line_error_bar

ggsave(paste(cwd,'/figures/spreading_time_summaries/all_summaries_network_size_plot_line_error_bar_fb100_groupby_id.pdf',sep=""),
       all_summaries_network_size_plot_line_error_bar,
       width = 7, height = 4)