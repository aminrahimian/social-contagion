
library(dplyr)
library(ggplot2)
library(RColorBrewer)

MODEL_1 = "(0.05,1)"
MODEL_2 = "(0.025,0.5)"
MODEL_3 = "(0.05,1(0.05,0.5))"
MODEL_4 = "(ORG-0.05,1)"
MODEL_5 = "REL(0.05,1)"
MODEL_6 = "(0.001,1)"

default_intervention_size = 10

# load data
st <- read.csv(
  "data/cai-data/output/cai_edgelist_spreading_data_dump.csv",
  stringsAsFactors = FALSE
)

table(table(st$network_id))
table(st$model)

nd <- read.csv(
  "data/cai-data/output/cai_edgelist_properties_data_dump.csv",
  stringsAsFactors = FALSE
)

# harmonize naming, add null clustering
nd <- nd %>%
  mutate(
    intervention_type = ifelse(
      intervention_type == "rewiring",
      "rewired", 
      intervention_type
    )
  ) %>%
  group_by(network_group, network_id) %>%
  mutate(
    avg_clustering_null = avg_clustering[intervention_type == "none"],
    density_null = number_edges / network_size^2
    )

# summarise spreading times, merge
nd.tmp <- nd %>%
  filter(intervention_type == "none") %>%
  select(
    -intervention_type, -intervention_size,
    -avg_clustering
  )

combined_summary_null <- st %>%
  filter(network_size > 10) %>%
  group_by(model, network_group, network_id) %>%
  summarise(
    time_to_spread_null_median = median(time_to_spread[intervention_type == "none"]),
    time_to_spread_null_mean = mean(time_to_spread[intervention_type == "none"]),
    time_to_spread_null_max = max(time_to_spread[intervention_type == "none"]),
    time_to_spread_median = median(time_to_spread),
    time_to_spread_mean = mean(time_to_spread),
    time_to_spread_max = max(time_to_spread)
  ) %>%
  inner_join(nd.tmp, by = c("network_group", "network_id")) %>%
  mutate(
    summary_str = sprintf(
      "N: %s, CC: %1.2f",
      network_size,
      avg_clustering_null
      )
  )


# plotting settings
theme_set(theme_bw())
theme_update(
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank()
)
intervention_name_map <- c(
  "none" = "original",
  "random_addition" = "random addition",
  "triad_addition" = "triadic addition",
  "rewired" = "rewired"
)
intervention_colors <- c(
  "original" = "black",
  "random addition" = brewer.pal(8, "Set1")[1],
  "triadic addition" = brewer.pal(8, "Set1")[2],
  "rewired" = brewer.pal(8, "Set1")[5]
)
intervention_shapes <- c(
  "original" = 16,
  "random addition" = 16,
  "triadic addition" = 17,
  "rewired" = 17
)

st_1 <- st %>%
  inner_join(combined_summary_null) %>%
  mutate(
    intervention = intervention_name_map[intervention_type]
    )


# plot ECDF averaging over networks
overall_ecdf_plot <- ggplot(
  aes(x = time_to_spread,
      color = intervention
  ),
  data = st_1 %>% filter(
    model == MODEL_1,
    intervention_size %in% c(0, default_intervention_size)
  )
) +
  scale_color_manual(values = intervention_colors) +
  scale_x_log10() +
  stat_ecdf(lwd = .3) +
  #facet_wrap( ~ factor(intervention_size)) +
  ylab("ECDF") +
  xlab("time to spread") +
  theme(legend.position = c(0.8, 0.3)) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )
overall_ecdf_plot

ggsave('figures/cai/cai_time_to_spread_ecdf_overall_model_1.pdf',
       overall_ecdf_plot, width = 4.5, height = 3.5)

overall_ecdf_plot %+% (
  st_1 %>% filter(
    model == MODEL_2,
    intervention_size %in% c(0, default_intervention_size)
  ))
ggsave('figures/cai/cai_time_to_spread_ecdf_overall_model_2.pdf',
       width = 4.5, height = 3.5)

overall_ecdf_plot %+% (
  st_1 %>% filter(
    model == MODEL_3,
    intervention_size %in% c(0, default_intervention_size)
  )) +
  theme(legend.position = c(0.22, 0.75))
ggsave('figures/cai/cai_time_to_spread_ecdf_overall_model_3.pdf',
       width = 4.5, height = 3.5)

overall_ecdf_plot %+% (
  st_1 %>% filter(
    model == MODEL_4,
    intervention_size %in% c(0, default_intervention_size)
  ))
ggsave('figures/cai/cai_time_to_spread_ecdf_overall_model_4.pdf',
       width = 4.5, height = 3.5)

overall_ecdf_plot %+% (
  st_1 %>% filter(
    model == MODEL_5,
    intervention_size %in% c(0, default_intervention_size)
  ))
ggsave('figures/cai/cai_time_to_spread_ecdf_overall_model_5.pdf',
       width = 4.5, height = 3.5)

overall_ecdf_plot %+% (
  st_1 %>% filter(
    model == MODEL_6,
    intervention_size %in% c(0, default_intervention_size)
  )) +
  theme(legend.position = c(0.22, 0.75))

ggsave('figures/cai/cai_time_to_spread_ecdf_overall_model_6.pdf',
       width = 4.5, height = 3.5)


overall_ecdf_plot_facet = ggplot(
  aes(x = time_to_spread,
      color = intervention 
  ),
  data = st_1 %>% filter(
    intervention_type != "none"
  )
) +
  scale_x_log10() +
  scale_color_manual(values = intervention_colors) +
  stat_ecdf(
    lwd = .6,
    data = st_1 %>% filter(
        intervention_type == "none"
    ) %>%
      select(-intervention_size)
  ) +
  stat_ecdf() +
  facet_grid(intervention_size ~ model, scales = "free_x") +
  ylab("ECDF") +
  xlab("time to spread") +
  theme(legend.position = "bottom") +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )
#overall_ecdf_plot_facet

ggsave('figures/cai/cai_time_to_spread_ecdf_overall_by_model_and_intervention_size.pdf',
       overall_ecdf_plot_facet, width = 12, height = 12)


# overlay ECDF for each network
tmp <- st_1 %>% filter(
  intervention_size %in% c(0, default_intervention_size),
  model == MODEL_1
) %>%
  mutate(
    intervention = factor(intervention)
    )
many_ecdf_plot = ggplot(
  aes(x = time_to_spread / time_to_spread_null_mean,
      color = intervention,
      group = paste(intervention_type, intervention_size, network_id)
  ),
  data = tmp
) +
  scale_x_log10(breaks = c(.25, .5, 1, 2, 4, 10), limits = c(.1, 11)) +
  scale_color_manual(values = intervention_colors, drop = FALSE) +
  stat_ecdf(alpha = .25, lwd = .2) +
  ylab("ECDF") +
  xlab("relative time to spread") +
  theme(legend.position = c(0.8, 0.3)) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  ) +
guides(colour = guide_legend(override.aes = list(alpha = .8, lwd = .4)))
many_ecdf_plot

ggsave('figures/cai/cai_time_to_spread_many_ecdfs_model_1.pdf',
       many_ecdf_plot, width = 5, height = 4)

# make version for revealing in talk
many_ecdf_plot %+%
  (tmp %>% filter(intervention_type %in% "none"))
ggsave('figures/cai/cai_time_to_spread_many_ecdfs_reveal_1.pdf',
       width = 5, height = 4)
many_ecdf_plot %+%
  (tmp %>% filter(intervention_type %in% c("none", "rewired")))
ggsave('figures/cai/cai_time_to_spread_many_ecdfs_reveal_2.pdf',
       width = 5, height = 4)
many_ecdf_plot %+%
  (tmp %>% filter(intervention_type %in% c("none", "rewired", "triad_addition")))
ggsave('figures/cai/cai_time_to_spread_many_ecdfs_reveal_3.pdf',
       width = 5, height = 4)
many_ecdf_plot %+%
  (tmp)
ggsave('figures/cai/cai_time_to_spread_many_ecdfs_reveal_4.pdf',
       width = 5, height = 4)

many_ecdf_plot_2 = ggplot(
  aes(x = time_to_spread / time_to_spread_null_mean,
      color = intervention,
      group = paste(intervention_type, intervention_size, network_id)
  ),
  data = st_1 %>% filter(
    intervention_size %in% c(0, default_intervention_size),
    model == MODEL_2)
) +
  scale_x_log10(breaks = c(.25, .5, 1, 2, 4)) +
  scale_color_manual(values = intervention_colors) +
  stat_ecdf(alpha = .25, lwd = .2) +
  ylab("ECDF") +
  xlab("relative time to spread") +
  theme(legend.position = c(0.8, 0.3)) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  ) +
guides(colour = guide_legend(override.aes = list(alpha = .7)))
many_ecdf_plot_2

ggsave('figures/cai/cai_time_to_spread_many_ecdfs_model_2.pdf',
       many_ecdf_plot_2, width = 5, height = 4)

many_ecdf_plot_3 = ggplot(
  aes(x = time_to_spread / time_to_spread_null_mean,
      color = intervention,
      group = paste(intervention_type, intervention_size, network_id)
  ),
  data = st_1 %>% filter(
    intervention_size %in% c(0, default_intervention_size),
    model == MODEL_2)
) +
  scale_x_log10(breaks = c(.1, .25, .5, 1, 2, 4)) +
  scale_color_manual(values = intervention_colors) +
  stat_ecdf(alpha = .25, lwd = .2) +
  ylab("ECDF") +
  xlab("relative time to spread") +
  theme(legend.position = c(0.8, 0.3)) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  ) +
guides(colour = guide_legend(override.aes = list(alpha = .7)))
many_ecdf_plot_3

ggsave('figures/cai/cai_time_to_spread_many_ecdfs_model_3.pdf',
       many_ecdf_plot_3, width = 5, height = 4)


many_ecdf_plot_facet_by_size = ggplot(
  aes(x = time_to_spread / time_to_spread_null_mean,
      color = intervention,
      group = paste(intervention_type, network_id)      
  ),
  data = st_1 %>% filter(
    intervention_type != "none"
  )
) +
  scale_x_log10(breaks = c(.1, 1, 10)) +
  scale_color_manual(values = intervention_colors) +
  stat_ecdf(
    alpha = .5, lwd = .2,
    data = st_1 %>% filter(
        intervention_type == "none"
    ) %>%
      select(-intervention_size)
  ) +
  stat_ecdf(alpha = .25, lwd = .2) +
  facet_grid(model ~ intervention_size) +
  ylab("ECDF") +
  xlab("relative time to spread") +
  theme(legend.position = "bottom") +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  ) +
guides(colour = guide_legend(override.aes = list(alpha = .7)))
many_ecdf_plot_facet_by_size

ggsave('figures/cai/cai_time_to_spread_many_ecdfs_by_model_and_intervention_size.pdf',
       many_ecdf_plot_facet_by_size, width = 12, height = 8)

###
# difference in ECDFs

st_1_ecdf <- st_1 %>%
  filter(intervention_size %in% c(0, default_intervention_size)) %>%
  group_by(model, network_group, network_id, intervention, intervention_type) %>%
  do({
    data.frame(
      time_to_spread = 1:.$time_to_spread_max,
      cdf = ecdf(.$time_to_spread)(1:.$time_to_spread_max)
    )
  })

st_1_ecdf <- st_1_ecdf %>%
  inner_join(combined_summary_null) %>%
  filter(time_to_spread <= time_to_spread_max)

st_1_ecdf_diff <- st_1_ecdf %>%
  filter(model != MODEL_4) %>%
  group_by(model, network_group, network_id) %>%
  mutate(
    cdf_diff_none = cdf - cdf[intervention_type == "none"],
    cdf_diff_triad_addition = cdf - cdf[intervention_type == "triad_addition"],
    )

many_diff_in_ecdf_plot = ggplot(
  aes(x = time_to_spread,
      y = cdf_diff_none,
      color = intervention,
      group = paste(intervention_type, network_id)
      ),
  data = st_1_ecdf_diff %>%
    filter(intervention_type == "rewired",
           model == MODEL_1)
) +
  scale_color_manual(values = intervention_colors) +
  scale_x_log10() +
  geom_hline(yintercept = 0, lwd = .3, lty = 2, alpha = .6) +
  geom_line(alpha = .25, lwd = .2) +
  ylab(expression(ECDF[rewired] - ECDF[original])) +
  xlab("time to spread") +
  theme(legend.position = "none") +
  guides(colour = guide_legend(override.aes = list(alpha = .7))) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )
many_diff_in_ecdf_plot

ggsave('figures/cai/cai_time_to_spread_many_diff_in_ecdfs_rewired_vs_none.pdf',
       many_diff_in_ecdf_plot, width = 5, height = 4)

many_diff_in_ecdf_plot_random_vs_triad = ggplot(
  aes(x = time_to_spread,
      y = cdf_diff_triad_addition,
      color = intervention,
      group = paste(intervention_type, network_id)
      ),
  data = st_1_ecdf_diff %>%
    filter(intervention_type == "random_addition",
           model == MODEL_1)
) +
  scale_color_manual(values = intervention_colors) +
  scale_x_log10() +
  geom_hline(yintercept = 0, lwd = .3, lty = 2, alpha = .6) +
  geom_line(alpha = .25, lwd = .2) +
  ylab(expression(ECDF[random] - ECDF[triadic])) +
  xlab("time to spread") +
  theme(legend.position = "none") +
  guides(colour = guide_legend(override.aes = list(alpha = .7))) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )
many_diff_in_ecdf_plot_random_vs_triad

ggsave('figures/cai/cai_time_to_spread_many_diff_in_ecdfs_random_vs_triadic.pdf',
       many_diff_in_ecdf_plot_random_vs_triad, width = 5, height = 4)

many_diff_in_ecdf_plot = ggplot(
  aes(x = time_to_spread,
      y = cdf_diff_none,
      color = intervention,
      group = paste(intervention_type, network_id)
      ),
  data = st_1_ecdf_diff %>%
    filter(intervention_type == "random_addition",
           model == MODEL_1)
) +
  scale_color_manual(values = intervention_colors) +
  scale_x_log10() +
  geom_hline(yintercept = 0, lwd = .3, lty = 2, alpha = .5) +
  geom_line(alpha = .25, lwd = .2) +
  ylab(expression(ECDF[random] - ECDF[original])) +
  xlab("time to spread") +
  theme(legend.position = "none") +
  guides(colour = guide_legend(override.aes = list(alpha = .7))) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )
many_diff_in_ecdf_plot

ggsave('figures/cai/cai_time_to_spread_many_diff_in_ecdfs_random_vs_none.pdf',
       many_diff_in_ecdf_plot, width = 5, height = 4)


# combine plots
many_with_insets <- many_ecdf_plot +
  theme(legend.position = "none") +
  scale_x_log10(breaks = c(.1, .25, .5, 1, 2, 4, 10), limits = c(.1, 16)) +
  annotation_custom(
    ggplotGrob(
      overall_ecdf_plot +
        theme(legend.position = "none") +
        scale_y_continuous(breaks = c(0, .5, 1)) +
        scale_x_log10(breaks = c(1, 3, 10, 30, 100), limits = c(min(st_1$time_to_spread), 230)) +
        theme(text = element_text(size=7),
              rect = element_rect(fill = "transparent"),
              plot.background = element_rect(color = NA))
   ), 
    xmin = .3, xmax = log10(20), ymin = -0.03, ymax = -0.03 + .5
  ) +
  annotation_custom(
    ggplotGrob(
      many_diff_in_ecdf_plot_random_vs_triad +
        theme(legend.position = "none") +
        xlab(NULL) +
        scale_y_continuous(breaks = c(0, .5)) +
        scale_x_log10(breaks = c(1, 3, 10, 30, 100), limits = c(min(st_1$time_to_spread), 230)) +
        theme(text = element_text(size=7),
              rect = element_rect(fill = "transparent"),
              plot.background = element_rect(color = NA))
   ), 
    xmin = .3, xmax = log10(20), ymin = .44, ymax = .44 + .5
  )
many_with_insets

ggsave('figures/cai/cai_time_to_spread_many_ecdfs_with_insets.pdf',
       many_with_insets, width = 5, height = 4)



# examine sample of networks
set.seed(1013)
sample_network_ids <- sample(unique(st_1$network_id), 16)
    

sample_ecdf_plot = ggplot(
  aes(x = time_to_spread,
      color = intervention,
      group = paste(intervention_type, network_id)
      ),
  data = st_1 %>% filter(network_id %in% sample_network_ids)
) +
  scale_color_manual(values = intervention_colors) +
  scale_x_log10() +
  stat_ecdf() +
  facet_wrap(
    ~ reorder(network_id, avg_clustering_null)
    ) +
  geom_text(
    aes(label = summary_str,
        x = 1000
        ),
    data = combined_summary_null %>% filter(network_id %in% sample_network_ids),
    inherit.aes = FALSE,
    y = .05,
    hjust = 1, size = 3
  ) +
  xlab("time to spread")
sample_ecdf_plot

ggsave('figures/cai/time_to_spread_sample_networks_ecdfs.pdf',
       sample_ecdf_plot, width = 12, height = 10)




sts <- st_1 %>%
  filter(network_size > 10) %>%
  group_by(model, network_group, network_id,
           intervention_type, intervention, intervention_size) %>%
  summarise(
    time_to_spread_median = median(time_to_spread),
    time_to_spread_mean = mean(time_to_spread),
    time_to_spread_max = max(time_to_spread)
  ) %>%
  inner_join(nd) %>%
  mutate(
    density = number_edges / network_size^2
    )

lm.1 <- felm(
  time_to_spread_mean ~ transitivity + avg_clustering + average_shortest_path_length + I(number_edges/network_size^2)
  | network_id,
  data = sts %>% filter(model == MODEL_1)
)
summary(lm.1)

tmp <- sts %>%
    filter(
      model == MODEL_1,
      network_id %in% 1:15,
      intervention_type %in% c("none", "random_addition", "triad_addition")
    ) %>%
  mutate(
    x = ifelse(intervention_type == "triad_addition", -1, 1) * intervention_size
    ) %>%
  arrange(
    network_id, x
    )

ggplot(
  aes(x = transitivity, y = time_to_spread_mean,
      group = network_id, #paste(network_id, intervention),
      color = intervention_size,
      shape = intervention
      ),
  data = tmp
) +
  scale_y_log10() +
  geom_path(alpha = .5) +
  geom_point(alpha = .5)



tmp <- sts %>%
    filter(
      model == MODEL_1,
      #network_id %in% 50:80,
      intervention_type %in% c("none", "rewired")
    ) %>%
  arrange(
    network_id, intervention_size
    )

ggplot(
  aes(x = transitivity / avg_degree,
      y = time_to_spread_mean,
      group = network_id, #paste(network_id, intervention),
      color = intervention_size,
      shape = intervention
      ),
  data = tmp
) +
  #scale_y_log10() +
  geom_path(alpha = .5) +
  geom_point(alpha = .5)


  
  # load data for different theta:
  
  THETA_2 = "[1, 0, 0, 0]"
  THETA_3 = "[0, 1, 0, 0]"
  THETA_4 = "[0, 0, 1, 0]"
  THETA_5 = "[0, 0, 0, 1]"
  
  THETA_DIST_2 = "[0.7, 0.2, 0.07, 0.03]"
  THETA_DIST_3 = "[0.2, 0.6, 0.15, 0.05]"
  THETA_DIST_4 = "[0.05, 0.15, 0.6, 0.2]"
  THETA_DIST_5 = "[0.03, 0.07, 0.2, 0.7]"
  
  # load data
  st_thetas <- read.csv(
    "data/cai-data/output/cai_edgelist_spreading_data_dump-thetas.csv",
    stringsAsFactors = FALSE
  )
  
  st_thetas_dist <- read.csv(
    "data/cai-data/output/cai_edgelist_spreading_data_dump-theta_dist.csv",
    stringsAsFactors = FALSE
  )
  
  table(table(st_thetas$network_id))
  table(st_thetas$model)
  
  table(table(st_thetas_dist$network_id))
  table(st_thetas_dist$model)
  
  st_thetas <- st_thetas %>%  
    mutate(network_id = as.character(network_id))
  
  st_thetas <- st_thetas %>%
    mutate(
      intervention = intervention_name_map[intervention_type]
    )
  
  st_thetas_dist <- st_thetas_dist %>%  
    mutate(network_id = as.character(network_id))
  
  st_thetas_dist <- st_thetas_dist %>%
    mutate(
      intervention = intervention_name_map[intervention_type]
    )
  
  # plot ECDF averaging over networks for each theta
  overall_ecdf_plot <- ggplot(
    aes(x = time_to_spread,
        color = intervention
    ),
    data = st_thetas %>% filter(
      theta_distribution == THETA_2,
      intervention_size %in% c(0, default_intervention_size)
    )
  ) +
    scale_color_manual(values = intervention_colors) +
    scale_x_log10() +
    stat_ecdf(lwd = .3) +
    ylab("ECDF") +
    xlab("time to spread") +
    theme(legend.position = "topleft") +
    annotation_logticks(
      sides = "b", size = .3,
      short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
    )
  overall_ecdf_plot
  
  ggsave('figures/cai/cai_time_to_spread_ecdf_overall_theta_2.pdf',
         overall_ecdf_plot, width = 4.5, height = 3.5)
  
  overall_ecdf_plot %+% (
    st_thetas %>% filter(
      theta_distribution == THETA_3,
      intervention_size %in% c(0, default_intervention_size)
    ))
  ggsave('figures/cai/cai_time_to_spread_ecdf_overall_theta_3.pdf',
         width = 4.5, height = 3.5)
  
  overall_ecdf_plot %+% (
    st_thetas %>% filter(
      theta_distribution == THETA_4,
      intervention_size %in% c(0, default_intervention_size)
    )) +
    theme(legend.position = c(0.22, 0.75))
  ggsave('figures/cai/cai_time_to_spread_ecdf_overall_theta_4.pdf',
         width = 4.5, height = 3.5)
  
  overall_ecdf_plot %+% (
    st_thetas %>% filter(
      theta_distribution == THETA_5,
      intervention_size %in% c(0, default_intervention_size)
    ))
  ggsave('figures/cai/cai_time_to_spread_ecdf_overall_theta_5.pdf',
         width = 4.5, height = 3.5)
  
  overall_ecdf_plot %+% (
    st_thetas_dist %>% filter(
      theta_distribution == THETA_DIST_2,
      intervention_size %in% c(0, default_intervention_size)
    ))
  ggsave('figures/cai/cai_time_to_spread_ecdf_overall_theta_dist_2.pdf',
         width = 4.5, height = 3.5)
  
  overall_ecdf_plot %+% (
    st_thetas_dist %>% filter(
      theta_distribution == THETA_DIST_3,
      intervention_size %in% c(0, default_intervention_size)
    ))
  ggsave('figures/cai/cai_time_to_spread_ecdf_overall_theta_dist_3.pdf',
         width = 4.5, height = 3.5)
  
  overall_ecdf_plot %+% (
    st_thetas_dist %>% filter(
      theta_distribution == THETA_DIST_4,
      intervention_size %in% c(0, default_intervention_size)
    ))
  ggsave('figures/cai/cai_time_to_spread_ecdf_overall_theta_dist_4.pdf',
         width = 4.5, height = 3.5)
  
  overall_ecdf_plot %+% (
    st_thetas_dist %>% filter(
      theta_distribution == THETA_DIST_5,
      intervention_size %in% c(0, default_intervention_size)
    ))
  ggsave('figures/cai/cai_time_to_spread_ecdf_overall_theta_dist_5.pdf',
         width = 4.5, height = 3.5)
