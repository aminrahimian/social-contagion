
library(dplyr)
library(ggplot2)
library(RColorBrewer)

network_type = "union"

MODEL_1 = "(0.05,1)"

default_intervention_size = 25


cwd <- dirname(rstudioapi::getSourceEditorContext()$path)

# load data
st <- read.csv(
  paste(cwd,sprintf(
    "/data/chami-%s-data/output/chami_%s_edgelist_spreading_data_dump.csv", 
    network_type, network_type),sep=""),
  stringsAsFactors = FALSE
)

nd <- read.csv(
  paste(cwd,sprintf(
    "/data/chami-%s-data/output/chami_%s_edgelist_properties_data_dump.csv", 
    network_type, network_type),sep=""),
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

st_1 <- st %>%
  inner_join(combined_summary_null)


# plotting settings
theme_set(theme_bw())
theme_update(
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank()
)
intervention_colors <- c(
  "none" = "black",
  "random_addition" = brewer.pal(8, "Set1")[1],
  "triad_addition" = brewer.pal(8, "Set1")[2],
  "rewired" = brewer.pal(8, "Set1")[5]
)
intervention_shapes <- c(
  "none" = 16,
  "random_addition" = 16,
  "triad_addition" = 17,
  "rewired" = 17
)


# plot ECDF averaging over networks
overall_ecdf_plot <- ggplot(
  aes(x = time_to_spread,
      color = intervention_type
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

ggsave(paste(cwd,
             '/figures/chami_union/chami_union_time_to_spread_ecdf_overall_model_1.pdf',
             sep=""),
       overall_ecdf_plot, width = 4.5, height = 3.5)

overall_ecdf_plot_facet_by_size = ggplot(
  aes(x = time_to_spread,
      color = intervention_type 
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
  facet_grid(model ~ intervention_size) +
  ylab("ECDF") +
  xlab("time to spread") +
  theme(legend.position = "bottom") +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )
overall_ecdf_plot_facet_by_size

ggsave(paste(cwd,
             '/figures/chami_union/chami_union_time_to_spread_ecdf_overall_by_intervention_size.pdf'
             ,sep=""),
       overall_ecdf_plot_facet_by_size, width = 12, height = 4)


# overlay ECDF for each network
many_ecdf_plot = ggplot(
  aes(x = time_to_spread / time_to_spread_null_mean,
      color = intervention_type,
      group = paste(intervention_type, intervention_size, network_id)
  ),
  data = st_1 %>% filter(
    intervention_size %in% c(0, default_intervention_size),
    model == MODEL_1)
) +
  scale_x_log10(breaks = c(.25, .5, 1, 2, 4, 10)) +
  scale_color_manual(values = intervention_colors) +
  stat_ecdf(alpha = .4, lwd = .2) +
  ylab("ECDF") +
  xlab("relative time to spread") +
  theme(legend.position = c(0.8, 0.3)) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  ) +
  guides(colour = guide_legend(override.aes = list(alpha = .7)))
many_ecdf_plot

ggsave(paste(cwd,
             '/figures/chami_union/chami_union_time_to_spread_many_ecdfs_model_1.pdf',
             sep=""),
       many_ecdf_plot, width = 5, height = 4)

many_ecdf_plot_facet_by_size = ggplot(
  aes(x = time_to_spread / time_to_spread_null_mean,
      color = intervention_type,
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
  stat_ecdf(alpha = .3, lwd = .2) +
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

ggsave(paste(cwd,
             '/figures/chami_union/chami_union_time_to_spread_many_ecdfs_by_intervention_size.pdf'
             ,sep=""),
       many_ecdf_plot_facet_by_size, width = 12, height = 4)

###
# difference in ECDFs

time_seq = 1:max(st_1$time_to_spread)
st_1_ecdf <- st_1 %>%
  filter(intervention_size %in% c(0, default_intervention_size)) %>%
  group_by(model, network_group, network_id, intervention_type) %>%
  do({
    data.frame(
      time_to_spread = time_seq,
      cdf = ecdf(.$time_to_spread)(time_seq)
    )
  })

st_1_ecdf <- st_1_ecdf %>%
  inner_join(combined_summary_null) %>%
  filter(time_to_spread <= time_to_spread_max)

st_1_ecdf_diff <- st_1_ecdf %>%
  group_by(model, network_group, network_id) %>%
  mutate(
    cdf_diff_none = cdf - cdf[intervention_type == "none"],
    cdf_diff_triad_addition = cdf - cdf[intervention_type == "triad_addition"],
  )

many_diff_in_ecdf_plot = ggplot(
  aes(x = time_to_spread,
      y = cdf_diff_none,
      color = intervention_type,
      group = paste(intervention_type, network_id)
  ),
  data = st_1_ecdf_diff %>%
    filter(intervention_type == "rewired",
           model == MODEL_1)
) +
  scale_color_manual(values = intervention_colors) +
  scale_x_log10() +
  geom_hline(yintercept = 0, lwd = .3, lty = 2, alpha = .6) +
  geom_line(alpha = .6, lwd = .2) +
  ylab(expression(ECDF[rewired] - ECDF[original])) +
  xlab("time to spread") +
  theme(legend.position = "none") +
  guides(colour = guide_legend(override.aes = list(alpha = .7))) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )
many_diff_in_ecdf_plot

ggsave(paste(cwd,'/figures/chami_union/chami_union_time_to_spread_many_diff_in_ecdfs_rewired_vs_none.pdf',
             sep=""),
       many_diff_in_ecdf_plot, width = 5, height = 4)

many_diff_in_ecdf_plot_random_vs_triad = ggplot(
  aes(x = time_to_spread,
      y = cdf_diff_triad_addition,
      color = intervention_type,
      group = paste(intervention_type, network_id)
  ),
  data = st_1_ecdf_diff %>%
    filter(intervention_type == "random_addition",
           model == MODEL_1)
) +
  scale_color_manual(values = intervention_colors) +
  scale_x_log10() +
  geom_hline(yintercept = 0, lwd = .3, lty = 2, alpha = .6) +
  geom_line(alpha = .6, lwd = .2) +
  ylab(expression(ECDF[random] - ECDF[triadic])) +
  xlab("time to spread") +
  theme(legend.position = "none") +
  guides(colour = guide_legend(override.aes = list(alpha = .7))) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )
many_diff_in_ecdf_plot_random_vs_triad

ggsave(paste(cwd,
             '/figures/chami_union/chami_union_time_to_spread_many_diff_in_ecdfs_random_vs_triadic.pdf',
             sep=""),
       many_diff_in_ecdf_plot_random_vs_triad, width = 5, height = 4)

many_diff_in_ecdf_plot = ggplot(
  aes(x = time_to_spread,
      y = cdf_diff_none,
      color = intervention_type,
      group = paste(intervention_type, network_id)
  ),
  data = st_1_ecdf_diff %>%
    filter(intervention_type == "random_addition",
           model == MODEL_1)
) +
  scale_color_manual(values = intervention_colors) +
  scale_x_log10() +
  geom_hline(yintercept = 0, lwd = .3, lty = 2, alpha = .5) +
  geom_line(alpha = .5, lwd = .2) +
  ylab(expression(ECDF[random] - ECDF[original])) +
  xlab("time to spread") +
  theme(legend.position = "none") +
  guides(colour = guide_legend(override.aes = list(alpha = .7))) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )
many_diff_in_ecdf_plot

ggsave(paste(cwd,
             '/figures/chami_union/chami_union_time_to_spread_many_diff_in_ecdfs_random_vs_none.pdf',
             sep=""),
       many_diff_in_ecdf_plot, width = 5, height = 4)


# combine plots
many_with_insets <- many_ecdf_plot +
  theme(legend.position = "none") +
  scale_x_log10(breaks = c(.1, .25, .5, 1, 2, 4, 10), limits = c(.25, 12)) +
  annotation_custom(
    ggplotGrob(
      overall_ecdf_plot +
        theme(legend.position = "none") +
        scale_y_continuous(breaks = c(0, .5, 1)) +
        scale_x_log10(breaks = c(1, 3, 10, 30, 100), limits = c(min(st_1$time_to_spread), max(st_1$time_to_spread))) +
        theme(text = element_text(size=7),
              rect = element_rect(fill = "transparent"),
              plot.background = element_rect(color = NA))
    ), 
    xmin = .3, xmax = log10(14), ymin = -0.03, ymax = -0.03 + .5
  ) +
  annotation_custom(
    ggplotGrob(
      many_diff_in_ecdf_plot_random_vs_triad +
        theme(legend.position = "none") +
        xlab(NULL) +
        scale_y_continuous(breaks = c(0, .5)) +
        scale_x_log10(breaks = c(1, 3, 10, 30, 100), limits = c(min(st_1$time_to_spread), max(st_1$time_to_spread))) +
        theme(text = element_text(size=7),
              rect = element_rect(fill = "transparent"),
              plot.background = element_rect(color = NA))
    ), 
    xmin = .3, xmax = log10(14), ymin = .44, ymax = .44 + .5
  )
many_with_insets

ggsave(paste(cwd,'/figures/chami_union/chami_union_time_to_spread_many_ecdfs_with_insets.pdf',
             sep=""),
       many_with_insets, width = 5, height = 4)



# examine sample of networks
set.seed(1013)
sample_network_ids <- sample(unique(st_1$network_id), 16)


sample_ecdf_plot = ggplot(
  aes(x = time_to_spread,
      color = intervention_type,
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

ggsave(paste(cwd,'/figures/chami_union/time_to_spread_sample_networks_ecdfs.pdf',sep = ""),
       sample_ecdf_plot, width = 12, height = 10)



###########################################################
###########################################################
############################################################
##############################################################

MODEL_1 = "(0.05,1)"

default_intervention_size = 10

# load data
st <- read.csv(
  sprintf("data/chami-%s-data/output/chami_%s_edgelist_spreading_data_dump.csv", network_type, network_type),
  stringsAsFactors = FALSE
)

nd <- read.csv(
  sprintf("data/chami-%s-data/output/chami_%s_edgelist_properties_data_dump.csv", network_type, network_type),
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

st_1 <- st %>%
  inner_join(combined_summary_null)


# plotting settings
theme_set(theme_bw())
theme_update(
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank()
)
intervention_colors <- c(
  "none" = "black",
  "random_addition" = brewer.pal(8, "Set1")[1],
  "triad_addition" = brewer.pal(8, "Set1")[2],
  "rewired" = brewer.pal(8, "Set1")[5]
)
intervention_shapes <- c(
  "none" = 16,
  "random_addition" = 16,
  "triad_addition" = 17,
  "rewired" = 17
)


# plot ECDF averaging over networks
overall_ecdf_plot <- ggplot(
  aes(x = time_to_spread,
      color = intervention_type
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

ggsave(paste(cwd,
             '/figures/chami_union/chami_union_time_to_spread_ecdf_overall_model_1.pdf',
             sep=""),
       overall_ecdf_plot, width = 4.5, height = 3.5)

overall_ecdf_plot_facet_by_size = ggplot(
  aes(x = time_to_spread,
      color = intervention_type 
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
  facet_grid(model ~ intervention_size) +
  ylab("ECDF") +
  xlab("time to spread") +
  theme(legend.position = "bottom") +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )
overall_ecdf_plot_facet_by_size

ggsave(paste(cwd,
             'figures/chami_union/chami_union_time_to_spread_ecdf_overall_by_intervention_size.pdf',
             sep=""),
       overall_ecdf_plot_facet_by_size, width = 12, height = 4)


# overlay ECDF for each network
many_ecdf_plot = ggplot(
  aes(x = time_to_spread / time_to_spread_null_mean,
      color = intervention_type,
      group = paste(intervention_type, intervention_size, network_id)
  ),
  data = st_1 %>% filter(
    intervention_size %in% c(0, default_intervention_size),
    model == MODEL_1)
) +
  scale_x_log10(breaks = c(.25, .5, 1, 2, 4, 10)) +
  scale_color_manual(values = intervention_colors) +
  stat_ecdf(alpha = .4, lwd = .2) +
  ylab("ECDF") +
  xlab("relative time to spread") +
  theme(legend.position = c(0.8, 0.3)) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  ) +
guides(colour = guide_legend(override.aes = list(alpha = .7)))
many_ecdf_plot

ggsave(paste(cwd,
             '/figures/chami_union/chami_union_time_to_spread_many_ecdfs_model_1.pdf',
             sep = ""),
       many_ecdf_plot, width = 5, height = 4)

many_ecdf_plot_facet_by_size = ggplot(
  aes(x = time_to_spread / time_to_spread_null_mean,
      color = intervention_type,
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
  stat_ecdf(alpha = .3, lwd = .2) +
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

ggsave(paste(cwd,
             '/figures/chami_union/chami_union_time_to_spread_many_ecdfs_by_intervention_size.pdf',
             sep=""),
       many_ecdf_plot_facet_by_size, width = 12, height = 4)

###
# difference in ECDFs

time_seq = 1:max(st_1$time_to_spread)
st_1_ecdf <- st_1 %>%
  filter(intervention_size %in% c(0, default_intervention_size)) %>%
  group_by(model, network_group, network_id, intervention_type) %>%
  do({
    data.frame(
      time_to_spread = time_seq,
      cdf = ecdf(.$time_to_spread)(time_seq)
    )
  })

st_1_ecdf <- st_1_ecdf %>%
  inner_join(combined_summary_null) %>%
  filter(time_to_spread <= time_to_spread_max)

st_1_ecdf_diff <- st_1_ecdf %>%
  group_by(model, network_group, network_id) %>%
  mutate(
    cdf_diff_none = cdf - cdf[intervention_type == "none"],
    cdf_diff_triad_addition = cdf - cdf[intervention_type == "triad_addition"],
    )

many_diff_in_ecdf_plot = ggplot(
  aes(x = time_to_spread,
      y = cdf_diff_none,
      color = intervention_type,
      group = paste(intervention_type, network_id)
      ),
  data = st_1_ecdf_diff %>%
    filter(intervention_type == "rewired",
           model == MODEL_1)
) +
  scale_color_manual(values = intervention_colors) +
  scale_x_log10() +
  geom_hline(yintercept = 0, lwd = .3, lty = 2, alpha = .6) +
  geom_line(alpha = .6, lwd = .2) +
  ylab(expression(ECDF[rewired] - ECDF[original])) +
  xlab("time to spread") +
  theme(legend.position = "none") +
  guides(colour = guide_legend(override.aes = list(alpha = .7))) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )
many_diff_in_ecdf_plot

ggsave(paste(cwd,
             '/figures/chami_union/chami_union_time_to_spread_many_diff_in_ecdfs_rewired_vs_none.pdf',
             sep=""),
       many_diff_in_ecdf_plot, width = 5, height = 4)

many_diff_in_ecdf_plot_random_vs_triad = ggplot(
  aes(x = time_to_spread,
      y = cdf_diff_triad_addition,
      color = intervention_type,
      group = paste(intervention_type, network_id)
      ),
  data = st_1_ecdf_diff %>%
    filter(intervention_type == "random_addition",
           model == MODEL_1)
) +
  scale_color_manual(values = intervention_colors) +
  scale_x_log10() +
  geom_hline(yintercept = 0, lwd = .3, lty = 2, alpha = .6) +
  geom_line(alpha = .6, lwd = .2) +
  ylab(expression(ECDF[random] - ECDF[triadic])) +
  xlab("time to spread") +
  theme(legend.position = "none") +
  guides(colour = guide_legend(override.aes = list(alpha = .7))) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )
many_diff_in_ecdf_plot_random_vs_triad

ggsave(paste(cwd,
             '/figures/chami_union/chami_union_time_to_spread_many_diff_in_ecdfs_random_vs_triadic.pdf',
             sep=""),
       many_diff_in_ecdf_plot_random_vs_triad, width = 5, height = 4)

many_diff_in_ecdf_plot = ggplot(
  aes(x = time_to_spread,
      y = cdf_diff_none,
      color = intervention_type,
      group = paste(intervention_type, network_id)
      ),
  data = st_1_ecdf_diff %>%
    filter(intervention_type == "random_addition",
           model == MODEL_1)
) +
  scale_color_manual(values = intervention_colors) +
  scale_x_log10() +
  geom_hline(yintercept = 0, lwd = .3, lty = 2, alpha = .5) +
  geom_line(alpha = .5, lwd = .2) +
  ylab(expression(ECDF[random] - ECDF[original])) +
  xlab("time to spread") +
  theme(legend.position = "none") +
  guides(colour = guide_legend(override.aes = list(alpha = .7))) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )
many_diff_in_ecdf_plot

ggsave(paste(cwd,
             '/figures/chami_union/chami_union_time_to_spread_many_diff_in_ecdfs_random_vs_none.pdf',
             sep=""),
       many_diff_in_ecdf_plot, width = 5, height = 4)


# combine plots
many_with_insets <- many_ecdf_plot +
  theme(legend.position = "none") +
  scale_x_log10(breaks = c(.1, .25, .5, 1, 2, 4, 10), limits = c(.25, 12)) +
  annotation_custom(
    ggplotGrob(
      overall_ecdf_plot +
        theme(legend.position = "none") +
        scale_y_continuous(breaks = c(0, .5, 1)) +
        scale_x_log10(breaks = c(1, 3, 10, 30, 100), limits = c(min(st_1$time_to_spread), max(st_1$time_to_spread))) +
        theme(text = element_text(size=7),
              rect = element_rect(fill = "transparent"),
              plot.background = element_rect(color = NA))
   ), 
    xmin = .3, xmax = log10(14), ymin = -0.03, ymax = -0.03 + .5
  ) +
  annotation_custom(
    ggplotGrob(
      many_diff_in_ecdf_plot_random_vs_triad +
        theme(legend.position = "none") +
        xlab(NULL) +
        scale_y_continuous(breaks = c(0, .5)) +
        scale_x_log10(breaks = c(1, 3, 10, 30, 100), limits = c(min(st_1$time_to_spread), max(st_1$time_to_spread))) +
        theme(text = element_text(size=7),
              rect = element_rect(fill = "transparent"),
              plot.background = element_rect(color = NA))
   ), 
    xmin = .3, xmax = log10(14), ymin = .44, ymax = .44 + .5
  )
many_with_insets

ggsave(paste(cwd,'/figures/chami_union/chami_union_time_to_spread_many_ecdfs_with_insets.pdf',sep=""),
       many_with_insets, width = 5, height = 4)



# examine sample of networks
set.seed(1013)
sample_network_ids <- sample(unique(st_1$network_id), 16)
    

sample_ecdf_plot = ggplot(
  aes(x = time_to_spread,
      color = intervention_type,
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

ggsave(paste(cwd,
             '/figures/chami_union/time_to_spread_sample_networks_ecdfs.pdf',sep=""),
       sample_ecdf_plot, width = 12, height = 10)

