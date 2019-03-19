
library(dplyr)
library(ggplot2)
library(RColorBrewer)

MODEL_1 = "(0.05,1)"
MODEL_2 = "(0.025,0.5)"
MODEL_3 = "(0.05,1(0.05,0.5))"
MODEL_4 = "(ORG-0.05,1)"
MODEL_5 = "REL(0.05,1)"
MODEL_6 = "(0.001,1)"
MODEL_7 = "(0,1)"

default_intervention_size = 10

# load data
st <- read.csv(
  #"data/fb100-data/output/fb100_edgelist_spreading_data_dump_add_random_none_rewired.csv",
  "data/fb100-data/output/fb100_edgelist_spreading_data_dump.csv",
  stringsAsFactors = FALSE
)

smallest.40 <- st %>% group_by(network_id) %>%
  summarise(
    network_size = first(network_size)
  ) %>%
  mutate(
    network_size_rev_rank = rank(-network_size)
  ) %>%
  filter(
    network_size_rev_rank <= 40
    )


st <- st %>%
  #filter(sample_id < 100) %>%
  semi_join(smallest.40)

table(table(st$network_id))
table(st$model)

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
  ## filter(
  ##   network_id %in% names(table(st$network_id))[which(table(st$network_id) == 800)]
  ##   ) %>%
  mutate(
    intervention = intervention_name_map[intervention_type]
  )
table(st_1$network_id)


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

ggsave('figures/fb100/fb100_time_to_spread_ecdf_overall_model_1.pdf',
       overall_ecdf_plot, width = 4.5, height = 3.5)


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
  scale_x_log10(breaks = c(.25, .5, 1, 2, 4, 10), limits = c(.3, 4)) +
  scale_color_manual(values = intervention_colors, drop = FALSE) +
  stat_ecdf(alpha = .4, lwd = .2) +
  ylab("ECDF") +
  xlab("relative time to spread") +
  theme(legend.position = c(0.8, 0.3)) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  ) +
guides(colour = guide_legend(override.aes = list(alpha = .8, lwd = .4)))
many_ecdf_plot

ggsave('figures/fb100/fb100_time_to_spread_many_ecdfs_model_1.pdf',
       many_ecdf_plot, width = 5, height = 4)

# make version for revealing in talk
many_ecdf_plot %+%
  (tmp %>% filter(intervention_type %in% "none"))
ggsave('figures/fb100/fb100_time_to_spread_many_ecdfs_reveal_1.pdf',
       width = 5, height = 4)
many_ecdf_plot %+%
  (tmp %>% filter(intervention_type %in% c("none", "rewired")))
ggsave('figures/fb100/fb100_time_to_spread_many_ecdfs_reveal_2.pdf',
       width = 5, height = 4)
many_ecdf_plot %+%
  (tmp %>% filter(intervention_type %in% c("none", "rewired", "triad_addition")))
ggsave('figures/fb100/fb100_time_to_spread_many_ecdfs_reveal_3.pdf',
       width = 5, height = 4)
many_ecdf_plot %+%
  (tmp)
ggsave('figures/fb100/fb100_time_to_spread_many_ecdfs_reveal_4.pdf',
       width = 5, height = 4)


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
    cdf_diff_triad_addition = cdf - cdf[intervention_type == "triad_addition"]
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
  geom_line(alpha = .5, lwd = .2) +
  ylab(expression(ECDF[rewired] - ECDF[original])) +
  xlab("time to spread") +
  theme(legend.position = "none") +
  guides(colour = guide_legend(override.aes = list(alpha = .7))) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )
many_diff_in_ecdf_plot

ggsave('figures/fb100/fb100_time_to_spread_many_diff_in_ecdfs_rewired_vs_none.pdf',
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
  geom_line(alpha = .5, lwd = .2) +
  ylab(expression(ECDF[random] - ECDF[triadic])) +
  xlab("time to spread") +
  theme(legend.position = "none") +
  guides(colour = guide_legend(override.aes = list(alpha = .7))) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )
many_diff_in_ecdf_plot_random_vs_triad

ggsave('figures/fb100/fb100_time_to_spread_many_diff_in_ecdfs_random_vs_triadic.pdf',
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

ggsave('figures/fb100/fb100_time_to_spread_many_diff_in_ecdfs_random_vs_none.pdf',
       many_diff_in_ecdf_plot, width = 5, height = 4)


# combine plots
many_with_insets <- many_ecdf_plot +
  theme(legend.position = "none") +
  scale_x_log10(breaks = c(.1, .25, .5, 1, 2, 4, 10), limits = c(.4, 14)) +
  annotation_custom(
    ggplotGrob(
      overall_ecdf_plot +
        theme(legend.position = "none") +
        scale_y_continuous(breaks = c(0, .5, 1)) +
        scale_x_log10(breaks = c(1, 3, 10, 30, 100), limits = c(1, 20)) +
        theme(text = element_text(size=7),
              rect = element_rect(fill = "transparent"),
              plot.background = element_rect(color = NA))
   ), 
    xmin = .3, xmax = log10(16), ymin = -0.03, ymax = -0.03 + .5
  ) +
  annotation_custom(
    ggplotGrob(
      #many_diff_in_ecdf_plot +
      many_diff_in_ecdf_plot_random_vs_triad +
        theme(legend.position = "none") +
        xlab(NULL) +
        scale_y_continuous(breaks = c(0, .5)) +
        scale_x_log10(breaks = c(1, 3, 10, 30, 100), limits = c(1, 30)) +
        theme(text = element_text(size=7),
              rect = element_rect(fill = "transparent"),
              plot.background = element_rect(color = NA))
   ), 
    xmin = .3, xmax = log10(16), ymin = .44, ymax = .44 + .5
  )
many_with_insets

ggsave('figures/fb100/fb100_time_to_spread_many_ecdfs_with_insets.pdf',
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
    ~ network_id
    ) +
  xlab("time to spread")
sample_ecdf_plot

ggsave('figures/fb100/time_to_spread_sample_networks_ecdfs.pdf',
       sample_ecdf_plot, width = 12, height = 10)

