
this.dir <- dirname(parent.frame(2)$ofile)
setwd(this.dir)

library(dplyr)
library(ggplot2)
library(RColorBrewer)

network_type = "friendship"

# load data
st <- read.csv(
  sprintf("data/chami-%s-data/output/chami_%s_edgelist_data_dump.csv", network_type, network_type),
  stringsAsFactors = FALSE
)

network_type = "friendship"
nd <- read.csv(
  sprintf("data/chami-%s-data/output/chami_%s_edgelist_clustering_data_dump.csv", network_type, network_type),
  stringsAsFactors = FALSE
)

table(table(st$network_id))

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
    density_null = number_edges / (network_size * (network_size - 1) / 2),
    avg_degree_null = number_edges * 2 / network_size
    )

# summarise spreading times, merge
nd.tmp <- nd %>%
  filter(intervention_type == "none") %>%
  select(
    -intervention_type, -intervention_size,
    -avg_clustering, -network_size
  )

combined_summary_null <- st %>%
  filter(network_size > 10) %>%
  group_by(network_group, network_id, network_size) %>%
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
      "N: %s, CC: %1.2f, avg d: %1.2f",
      network_size,
      avg_clustering_null,
      avg_degree_null
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
  "random_addition" = brewer.pal(3, "Set1")[1],
  "triad_addition" = brewer.pal(3, "Set1")[2],
  "rewired" = brewer.pal(5, "Set1")[5]
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
  data = st#_1
) +
  scale_color_manual(values = intervention_colors) +
  stat_ecdf(lwd = .6) +
  facet_wrap( ~ factor(intervention_size)) +
  ylab("ECDF") +
  xlab("time to global spread") +
  theme(legend.position = c(0.8, 0.3))
overall_ecdf_plot

ggsave('figures/chami/time_to_spread_ecdf_overall.pdf',
       overall_ecdf_plot, width = 4.5, height = 3.5)



# overlay ECDF for each network
many_ecdf_plot = ggplot(
  aes(x = time_to_spread / time_to_spread_null_mean,
      color = intervention_type,
      group = paste(intervention_type, network_id)
      ),
  data = st_1 %>% filter(intervention_size %in% c(0, 25))
) +
  scale_x_log10(breaks = c(.1, .5, 1, 2, 10), limits = c(.09, 10.2)) +
  scale_color_manual(values = intervention_colors) +
  stat_ecdf(alpha = .7, lwd = .3) +
  ylab("ECDF") +
  xlab("ratio of time to global spread") +
  theme(legend.position = c(0.8, 0.3)) +
  annotation_logticks(
    sides = "b",
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )# +
  #guides(colour = guide_legend(override.aes = list(alpha = .7)))
many_ecdf_plot

ggsave('figures/chami/time_to_spread_many_ecdfs.pdf',
       many_ecdf_plot, width = 5, height = 4)


facet_ecdf_plot = ggplot(
  aes(x = time_to_spread / time_to_spread_null_mean,
      color = intervention_type,
      group = paste(intervention_type, intervention_size, network_id)
      ),
  data = st_1 %>% filter(intervention_size %in% c(0, 25))
) +
  facet_wrap( ~ network_id) +
  scale_x_log10(breaks = c(.1, .5, 1, 2, 10), limits = c(.09, 10.2)) +
  scale_color_manual(values = intervention_colors) +
  stat_ecdf(alpha = .7, lwd = .3) +
  ylab("ECDF") +
  xlab("ratio of time to global spread") +
  theme(legend.position = c(0.8, 0.3)) +
  annotation_logticks(
    sides = "b",
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )
facet_ecdf_plot

###
# difference in ECDFs

time_seq = 1:max(st_1$time_to_spread)
st_1_ecdf <- st_1 %>%
  group_by(network_group, network_id, intervention_type) %>%
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
  inner_join(combined_summary_null) %>%
  group_by(network_group, network_id) %>%
  mutate(
    cdf_diff_none = cdf - cdf[intervention_type == "none"],
    cdf_diff_triad_addition = cdf - cdf[intervention_type == "triad_addition"],
    )


many_diff_in_ecdf_plot = ggplot(
  aes(x = time_to_spread,
      y = cdf_diff_triad_addition,
      color = intervention_type,
      group = paste(intervention_type, network_id)
      ),
  data = st_1_ecdf_diff %>%
    filter(intervention_type == "random_addition")
) +
  scale_color_manual(values = intervention_colors) +
  geom_hline(yintercept = 0, lwd = .3, lty = 2, alpha = .5) +
  geom_line(alpha = .4, lwd = .3) +
  ylab(expression(ECDF[random] - ECDF[triadic])) +
  xlab("time to spread") +
  theme(legend.position = "none") +
  guides(colour = guide_legend(override.aes = list(alpha = .7)))
many_diff_in_ecdf_plot

ggsave('figures/chami/time_to_spread_many_diff_in_ecdfs_random_vs_triadic.pdf',
       many_diff_in_ecdf_plot, width = 5, height = 4)

facet_diff_in_ecdf_plot = ggplot(
  aes(x = time_to_spread,
      y = cdf_diff_triad_addition,
      color = intervention_type,
      group = paste(intervention_type, network_id)
      ),
  data = st_1_ecdf_diff %>%
    filter(intervention_type == "random_addition")
) +
  facet_wrap( ~ network_id) +
  scale_color_manual(values = intervention_colors) +
  geom_hline(yintercept = 0, lwd = .3, lty = 2, alpha = .5) +
  geom_line(alpha = .9, lwd = .3) +
  ylab(expression(ECDF[random] - ECDF[triadic])) +
  xlab("time to spread") +
  theme(legend.position = "none") +
  guides(colour = guide_legend(override.aes = list(alpha = .7))) +
  geom_text(
    aes(label = summary_str,
        x = 0
        ),
    data = combined_summary_null,
    inherit.aes = FALSE,
    y = .05,
    hjust = 0
  )
facet_diff_in_ecdf_plot


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
  stat_ecdf() +
  facet_wrap(
    ~ reorder(network_id, avg_clustering_null),
    scales = "free_x"
    ) +
  geom_text(
    aes(label = summary_str,
        x = time_to_spread_null_max
        ),
    data = st_1_network_summary %>% filter(network_id %in% sample_network_ids),
    inherit.aes = FALSE,
    y = .05,
    hjust = 1
  )
sample_ecdf_plot

ggsave('figures/chami/time_to_spread_sample_networks_ecdfs.pdf',
       sample_ecdf_plot, width = 12, height = 10)


st_1_summary = st_1 %>%
  inner_join(
    nd %>% select(-intervention_size)
  ) %>%
  group_by(network_group, network_id, intervention_type,
           network_size, number_edges,
           density_null, avg_clustering
           ) %>%
  summarise(
    time_to_spread_mean = mean(time_to_spread),
    time_to_spread_median = median(time_to_spread),
    )

ggplot(
  aes(x = avg_clustering, y = time_to_spread_mean,
      group = network_id),
  data = st_1_summary %>%
    filter(intervention_type %in% c("none", "rewired"))
) +
  geom_line(alpha = .3) +
  geom_point(aes(shape = intervention_type, color = intervention_type)) +
  scale_color_manual(values = intervention_colors) +
  scale_shape_manual(values = intervention_shapes)

ggplot(
  aes(x = avg_clustering, y = time_to_spread_mean,
      group = network_id),
  data = st_1_summary %>%
    filter(intervention_type %in% c("triad_addition", "random_addition"))
) +
  geom_line(alpha = .3) +
  geom_point(aes(shape = intervention_type, color = intervention_type)) +
  scale_color_manual(values = intervention_colors) +
  scale_shape_manual(values = intervention_shapes)

ggplot(
  aes(x = avg_clustering, y = time_to_spread_mean,
      group = network_id),
  data = st_1_summary %>%
    filter(intervention_type %in% c("none", "rewired"))
) +
  facet_wrap(ntile(density_null, 4) ~ ntile(network_size, 4), labeller = label_both, nrow = 2) +
  geom_line(alpha = .3) +
  geom_point(aes(shape = intervention_type, color = intervention_type)) +
  scale_color_manual(values = intervention_colors) +
  scale_shape_manual(values = intervention_shapes)
