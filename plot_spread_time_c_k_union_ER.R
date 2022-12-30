library(latex2exp)
library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(Hmisc)
library(plotrix)

cwd <- dirname(rstudioapi::getSourceEditorContext()$path)

ck_union_ER_vs_kspreading_data <- read.csv(
  paste(cwd,"/data/theory-simulations/ck_union_ER_vs_k/output/ck_union_ER_vs_kspreading_data_dump.csv",sep=""),
  stringsAsFactors = FALSE
)


ck_union_ER_vs_kspreading_data <- ck_union_ER_vs_kspreading_data %>%
  mutate(
    time_to_spread_lb = time_to_spread - std_in_spread,
    time_to_spread_ub = time_to_spread + std_in_spread
  )

# ploting

theme_set(theme_bw())
theme_update(
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank()
)

size_colors <- c(
  "250" = brewer.pal(8, "Set1")[1],
  "1000" = brewer.pal(8, "Set1")[2],
  "3000" = brewer.pal(8, "Set1")[3],
  "5000" = brewer.pal(8, "Set1")[4],
  "7000" = brewer.pal(8, "Set1")[5]
)

size_shapes <- c(
  "250" = 20,
  "1000" = 21,
  "3000" = 22,
  "5000" = 23,
  "7000" = 24
)

# Use 95% confidence interval instead of SEM
ck_union_ER_vs_k_plot_line_error_bar <- ggplot(
  aes(x = k, 
      y=time_to_spread,
      group=factor(network_size),
      color=factor(network_size),
      shape=factor(network_size),
      fill=factor(network_size)),
  data = ck_union_ER_vs_kspreading_data) +
  scale_color_manual(name = "network size",values = size_colors) + 
  scale_fill_manual(name = "network size",values = size_colors) +
  scale_shape_manual(name = "network size",values = size_shapes) +
  geom_errorbar(aes(ymin=time_to_spread_lb, ymax=time_to_spread_ub), 
                width=.1) +
  geom_line() +
  geom_point(size=2) +
  theme(
    legend.justification=c(1, 1),
    legend.position=c(0.4, 0.98),
    legend.key = element_rect(size = 2),
    legend.key.size = unit(1, 'lines')
  ) + xlab("order of cycle-power (k)") + ylab("mean time to spread")

ck_union_ER_vs_k_plot_line_error_bar

ggsave(paste(cwd,'/data/theory-simulations/ck_union_ER_vs_k/output/ck_union_ER_vs_k_plot_line_error_bar.pdf',sep=""),
       ck_union_ER_vs_k_plot_line_error_bar,
       width = 5, height = 4)

k_colors <- c(
  "5" = brewer.pal(8, "Set1")[3],
  "6" = brewer.pal(8, "Set1")[4],
  "7" = brewer.pal(8, "Set1")[5]
)

k_shapes <- c(
  "5" = 21,
  "6" = 22,
  "7" = 23
)

ck_union_ER_network_size_vs_kspreading_data<-ck_union_ER_vs_kspreading_data%>%
  filter(k >= 5)

ck_union_ER_vs_network_size_plot_line_error_bar <- ggplot(
  aes(x = network_size, 
      y=time_to_spread,
      group=factor(k),
      color=factor(k),
      shape=factor(k),
      fill=factor(k),
      ),
  data = ck_union_ER_network_size_vs_kspreading_data) +
  scale_color_manual(name = "cycle-power (k)",values = k_colors) + 
  scale_fill_manual(name = "cycle-power (k)",values = k_colors) +
  scale_shape_manual(name = "cycle-power (k)",values = k_shapes) +
  scale_x_continuous(breaks=c(250,1000,3000,5000,7000))+
  geom_errorbar(aes(ymin=time_to_spread_lb, ymax=time_to_spread_ub), 
                width=90) +
  geom_line() +
  geom_point(size=2) +
  theme(
    legend.justification=c(1, 1),
    legend.position=c(0.3, 0.98),
    legend.key = element_rect(size = 1),
    legend.key.size = unit(2, 'lines')
  ) + xlab("network size") + ylab("mean time to spread")

ck_union_ER_vs_network_size_plot_line_error_bar

ggsave(paste(cwd,'/data/theory-simulations/ck_union_ER_vs_k/output/ck_union_ER_vs_network_size_plot_line_error_bar.pdf',sep=""),
       ck_union_ER_vs_network_size_plot_line_error_bar,
       width = 5, height = 4)

