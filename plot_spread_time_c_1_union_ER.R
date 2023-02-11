library(latex2exp)
library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(Hmisc)
library(plotrix)

cwd <- dirname(rstudioapi::getSourceEditorContext()$path)

c1_union_ER_vs_q_data <- read.csv(
  paste(cwd,"/data/theory-simulations/c1_union_ER/output/c1_union_ERspreading_data_dump.csv",sep=""),
  stringsAsFactors = FALSE
)


c1_union_ER_vs_q_data <- c1_union_ER_vs_q_data %>%
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
  "0.025" = brewer.pal(8, "Set1")[1],
  "0.05" = brewer.pal(8, "Set1")[2],
  "0.075" = brewer.pal(8, "Set1")[3]
)

size_shapes <- c(
  "0.025" = 20,
  "0.05" = 21,
  "0.075" = 22
)

# Use 95% confidence interval instead of SEM
c1_union_ER_vs_q_plot_line_error_bar <- ggplot(
  aes(x =network_size, 
      y=time_to_spread,
      group=factor(q),
      color=factor(q),
      shape=factor(q),
      fill=factor(q)),
  data = c1_union_ER_vs_q_data) +
  scale_color_manual(name = "q",values = size_colors) + 
  scale_fill_manual(name = "q",values = size_colors) +
  scale_shape_manual(name = "q",values = size_shapes) +
  geom_errorbar(aes(ymin=time_to_spread_lb, ymax=time_to_spread_ub), 
                width=.1) +
  geom_line() +
  geom_point(size=2) +
  theme(
    legend.justification=c(1, 1),
    legend.position=c(0.2, 0.98),
    legend.key = element_rect(size = 2),
    legend.key.size = unit(1, 'lines')
  ) + xlab("network size") + ylab("mean time to spread")

c1_union_ER_vs_q_plot_line_error_bar

ggsave(paste(cwd,"/data/theory-simulations/c1_union_ER/output/c1_union_ER_vs_q_plot_line_error_bar.pdf",sep=""),
       c1_union_ER_vs_q_plot_line_error_bar,
       width = 5, height = 4)

