

library(dplyr)
library(ggplot2)
library(RColorBrewer)

# load data

# each dataset will have a column called k for the number of reinforcing signals
# and another column called ratio_k for the ratio of adoptions at k to adoptions at k-1
# any additional column is dropped
# each processed dataset will start with the (k=1,ratio_k=1) point

centola_data <- read.csv(
  "data/empirical_adoption_rates/centola_data_from_fig3.csv",
  stringsAsFactors = FALSE
)

centola_data$k<- centola_data$reinforcing_signals

centola_data$ratio_k<- centola_data$hazard_ratio

centola_data_processed <- centola_data[c("k","ratio_k")]


#initial_row<-data.frame(1.0,1.0)
#names(initial_row)<-c("k","ratio_k")

#centola_data_processed <- rbind(initial_row , centola_data_processed)

centola_data_processed$group <- rep("Centola (2010)",length(centola_data_processed$k))

# load data
bakshy_role_data <- read.csv(
  "data/empirical_adoption_rates/bakshy_role_data_from_fig4a.csv",
  stringsAsFactors = FALSE
)

#bakshy_role_data_feed_0

bakshy_role_data_feed_0 <- bakshy_role_data %>% filter(feed == "0")

# drop the feed column

bakshy_role_data_feed_0 <- bakshy_role_data_feed_0[c("sharing_friends","prob_sharing")]

# repeat the first before foring ratios

repeat_initial_row<-data.frame(bakshy_role_data_feed_0$sharing_friends[1],bakshy_role_data_feed_0$prob_sharing[1])

names(repeat_initial_row)<-c("sharing_friends","prob_sharing")

bakshy_role_data_feed_0 <- rbind(repeat_initial_row, bakshy_role_data_feed_0)

# construct ratios

ratios <- exp(diff(log(bakshy_role_data_feed_0$prob_sharing)))

# remove the first row

bakshy_role_data_feed_0 <- bakshy_role_data_feed_0[-c(1), ]

# add k and ratio_k columns 

bakshy_role_data_feed_0$k <- bakshy_role_data_feed_0$sharing_friends

bakshy_role_data_feed_0$ratio_k<- ratios

bakshy_role_data_feed_0_processed <- bakshy_role_data_feed_0[c("k","ratio_k")]

bakshy_role_data_feed_0_processed$group <- rep("bakshy_role_no_feed",length(bakshy_role_data_feed_0_processed$k))

# remove the first row

bakshy_role_data_feed_0_processed <- bakshy_role_data_feed_0_processed[-c(1),]

################bakshy_role_data_feed_1

bakshy_role_data_feed_1 <- bakshy_role_data %>% filter(feed == "1")

# drop the feed column

bakshy_role_data_feed_1 <- bakshy_role_data_feed_1[c("sharing_friends","prob_sharing")]

# repeat the first before foring ratios

repeat_initial_row<-data.frame(bakshy_role_data_feed_1$sharing_friends[1],bakshy_role_data_feed_1$prob_sharing[1])

names(repeat_initial_row)<-c("sharing_friends","prob_sharing")

bakshy_role_data_feed_1 <- rbind(repeat_initial_row, bakshy_role_data_feed_1)

# construct ratios

ratios <- exp(diff(log(bakshy_role_data_feed_1$prob_sharing)))

# remove the first row

bakshy_role_data_feed_1 <- bakshy_role_data_feed_1[-c(1), ]

# add k and ratio_k columns 

bakshy_role_data_feed_1$k <- bakshy_role_data_feed_1$sharing_friends

bakshy_role_data_feed_1$ratio_k<- ratios

bakshy_role_data_feed_1_processed <- 
  bakshy_role_data_feed_1[c("k","ratio_k")]

bakshy_role_data_feed_1_processed$group <- 
  rep("Bakshy et. al. (2012)",length(bakshy_role_data_feed_1_processed$k))


# remove the first row

bakshy_role_data_feed_1_processed <- bakshy_role_data_feed_1_processed[-c(1),]

# combine all processed data frames

empirical_adoptions_rates = rbind(centola_data_processed, 
                                  bakshy_role_data_feed_0_processed, 
                                  bakshy_role_data_feed_1_processed)

write.csv(empirical_adoptions_rates, file = "data/empirical_adoption_rates/empirical_adoptions_rates.csv")

# plotting settings

theme_set(theme_bw())
theme_update(
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank()
)
group_colors <- c(
  "Centola (2010)" = "black",
  "bakshy_role_no_feed" = brewer.pal(8, "Set1")[1],
  "Bakshy et. al. (2012)" = brewer.pal(8, "Set1")[2]
)
intervention_shapes <- c(
  "Centola (2010)" = 16,
  "bakshy_role_no_feed" = 16,
  "Bakshy et. al. (2012)" = 17
)

# plot ECDF averaging over networks
empirical_adoption_rates_plot <- ggplot(
  aes(x = k, y=ratio_k,
      color = group
  ),
  data = empirical_adoptions_rates%>%filter(group != "bakshy_role_no_feed")
) +
  geom_line()+
  scale_color_manual(values = group_colors) +
  #scale_x_log10() +
  #stat_ecdf(lwd = .3) +
  #facet_wrap( ~ factor(intervention_size)) +
  ylab("p(k)/p(k-1)") +
  xlab("k") +
  theme(legend.position = c(0.8, 0.3))
  #annotation_logticks(
   # sides = "b", size = .3,
  #  short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  #)
empirical_adoption_rates_plot

ggsave('figures/empirical_adoption_rates/empirical_adoption_rates.pdf',
       empirical_adoption_rates_plot, width = 4.5, height = 3.5)

