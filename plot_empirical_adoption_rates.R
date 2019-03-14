
library(latex2exp)
library(dplyr)
library(ggplot2)
library(RColorBrewer)

##############################################centola dat

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

##########################################Bakshy_data

# load data
bakshy_role_data <- read.csv(
  "data/empirical_adoption_rates/bakshy_role_data_from_fig4a.csv",
  stringsAsFactors = FALSE
)

#bakshy_role_data_feed_0

bakshy_role_data_feed_0 <- bakshy_role_data %>% filter(feed == "0")

# drop the feed column

bakshy_role_data_feed_0 <- bakshy_role_data_feed_0[c("sharing_friends","prob_sharing")]

# repeat the first row before forming ratios

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

# repeat the first row before forming ratios

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

############################mosted_complex_data

# load data
mønsted_complex_data <- read.csv(
  "data/empirical_adoption_rates/mønsted_complex_fig3a.csv",
  stringsAsFactors = FALSE
)

mønsted_complex_data <- mønsted_complex_data %>% filter(type == "unique")

# drop the type column

mønsted_complex_data_type_unique <- mønsted_complex_data[c("num_adopters","prob")]

# repeat the first row before forming ratios

repeat_initial_row<-data.frame(mønsted_complex_data_type_unique$num_adopters[1],mønsted_complex_data_type_unique$prob[1])

names(repeat_initial_row)<-c("num_adopters","prob")

mønsted_complex_data_type_unique <- rbind(repeat_initial_row, mønsted_complex_data_type_unique)

# construct ratios

ratios <- exp(diff(log(mønsted_complex_data_type_unique$prob)))

# remove the first row

mønsted_complex_data_type_unique <- mønsted_complex_data_type_unique[-c(1), ]

# add k and ratio_k columns 

mønsted_complex_data_type_unique$k <- mønsted_complex_data_type_unique$num_adopters

mønsted_complex_data_type_unique$ratio_k<- ratios

mønsted_complex_data_type_unique_processed <- mønsted_complex_data_type_unique[c("k","ratio_k")]

mønsted_complex_data_type_unique_processed$group <- rep("Mønsted et. al. (2017)",length(mønsted_complex_data_type_unique_processed$k))

# remove the first row

mønsted_complex_data_type_unique_processed <- mønsted_complex_data_type_unique_processed[-c(1),]

# remove the last two rows

mønsted_complex_data_type_unique_processed <- head(mønsted_complex_data_type_unique_processed,-2)

############################ugander_structural_data

# load data
ugander_structural_data <- read.csv(
  "data/empirical_adoption_rates/ugander_structural_bonus_fig.csv",
  stringsAsFactors = FALSE
)

ugander_structural_data <- ugander_structural_data %>% filter(type == "recruitment")

# drop the type column

ugander_structural_data_type_recruitment <- ugander_structural_data[c("neighborhood_size","relative_conversion_rate")]

# repeat the first row before forming ratios

repeat_initial_row<-data.frame(ugander_structural_data_type_recruitment$neighborhood_size[1],ugander_structural_data_type_recruitment$relative_conversion_rate[1])

names(repeat_initial_row)<-c("neighborhood_size","relative_conversion_rate")

ugander_structural_data_type_recruitment <- rbind(repeat_initial_row, ugander_structural_data_type_recruitment)

# construct ratios

ratios <- exp(diff(log(ugander_structural_data_type_recruitment$relative_conversion_rate)))

# remove the first row

ugander_structural_data_type_recruitment <- ugander_structural_data_type_recruitment[-c(1),]

# add k and ratio_k columns 

ugander_structural_data_type_recruitment$k <- ugander_structural_data_type_recruitment$neighborhood_size

ugander_structural_data_type_recruitment$ratio_k<- ratios

ugander_structural_data_type_recruitment_processed <- ugander_structural_data_type_recruitment[c("k","ratio_k")]

ugander_structural_data_type_recruitment_processed$group <- rep("Ugander et. al. (2012)",length(ugander_structural_data_type_recruitment_processed$k))

# remove the first row

ugander_structural_data_type_recruitment_processed <- ugander_structural_data_type_recruitment_processed[-c(1),]

# remove the last four rows

ugander_structural_data_type_recruitment_processed <- head(ugander_structural_data_type_recruitment_processed,-4)

# ugander_structural_data_type_recruitment_processed <- ugander_structural_data_type_recruitment_processed[-nrow(ugander_structural_data_type_recruitment_processed),]

# combine all processed data frames

empirical_adoptions_rates = rbind(centola_data_processed, 
                                  bakshy_role_data_feed_0_processed, 
                                  bakshy_role_data_feed_1_processed,
                                  mønsted_complex_data_type_unique_processed,
                                  ugander_structural_data_type_recruitment_processed)

write.csv(empirical_adoptions_rates, file = "data/empirical_adoption_rates/empirical_adoptions_rates.csv")

#################################################### plotting

# plotting settings

theme_set(theme_bw())
theme_update(
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank()
)
group_colors <- c(
  "Centola (2010)" = "black",
  "bakshy_role_no_feed" = brewer.pal(8, "Set1")[1],
  "Bakshy et. al. (2012)" = brewer.pal(8, "Set1")[2],
  "Mønsted et. al. (2017)" = brewer.pal(8, "Set1")[3],
  "Ugander et. al. (2012)" = brewer.pal(8, "Set1")[4]
)
intervention_shapes <- c(
  "Centola (2010)" = 16,
  "bakshy_role_no_feed" = 16,
  "Bakshy et. al. (2012)" = 17,
  "Mønsted et. al. (2017)" = 18,
  "Ugander et. al. (2012)" = 19
)

# plot ECDF averaging over networks
empirical_adoption_rates_plot <- ggplot(
  aes(x = k, y=ratio_k,
      color = group
  ),
  data = empirical_adoptions_rates%>%filter(group != "bakshy_role_no_feed")
) +
  geom_line()+
  geom_point()+
  scale_color_manual(values = group_colors) +
  scale_color_discrete(name = "Dataset/study")+
  #scale_x_log10() +
  #stat_ecdf(lwd = .3) +
  #facet_wrap( ~ factor(intervention_size)) +
  #ylab(unname(TeX(c("$p(k)/p(k-1)$")))) +
  #xlab(unname(TeX(c("k")))) +
  ylab("p(k)/p(k-1)") +
  xlab("k") +
  theme(legend.position = c(0.7, 0.7))
  #annotation_logticks(
   # sides = "b", size = .3,
  #  short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  #)
empirical_adoption_rates_plot

ggsave('figures/empirical_adoption_rates/empirical_adoption_rates.pdf',
       empirical_adoption_rates_plot, width = 4.5, height = 3.5)

