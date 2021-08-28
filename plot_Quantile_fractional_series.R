library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(NLP)
library(stringr)
library(tidyverse)
library(ggplot2)
library(tidyselect)
library("prettyR")
library(latex2exp)
library(RColorBrewer)
library(Hmisc)
library(plotrix)
library("writexl")



#load

getwd()
setwd("/ihome/arahimian/cah259/contagion/data/cai-data/output/")
txt <- gsub("[()]", "", readLines("cai_edgelist_spreading_data_dump.csv"))
st <- read.csv(text=txt, header = T,sep = ",")


MODEL_1 = "(0.05,1)"


theme_set(theme_bw())
theme_update(
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank(),
  #axis.title.y=element_blank()
)

intervention_colors <- c(
  "none" = "black",
  "random addition" = brewer.pal(8, "Set1")[1],
  "triad addition" = brewer.pal(8, "Set1")[2],
  "rewired" = brewer.pal(8, "Set1")[5]
)
default_intervention_size = 10


quantil<-function(ch,q){
  a<-str_split(ch, ", ")
  a<-unlist(a)
  a<-as.numeric(a)
  return(sum(a<=q))
}

df<-st

df<-st %>% mutate(q0.9=mapply(quantil,fractional_evolution,0.9),
                  q0.75=mapply(quantil,fractional_evolution,0.75),
                  q0.5=mapply(quantil,fractional_evolution,0.5),
                  q0.25=mapply(quantil,fractional_evolution,0.25))



df<-df %>% group_by(network_id) %>% 
  filter(intervention_type=="none") %>% 
  summarise(ref0.9=mean(q0.9),
            ref0.75=mean(q0.75),
            ref0.5=mean(q0.5),
            ref0.25=mean(q0.25)) %>%
  left_join(df,.,by="network_id") %>% 
  mutate(normq0.9=q0.9/ref0.9,
         normq0.75=q0.75/ref0.75,
         normq0.5=q0.5/ref0.5,
         normq0.25=q0.25/ref0.25) %>%
  select(-c("ref0.9","ref0.75","ref0.5","ref0.25"))
  


ECDF_plot<-df %>%filter(theta_distribution=="[1, 0, 0, 0]")%>% 
  select(network_id,intervention_type, normq0.25) %>%
  ggplot(aes(normq0.25, 
                   col=intervention_type,
                   group=interaction(intervention_type, network_id)))+
  stat_ecdf(geom = "step",alpha = .8, lwd = .2)+
  scale_color_manual(values = intervention_colors)+
  scale_x_log10(breaks = c(0.1,.25, .5, 1, 2, 4, 10), limits = c(.1, 15)) +
  xlab("relative time to spread")+
  ylab("ECDF 25%")

ECDF_plot

ggsave("ECDF_plot_25.png",ECDF_plot)


#####plotting aggregates#########################


overall_ecdf_plot <- ggplot(
  aes(x = q0.9,
      color = intervention_type
  ),
  data = df %>% filter(theta_distribution=="[1, 0, 0, 0]")) +
  scale_color_manual(values = intervention_colors) +
  scale_x_log10(breaks = c(3,10,30,100), limits = c(2, 120)) +
  stat_ecdf(lwd = .5) +
  #facet_wrap( ~ factor(intervention_size)) +
  ylab("ECDF") +
  xlab("time to spread") +
  theme(legend.position = c(0.8, 0.3)) +
  annotation_logticks(
    sides = "b", size = .3,
    short = unit(0.05, "cm"), mid = unit(0.1, "cm"), long = unit(0.2, "cm")
  )
overall_ecdf_plot


  
#####################################################


all_data<-rep_n_stack(df,to.stack=c("q0.9","q0.75","q0.5","q0.25"),stack.names=c("percentage", "fractional_time"))

q <- qnorm(1 - .05/2)

intervention_shapes <- c(
  "none" = 21,
  "random addition" = 22,
  "triad addition" = 23,
  "rewired" = 24
)

all_data$network_id<-as.factor(all_data$network_id)
all_data$intervention_type<-as.factor(all_data$intervention_type)
all_data$percentage<-as.factor(all_data$percentage)

all_data <- all_data %>% select(-c(time_to_spread, fractional_evolution)) 

all_data<-all_data %>% 
  rename(time_to_spread= fractional_time)


levels(all_data$percentage) <- c("0.25","0.5","0.75","0.9")
levels(all_data$intervention_type) <- c("none","random addition","rewired","triad addition")
all_data<-all_data %>% 
  rename(intervention= intervention_type)


all_summaries_group_by_percentages <- all_data %>%
  filter(theta_distribution=="[1, 0, 0, 0]") %>% 
  group_by(percentage, network_id, intervention) %>%
  summarise(
    time_to_spread = mean(time_to_spread)
  ) %>%
  group_by(percentage, network_id) %>%
  mutate(
    time_to_spread_diff = time_to_spread - time_to_spread[intervention == "none"]
  ) %>%
  group_by(percentage, intervention) %>%
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







all_summaries_group_by_percentages

all_summaries_group_by_id_plot <- ggplot(
  aes(x = percentage, y=time_to_spread_mean, color=intervention),
  data = all_summaries_group_by_percentages, #%>%filter(group != "bakshy_role_no_feed"),
  xlab='',ylim = c(1,40))+ 
  ylab("time to spread")+
  xlab("Quantile")+
  scale_color_manual(values = intervention_colors) + 
  scale_fill_manual(values = intervention_colors) +
  #scale_shape_manual(values = intervention_shapes) +
  geom_pointrange(aes(ymin=time_to_spread_lb_diff, ymax=time_to_spread_ub_diff,shape=intervention),
                  position=position_dodge(width=0.75))+
  coord_cartesian(xlim = c(1,30)) + 
  theme(legend.justification=c(1,1), legend.position=c(0.3,0.95)) + 
  scale_y_log10(breaks = c(4,6,8,12,16,20,24)) + 
  coord_flip(ylim = c(3,25))

all_summaries_group_by_id_plot



################################################################






