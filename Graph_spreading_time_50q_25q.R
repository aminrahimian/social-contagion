library(dplyr)
library(ggplot2)
library(RColorBrewer)
library(NLP)
library(stringr)
library(tidyverse)
library(ggplot2)



#load

#getwd()
setwd("/ihome/arahimian/cah259/contagion/data/cai-data/output/")
txt <- gsub("[()]", "", readLines("cai_edgelist_spreading_data_dump.csv"))
st <- read.csv(text=txt, header = T,sep = ",")



for (i in (1:dim(st)[1])){
  a<-st$fractional_evolution[i]
  a<-str_split(a, ", ")
  temp=c()
  for (j in 1:(st$time_to_spread[i]+1)){
    
    temp=c(temp, as.double(a[[1]][j]))
  }
  temp=list(temp)
  st$fractional_serie[i]=temp
}


for (i in 1:dim(st)[1]){
  n=length(st$fractional_serie[[i]])
  time=1
  while(st$fractional_serie[[i]][time]<=0.5){
    time=time+1
  }
  st$time_q50[i]=time-1
}


for (i in 1:dim(st)[1]){
  n=length(st$fractional_serie[[i]])
  time=1
  while(st$fractional_serie[[i]][time]<=0.25){
    time=time+1
  }
  
  st$time_q25[i]=time-1
}


###Create new column with ECDF_50

a=c(92,175)
type=c("none","rewired","random_addition","triad_addition")
final50=c()
for(i in a){
  
  ref<- st %>% filter(network_id==i, intervention_type=="none")%>%
    summarise(mean(time_q50))
  
  ref=as.double(ref)

  
  df<-c()
  for(j in type){
    temp<-st %>% filter(network_id==i, intervention_type==j)%>%
      mutate(norm_q50=time_q50/ref1)
    acum<-temp[with(temp, order(norm_q50)),]
    l=min(acum$norm_q50)
    u=max(acum$norm_q50)
    acum<-acum%>%mutate(ECDF_50=(norm_q50-l)/(u-l))

    df<-rbind(df,acum)
  }
  final50=rbind(final50,df)
}

###Create new column with ECDF_25

final25=c()
for(i in a){
  
  ref<- st %>% filter(network_id==i, intervention_type=="none")%>%
    summarise(mean(time_q25))
  
  ref=as.double(ref)
  df<-c()
  for(j in type){
    temp<-st %>% filter(network_id==i, intervention_type==j)%>%
      mutate(norm_q25=time_q25/ref)
    acum<-temp[with(temp, order(norm_q25)),]
    l=min(acum$norm_q25)
    u=max(acum$norm_q25)
    acum<-acum%>%mutate(ECDF_25=(norm_q25-l)/(u-l))
    
    df<-rbind(df,acum)
  }
  final25=rbind(final25,df)
}



#final$network_id<-as.factor(final$network_id)
#final$intervention_type<-as.factor(final$intervention_type)





base<- final50 %>% filter(network_id==1)

p<-ggplot(base, aes(norm_q50, ECDF_50, col=intervention_type)) + geom_step()
p

b=c(2:175)
for (i in b){
    print(i)
    temp<-final50 %>% filter(network_id==i)
    p<-p+geom_step(data=temp,aes(norm_q50, ECDF_50)) 
  
}
p



ggsave("q_50.pdf",p)

### for 25q

base<- final25 %>% filter(network_id==1)

q<-ggplot(base, aes(norm_q25, ECDF_25, col=intervention_type)) + geom_step()


b=c(2:175)
for (i in b){
  print(i)
  temp<-final25 %>% filter(network_id==i)
  q<-q+geom_step(data=temp,aes(norm_q25, ECDF_25)) 
  
}


ggsave("q_25.pdf",p)
