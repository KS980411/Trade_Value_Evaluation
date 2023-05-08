rm(list= ls())
library(tictoc)
library(tidyverse)
library(gridExtra)
library(ggthemes)
library(RColorBrewer)
library(stringr)
library(lubridate)


Sys.getlocale()
Sys.setlocale("LC_ALL","C")

setwd('C:/Users/pain17/Desktop/project22_2')

dat1 <- read.csv('KBO 타자 연도, 포지션, WAR.csv', header = T, encoding = "UTF-8", na.strings = )
dat2 <- read.csv('KBO 투수 연도, WAR.csv', header = T, encoding = "UTF-8", na.strings = )
dat3 <- read.csv('KBO major level index.csv', header = T, encoding = "UTF-8", na.strings = )

Sys.setlocale("LC_ALL","Korean")

dat1 <- dat1 %>% spread(key = 'year' , value = 'WAR') %>% 
  rename (war_20 = '20' , war_21 = '21', war_22 = '22') %>% 
  group_by(name) %>% summarise(WAR_20 = sum(war_20, na.rm = T),
                               WAR_21 = sum(war_21, na.rm = T),
                               WAR_22 = sum(war_22, na.rm = T))

dat2 <- dat2 %>% spread(key = 'year' , value = 'WAR') %>% 
  rename (war_20 = '20' , war_21 = '21', war_22 = '22') %>% 
  group_by(name) %>% summarise(WAR_20 = sum(war_20, na.rm = T),
                               WAR_21 = sum(war_21, na.rm = T),
                               WAR_22 = sum(war_22, na.rm = T))

dat3 <- dat3 %>% rename (name = 'X.U.FEFF.name' , etc = 'X.U.BE44..U.ACE0.')

dat_final <- rbind(dat1 , dat2) %>% group_by(name) %>% summarize(war_20 = sum(WAR_20, na.rm = T),
                                                            war_21 = sum(WAR_21, na.rm = T),
                                                            war_22 = sum(WAR_22, na.rm = T))
dat_final <- left_join(dat3, dat_final , by = 'name')

setwd('C:/Users/pain17/Desktop')

write.csv(dat_final, 'dat_final_KBO_major.csv', row.names = T)


