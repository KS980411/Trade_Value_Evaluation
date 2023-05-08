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

dat1 <- read.csv('df_tradevalues_final.csv', header = T, na.strings = )
dat2 <- read.csv('minor_prospect.csv', header = T, na.strings = )
dat3 <- read.csv('major_roster.csv', header = T, na.strings = )

Sys.setlocale("LC_ALL","Korean")

#dat1----

dat1 <- dat1 %>% filter(level == 'Minors')
name1 <- data.frame(do.call('rbind',strsplit(as.character(dat1$name),split = ', ')))
name2 <- data.frame(rename = paste(name1$X2,name1$X1,sep = ' '))
dat1_1 <- cbind(name2,dat1[,3:11])

#dat2----

dat2$Age <- as.integer(round(dat2$Age,0))
dat2_1 <- dat2 %>% select(Name, Pos, FV, ETA, Age ) %>% rename(rename = Name, pos = Pos, fv = FV , eta = ETA, age = Age)

#dat3----

dat3 <- dat3 %>% rename(rename = Name) %>% mutate(in_major_40_roster = 'Yes')


#final_1----

dat_final <- left_join(dat1_1 , dat2_1, by = 'rename')
dat_final <- left_join(dat_final, dat3, by = 'rename')
dat_final_1 <- dat_final %>% transmute(NAME = rename,
                                    POS = ifelse(pos.y %in% c('SIRP','MIRP'), 'RP',
                                                 ifelse(pos.y %in% c('LF','CF','RF'),'OF',
                                                        ifelse(pos.y == 'SP/3B','SP',pos.y))),
                                    AGE = age.y,
                                    median.trade.value = median.trade.value,
                                    FV = fv,
                                    ETA = eta,
                                    MAJOR_40 = ifelse(in_major_40_roster == 'NA' , '0' , '1'))

dat_final_2 <- dat_final_1[c(order(dat_final_1$NAME)),]

setwd('C:/Users/pain17/Desktop')

write.csv(dat_final_2, ' dat_final.csv', row.names = T)







