
library("readr")
library('tidyr') 
library('dplyr')
library(ggplot2)
library(readr)
library(mlbench)
library(caret)
##My computer has limited RAM, so I have to narrow it down to only 100k rows
loan <- read.csv("C:/Users/ejy563/Downloads/loan.csv",nrows=100000)

#take a look
summary(loan)
glimpse(loan)
##calculate number of NAs for each column
get_na<-loan %>% summarize_all(funs(sum(is.na(.)) / length(.)))
##keep the ones that has percentage of NA< 20%
loan1<-loan[,which(get_na<0.2)]

##Dealing with NAs is always tricky. I do not want to go ahead just ignore NAs or impute with mean. Next, I am gonna
##check the distribution of each columns that has NAs against target to see if they are randomly missing or the reason
## that they are missing actually has something to do with the target

##check missing values for columns has NAs
col_missing<-names(loan[,which(get_na>0 & get_na<0.2)])
# [1] "dti"                   "revol_util"            "mths_since_rcnt_il"    "il_util"              
# [5] "all_util"              "avg_cur_bal"           "bc_open_to_buy"        "bc_util"              
# [9] "mo_sin_old_il_acct"    "mths_since_recent_bc"  "mths_since_recent_inq" "num_tl_120dpd_2m"     
# [13] "percent_bc_gt_75"
##Next, I wanna know if these missing values are randomly missing ot they are missing for some reason that is 
##somehow related to our target values.
##Set our target values first
bad_indicators <- c("Charged Off ",
                    "Charged Off",
                    "Default",
                    "Does not meet the credit policy. Status:Charged Off",
                    "In Grace Period",
                    "Default Receiver",
                    "Late (16-30 days)",
                    "Late (31-120 days)")
loan1$target <- ifelse(loan1$loan_status %in% bad_indicators,1,0)
##Now we wanna see how our columns distributed against targets. Take column dti as an example.
## get average default rate for missing dti.
##default_rate_dti1 is zero. In other word, None of the rows that has a missing dti has a target equal to 1
default_rate_dti1<-loan1%>%filter(is.na(dti))%>%summarise(sum(target)/n())
##get average default rate for non-missing dti.The rate is 0.00776,which is equal to mean(loan1$target). In other 
##words, all the loans that has target equal to 1 are the loans that do not have missing dti.
##On the first glance, missing values seem not being random. However, target values are very imbalanced with only 776
##rows has target value 1.This might also have something to do with me only select 100k rows.
default_rate_dti2<-loan1%>%filter(!is.na(dti))%>%summarise(sum(target)/n())

##Let's check with revol_util
## get average default rate for missing revol_util.
##default_rate_revol_util1=0.008264463
default_rate_revol_util1<-loan1%>%filter(is.na(revol_util))%>%summarise(sum(target)/n())
##get average default rate for non-missing revol_util.
##default_rate_revol_util2=0.007759389
default_rate_revol_util2<-loan1%>%filter(!is.na(revol_util))%>%summarise(sum(target)/n())

##let's check with mths_since_rcnt_il
##0.01421114
default_rate_mths_since_rcnt_il1<-loan1%>%filter(is.na(mths_since_rcnt_il))%>%summarise(sum(target)/n())
##0.007529621
default_rate_mths_since_rcnt_il2<-loan1%>%filter(!is.na(mths_since_rcnt_il))%>%summarise(sum(target)/n())

##let's check with il_util
##0.01047789
default_rate_il_util1<-loan1%>%filter(is.na(il_util))%>%summarise(sum(target)/n())
##0.007255655
default_rate_il_util2<-loan1%>%filter(!is.na(il_util))%>%summarise(sum(target)/n())

## Let's check the rest with a function
rate1<-as.data.frame(sapply(loan1[,c("all_util", "avg_cur_bal","bc_open_to_buy", "bc_util","mo_sin_old_il_acct","mths_since_recent_bc","mths_since_recent_inq", "num_tl_120dpd_2m",     
"percent_bc_gt_75")], function(x)loan1%>%filter(is.na(x))%>%summarise(sum(target)/n()) ))
rate2<-as.data.frame(sapply(loan1[,c("all_util", "avg_cur_bal","bc_open_to_buy", "bc_util","mo_sin_old_il_acct","mths_since_recent_bc","mths_since_recent_inq", "num_tl_120dpd_2m",     
                       "percent_bc_gt_75")], function(x)loan1%>%filter(!is.na(x))%>%summarise(sum(target)/n()) ))
## get rate1 and rate2 together
missing_value_default_date<-rbind(rate1,rate2)

##Conlusion: For some variables, there are some interesting findings. For instance, for column all_util, all the missing
##value rows does not have any target being 1. On the other hand, the dataset
##is extremely imbalanced.Such a desprepency might be something to dive further. But for this project purpose, I need to move on to other
##steps.

##Cheking duplicates. There is none.
sum(duplicated(loan1))

##Check variables that has single value.Ignore them if there is any.
loan2<-loan1[,which(sapply(loan1,function(x)length(unique(x)))>1)]

##However, there are columns,although has more than 1 values, has very imbalanced value1 vs value2 ratio. Such as
##columns containing hardship/settlement. So I am removing them too.
loan3<-loan2[,-which(grepl('hardship',names(loan2)))]
loan3<-loan3[,-which(grepl('settlement',names(loan3)))]

##get categorical columns and convert them into numeric
categorical_col<-sapply(loan3,is.factor)
loan_temp<-sapply(loan3[,categorical_col],unclass)   
loan4<-cbind(loan3[,!categorical_col],loan_temp) 

##Feature selection
##First, get the reduntant features that are highly correlated
set.seed(7)
# calculate correlation matrix
correlationMatrix <- cor(loan4)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=.5,verbose = FALSE,exact = FALSE)
# print indexes of highly correlated attributes
print(highlyCorrelated)