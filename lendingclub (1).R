library(MASS)
library("readr")
library('tidyr') 
library('dplyr')
library(ggplot2)
library(readr)
library(mlbench)
library(caret)
library(xgboost)
library(outliers)
##My computer has limited RAM, so I have to narrow it down to only 100k rows
loan <- read.csv("/Users/Amber/Downloads/loan.csv",nrows=100000)

#################################################QA attributes, fix missing values###################################
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
loan1<-na.omit(loan1)
##Check outliers
numeric_col<-sapply(loan1,is.numeric)
outliers<-sapply(loan1[,numeric_col],function(x)outlier(x))
loan1<-loan1[!loan1 %in% boxplot.stats(loan1[,numeric_col])$out]

##################################################Feature reduction##################################################
##Cheking duplicates. There is none.
sum(duplicated(loan1))

##Check variables that has single value.Ignore them if there is any.
loan2<-loan1[,which(sapply(loan1,function(x)length(unique(x)))>1)]

##However, there are columns,although has more than 1 values, has very imbalanced value1 vs value2 ratio. Such as
##columns containing hardship/settlement. So I am removing them too.
loan3<-loan2[,-which(grepl('hardship',names(loan2)))]
loan3<-loan2[,-which(grepl('settlement',names(loan2)))]
##################################################Convert into Numeric using one hot encoding########################
##get categorical columns and convert them into numeric using one hot encoding
categorical_col<-sapply(loan3,is.factor)
loan_temp<-as.data.frame(as.matrix(model.matrix(~. -1, data = loan3[,categorical_col])))
loan4<-cbind(loan3[,!categorical_col],loan_temp) 

##################################################Feature selection:2 methods########################################
##First, get the reduntant features that are highly correlated
set.seed(7)
# calculate correlation matrix
correlationMatrix <- cor(loan4)
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=.5,verbose = FALSE,exact = FALSE,names=FALSE)
# print indexes of highly correlated attributes
print(highlyCorrelated)
##remove the ones that are highly correlated
loan5 <- loan4[, -c(3, 15, 16, 18 ,29, 31 ,34, 35, 36, 37, 40, 41, 44 ,56, 57, 58, 59, 60, 61, 62, 63, 65, 66, 69, 70, 71, 72, 76, 82, 84, 90,1,2,10,17,5,19,28
                                  ,26 ,13, 25, 43, 49 , 9, 14,  8, 11,  4, 51, 22, 92, 93)]
##Build a base model
full.model <- lm(target ~., data = loan5)
# Stepwise regression model
step.model <- stepAIC(full.model, direction = "both", 
                      trace = FALSE)
summary(step.model)
##Combining domain knowledge and model results. The variables that I will be using are 
# annual_inc + dti+revol_bal + total_rec_int + 
# total_rec_late_fee + avg_cur_bal + chargeoff_within_12_mths + 
#  mo_sin_old_rev_tl_op + mths_since_recent_inq + num_tl_90g_dpd_24m + 
# percent_bc_gt_75 + term + grade  + verification_status + 
#  initial_list_status 
loan5<-loan5%>%select(annual_inc, dti,revol_bal, total_rec_int,
                        total_rec_late_fee,avg_cur_bal,chargeoff_within_12_mths, 
                        mo_sin_old_rev_tl_op,mths_since_recent_inq,num_tl_90g_dpd_24m, 
                        percent_bc_gt_75,term,grade,verification_status,
                        initial_list_status,target)
##################################################Building xgboost#################################################
##split into train/test
loan5_matrix <- data.matrix(loan5)
set.seed(123)
numberOfTrainingSamples <- round(length(loan5[,16]) * .7)

# training data
train_data <- loan5_matrix[1:numberOfTrainingSamples,]
train_labels <- loan5[,16][1:numberOfTrainingSamples]

# testing data
test_data <- loan5_matrix[-(1:numberOfTrainingSamples),]
test_labels <- loan5[,16][-(1:numberOfTrainingSamples)]
# put our testing & training data into two seperates Dmatrixs objects
dtrain <- xgb.DMatrix(data = train_data, label= train_labels)
dtest <- xgb.DMatrix(data = test_data, label= test_labels)
##xgboost
##train error:0.072
model <- xgboost(data = dtrain,
                 nround = 10,
                 objective = "reg:logistic") 

# generate predictions for our held-out testing data
pred <- predict(model, dtest)

# get & print the classification error.
err <- mean(as.numeric(pred > 0.5) != test_labels)
print(paste("test-error=", err))

##The test error was 0.0087.The dataset itself is very imbalanced,
##and I think that is why we are getting such a perfect error rate. If I were to have more time to work on a better 
##computer, we might get a different anwser. For the purpose of this project, I will just stop here.

