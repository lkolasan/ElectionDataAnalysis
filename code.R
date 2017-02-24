##Using data from 2010 Congressional elections, we intend to build a classifier that would predict the election’s outcome. The data set includes information about the campaign funds, social media (Twitter, Facebook, and YouTube) campaigns, and demographics (age, gender) of 941 candidates who were in race in the general elections for The 112th House of Representatives seats in The U.S. Congress.##
##For descriptions of variables in the data, please refer to: http://www.fec.gov/finance/disclosure/metadata/DataDictionaryWEBALL.shtml#search=%22trans_from_auth%22##

ecd <- read.csv("election_campaign_data.csv", sep=",", header=T, strip.white = T, na.strings = c("NA","NaN","","?"))
str(ecd)
drop <- c("cand_id", "last_name", "first_name", "twitterbirth", "facebookdate", "facebookjan", "youtubebirth")
ecd2 <- ecd[,!(names(ecd) %in% drop)]
ecd2$twitter <- as.factor(ecd2$twitter)
ecd2$facebook <- as.factor(ecd2$facebook)
ecd2$youtube <- as.factor(ecd2$youtube)
ecd2$cand_ici <- as.factor(ecd2$cand_ici)
ecd2$gen_election <- as.factor(ecd2$gen_election)
ecddata <- ecd2[complete.cases(ecd2),]
n = nrow(ecddata)
trainIndex = sample(1:n, size = round(0.7*n), replace=FALSE)
train_data = ecddata[trainIndex,] 
test_data = ecddata[-trainIndex,] # We take the remaining 30% as the testing data
summary(train_data)

########Random Forest model##########
library(randomForest)
set.seed(32) 
rf <-randomForest(gen_election~., data=train_data, ntree=10, na.action=na.exclude, importance=T,proximity=T) 
print(rf)
set.seed(32) 
rf <-randomForest(gen_election~., data=train_data, ntree=20, na.action=na.exclude, importance=T,proximity=T) 
print(rf)
set.seed(32) 
rf <-randomForest(gen_election~., data=train_data, ntree=30, na.action=na.exclude, importance=T,proximity=T) 
print(rf)
set.seed(32) 
rf <-randomForest(gen_election~., data=train_data, ntree=40, na.action=na.exclude, importance=T,proximity=T) 
print(rf)
set.seed(32) 
rf <-randomForest(gen_election~., data=train_data, ntree=470, na.action=na.exclude, importance=T,proximity=T) 
print(rf)
mtry <- tuneRF(train_data[-26], train_data$gen_election, ntreeTry=470,  stepFactor=1.5, improve=0.01, trace=TRUE, plot=TRUE, , na.action=na.exclude)
set.seed(32)
rf <-randomForest(gen_election~., data=train_data, mtry=4, importance=TRUE, ntree=470)
print(rf)

library(caret)
predicted_values = predict(rf, type = "prob", test_data)
head(predicted_values)
threshold <- 0.5 
pred <- factor( ifelse(predicted_values[,2] > threshold, "W", "L") )
head(pred)
levels(test_data$gen_election)[2]
confusionMatrix(pred, test_data$gen_election, positive = levels(test_data$gen_election)[2])

library(ROCR)
library(ggplot2)
predicted_values <- predict(rf, test_data,type= "prob")[,2] 
pred <- prediction(predicted_values, test_data$gen_election)
perf <- performance(pred, measure = "tpr", x.measure = "fpr")
auc <- performance(pred, measure = "auc")
auc <- auc@y.values[[1]]

roc.data <- data.frame(fpr=unlist(perf@x.values),
                       tpr=unlist(perf@y.values),
                       model="RF")
ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
  geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr)) +
  ggtitle(paste0("ROC Curve w/ AUC=", auc))

varImpPlot(rf)

############### Neural Networks Model ###################

library(nnet)
ann <- nnet(gen_election ~ ., data=train_data, size=5, maxit=1000)
ann
predicted_values1 <- predict(ann, test_data,type= "raw")
head(predicted_values1)
threshold1 <- 0.5 

pred1 <- factor( ifelse(predicted_values1[,1] > threshold1, "W", "L") )
head(pred1)
levels(test_data$gen_election)[2]
confusionMatrix(pred1, test_data$gen_election, positive = levels(test_data$gen_election)[2])

library(ROCR)
library(ggplot2)
predicted_values1 <- predict(ann, test_data,type= "raw")
pred1 <- prediction(predicted_values1, test_data$gen_election)
perf <- performance(pred1, measure = "tpr", x.measure = "fpr")
auc <- performance(pred1, measure = "auc")
auc <- auc@y.values[[1]]
roc.data <- data.frame(fpr=unlist(perf@x.values),
                       tpr=unlist(perf@y.values),
                       model="ANN")
ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
  geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr)) +
  ggtitle(paste0("ROC Curve w/ AUC=", auc))

ann <- nnet(gen_election ~ ., data=train_data, size=24, maxit=1000)
ann
predicted_values1 <- predict(ann, test_data,type= "raw")
head(predicted_values1)
threshold1 <- 0.5 

pred1 <- factor( ifelse(predicted_values1[,1] > threshold1, "W", "L") )
head(pred1)
levels(test_data$gen_election)[2]
confusionMatrix(pred1, test_data$gen_election, positive = levels(test_data$gen_election)[2])

library(ROCR)
library(ggplot2)
predicted_values1 <- predict(ann, test_data,type= "raw")
pred1 <- prediction(predicted_values1, test_data$gen_election)
perf <- performance(pred1, measure = "tpr", x.measure = "fpr")
auc <- performance(pred1, measure = "auc")
auc <- auc@y.values[[1]]
roc.data <- data.frame(fpr=unlist(perf@x.values),
                       tpr=unlist(perf@y.values),
                       model="ANN")
ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
  geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr)) +
  ggtitle(paste0("ROC Curve w/ AUC=", auc))

############## Gradient Boosting Model #################

set.seed(32)
gbm_caret <- train(as.factor(gen_election) ~ ., 
                   data = train_data, method = "gbm", trControl = trainControl
                   (method = "repeatedcv", number = 4, repeats = 4),verbose = FALSE)
summary(gbm_caret)

predicted_values_gbm <- predict(gbm_caret, test_data,type= "prob")[,2] 
threshold <- 0.5
pred_gbm<-factor(ifelse(predicted_values_gbm > threshold, 'W','L'))
pred_gbm
test_data$gen_election
confusionMatrix(pred_gbm, test_data$gen_election, 
                positive = 'W')

library(ROCR)
library(ggplot2)
predicted_values_gbm <- predict(gbm_caret, test_data,type= "prob")[,2]
pred_gbm <- prediction(predicted_values_gbm, test_data$gen_election)

perf <- performance(pred_gbm, measure = "tpr", x.measure = "fpr")
auc <- performance(pred_gbm, measure = "auc")
auc <- auc@y.values[[1]]

roc.data <- data.frame(fpr=unlist(perf@x.values),
                       tpr=unlist(perf@y.values),
                       model="RF")

ggplot(roc.data, aes(x=fpr, ymin=0, ymax=tpr)) +
  geom_ribbon(alpha=0.2) +
  geom_line(aes(y=tpr)) +
  ggtitle(paste0("ROC Curve w/ AUC=", auc))

###### Code for identyfyng trends #######
ftable(xtabs(~twitter+facebook+youtube+gen_election, data=ecddata))
ftable(xtabs(~opp_fund+gen_election, data=ecddata))
ftable(xtabs(~coh_cop+gen_election, data=ecddata))
