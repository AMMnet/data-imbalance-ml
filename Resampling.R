############RESAMPLING METHODS WITH R###############################
#######################AUTHOR: OLAWALE AWE, PhD.####################
####################################################################
####Install Packages and Libraries
list.of.packages <- c("psych", "ggplot2", "caretEnsemble", "tidyverse", "mlbench", "caret", "flextable", "mltools", "tictoc", "ROSE", "kernlab", "smotefamily", "klaR", "ada")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

### Load Some Necessary Packages and Libraries
library(psych)
library(ggplot2)
library(caretEnsemble)
library(tidyverse)
library(mlbench)
library(caret)
library(flextable)
library(mltools) 
library(tictoc)
library(ROSE)
library(smotefamily)

####################################################################
####################################################################
###Confirm and set your working directory
getwd()
#setwd()
####################################################################
# Load the health (malaria) data
#NOTE: please change the filepath according to where you've downloaded the data.
mal_data = read.csv("Data/Malaria-Data.csv", header = TRUE)
dim(mal_data)
mal_data
head(mal_data)
names(mal_data)
#str(odata)
attach(mal_data)
summary(mal_data) ###Descriptive Statistics
describe(mal_data)###Descriptive Statistics
sum(is.na(mal_data))###Check for missing data

###Rename the classes of the Target variable and plot it to determine imbalance
mal_data$overweight <- factor(mal_data$overweight, 
                           levels = c(0,1), 
                           labels = c('Normal', 'Overweight'))
###Plot Target Variable
plot(factor(overweight), names= c('Normal', 'Overweight'), col=c(2,3), ylim=c(0, 600), ylab='Respondent', xlab='BMI Category')
box()

###Assuming that all the EDA and feature selection has been performed.
################################################################
#Perform Featureplot to see the data distribution at a glance
featurePlot(x = odata[, -which(names(odata) == "overweight")],   # Predictors
            y = odata$overweight,                               # Target variable
            plot = "box",                                       # Type of plot (e.g., "box", "density", "scatter")
            strip = strip.custom(strip.names = TRUE),            # Add strip labels
            scales = list(x = list(relation = "free"),           # Scales for x-axis
                          y = list(relation = "free")))          # Scales for y-axis
#####Pairs Plot
#plot(odata)
#pairs(odata)
################################################################
######FEATURE SELECTION FOR CLASSIFICATION
### Boruta algorithm to determine the best variables
library(Boruta)
borC = Boruta(factor(overweight)~., data = odata, doTrace = 2, maxRuns=500)
print(borC)
par(pty='m')
plot(borC,las=2,cex.axis=0.7)
plotImpHistory(borC)
bor1=TentativeRoughFix(borC)
attStats(bor1)
##################################################################
#################################################################
###DATA PARTITION
##################################################################
ind=sample(2, nrow(odata),replace =TRUE, prob=c(0.70,0.30))
train=odata[ind==1,]
test= odata[ind==2,]
#Get the dimensions of your train and test data
dim(train)
dim(test)
###########################################################
###########################################################
# Model building with caret----------------------------------------------------------
###########################################################
###VIEW MODELS IN CARET
models= getModelInfo()
names(models)
##########################################################
# Prepare training scheme for cross-validation
control <- trainControl(method="repeatedcv", number=10, repeats=5)
#####TRAIN YOUR ML MODELS
# train an SVM model
set.seed(123)
tic()
SvmModel <- train(factor(overweight)~., data=train, method="svmRadial", trControl=control, na.action = na.omit)
toc()
SvmModel
Svmpred= predict(SvmModel,newdata = test)
SVM.cM<- confusionMatrix(Svmpred,as.factor(test$overweight), positive = 'Overweight', mode='everything')
SVM.cM
m1<- SVM.cM$byClass[c(1, 2, 5, 7, 11)]
m1
#plotting confusion matrix
SVM.cM$table
fourfoldplot(SVM.cM$table, col=rainbow(4), main="Imbalanced SVM Confusion Matrix")
plot(varImp(SvmModel, scale=T))

# Train an Random Forest model
set.seed(123)
tic()
RFModel <- train(factor(overweight)~., data=train, method="rf", trControl=control)
toc()
RFModel
RFpred=predict(RFModel,newdata = test)
RF.cM<- confusionMatrix(RFpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
m2<- RF.cM$byClass[c(1, 2, 5, 7, 11)]
m2
#plotting confusion matrix
RF.cM$table
fourfoldplot(RF.cM$table, col=rainbow(4), main="Imbalanced RF Confusion Matrix")
plot(varImp(RFModel, scale=T))

# Train an Logisitic Regression model
set.seed(123)
lrModel <- train(factor(overweight)~., data=train, method="glm", trControl=control)
lrModel
lrpred=predict(lrModel,newdata = test)
lr.cM<- confusionMatrix(lrpred,as.factor(test$overweight), positive = 'Overweight', mode='everything')
m3<- lr.cM$byClass[c(1, 2, 5, 7, 11)]
m3
#plotting confusion matrix
lr.cM$table
fourfoldplot(lr.cM$table, col=rainbow(4), main="Imbalanced LR Confusion Matrix")
plot(varImp(lrModel, scale=T))

##############################################################
####CREATE ROC curve
##################################################################
# Make predictions on the test set using type='prob'
predlr <- predict(lrModel, newdata = test, type = "prob")
# Load the ROCR package
library(ROCR)
# Create a prediction object needed by ROCR
pred_lr <- prediction(predlr[, "Overweight"], test$overweight)
# Calculate performance measures like ROC curve
perf_lr <- performance(pred_lr, "tpr", "fpr")
# Plot the ROC curve
plot(perf_lr, colorize = TRUE, main = "ROC Curve")
# Compute AUC
auc_value <- performance(pred_lr, "auc")@y.values[[1]]
auc_label <- paste("AUC =", round(auc_value, 2))
# Add AUC value as text on the plot
text(0.5, 0.3, auc_label, col = "blue", cex = 1.5)  # Adjust position and other text parameters as needed

# Train a k- Nearest Neigbour model
set.seed(123)
knnModel <- train(factor(overweight)~., data=train, method="knn", trControl=control)
knnModel
knnpred=predict(knnModel,newdata = test)
knn.cM<- confusionMatrix(knnpred,as.factor(test$overweight), positive = 'Overweight', mode='everything')
m4<- knn.cM$byClass[c(1, 2, 5, 7, 11)]
m4
#plotting confusion matrix
knn.cM$table
fourfoldplot(knn.cM$table, col=rainbow(4), main="Imbalanced KNN Confusion Matrix")
plot(varImp(knnModel, scale=T))

# Train a Neural Net model
set.seed(123)
nnModel <- train(factor(overweight)~., data=train, method="nnet", trControl=control)
nnModel
nnpred=predict(nnModel,newdata = test)
nn.cM<- confusionMatrix(nnpred,as.factor(test$overweight), positive = 'Overweight', mode='everything')
m5<- nn.cM$byClass[c(1, 2, 5, 7, 11)]
m5
#plotting confusion matrix
nn.cM$table
fourfoldplot(nn.cM$table, col=rainbow(4), main="Imbalanced NN Confusion Matrix")
plot(varImp(nnModel, scale=T))

# Train a Naive Bayes model
#Note; you may see warnings; This is not an error or indication that the code is 'wrong', it is just information to let you know that one of your observations is producing some unusual probabilities - something you may want to examine in either your data or modeling approach
set.seed(123)
nbModel <- train(factor(overweight)~., data=train, method="nb", trControl=control)
nbModel
nbpred=predict(nbModel,newdata = test)
nb.cM<- confusionMatrix(nbpred,as.factor(test$overweight), positive = 'Overweight', mode='everything')
m6<- nb.cM$byClass[c(1, 2, 5, 7, 11)]
m6
#plotting confusion matrix
nb.cM$table
fourfoldplot(nb.cM$table, col=rainbow(4), main="Imbalanced NB Confusion Matrix")
plot(varImp(nbModel, scale=T))

#Train a Linear Discriminant Analysis model
set.seed(123)
ldaModel <- train(overweight~., data=train, method="lda", trControl=control)
ldaModel
ldapred=predict(ldaModel,newdata = test)
lda.cM<- confusionMatrix(ldapred,as.factor(test$overweight), positive = 'Overweight', mode='everything')
m7<- lda.cM$byClass[c(1, 2, 5, 7, 11)]
m7
#plotting confusion matrix
lda.cM$table
fourfoldplot(lda.cM$table, col=rainbow(4), main="Imbalanced LDA Confusion Matrix")
plot(varImp(ldaModel, scale=T))

# Train a Bagging model
set.seed(123)
bagModel <- train(factor(overweight)~., data=train, method="treebag", trControl=control)
bagModel
bagpred=predict(bagModel,newdata = test)
bag.cM<- confusionMatrix(bagpred,as.factor(test$overweight), positive = 'Overweight', mode='everything')
m8<- bag.cM$byClass[c(1, 2, 5, 7, 11)]
m8
#plotting confusion matrix
bag.cM$table
fourfoldplot(bag.cM$table, col=rainbow(4), main="Imbalanced Bagging Confusion Matrix")
plot(varImp(bagModel, scale=T))

# Train a Boosting model
set.seed(123)
boModel <- train(overweight~., data=train, method="ada", trControl=control)
boModel
bopred=predict(boModel,newdata = test)
bo.cM<- confusionMatrix(bopred,as.factor(test$overweight), positive = 'Overweight', mode='everything')
m9<- bo.cM$byClass[c(1, 2, 5, 7, 11)]
m9
#plotting confusion matrix
bo.cM$table
fourfoldplot(bo.cM$table, col=rainbow(4), main="Imbalanced Boosting Confusion Matrix")
plot(varImp(boModel, scale=T))
############################### TABULATE #########################################
measure <-round(data.frame(SVM= m1, RF= m2, LR = m3, KNN=m4, NN=m5, NB=m6, Bagging = m8, Boosting = m9), 3)
dim(measure)
rownames(measure)=c('Sensitivity', 'Specificity', 'Precision','F1-Score', 'Balanced Accuracy')
flextable(measure)
#xtable(measure.score, digits = 3)

# collect all resamples and compare
results <- resamples(list(SVM=SvmModel, Bagging=bagModel,LR=lrModel,NB=nbModel,
                          RF=RFModel))
library(dplyr)

# # summarize the distributions of the results 
# summary(results)
# # boxplots of results
# bwplot(results)
# # dot plots of results
# dotplot(results)
##################################################################
##################################################################
# Handle Imbalanced: Oversampled data --------------------------------------------------------
##################################################################
over <- ovun.sample(factor(overweight)~., data = train, method = "over")$data
over
#xtable(table(over$class))

# Model building ----------------------------------------------------------
# prepare training scheme for cross-validation
control <- trainControl(method="repeatedcv", number=10, repeats=5)

# Train a SVM model
set.seed(123)
over.svmModel <- train(overweight~., data=over, method="svmRadial", trControl=control)
over.svmModel
over.svmpred=predict(over.svmModel,newdata = test)
over.SVM.cM<- confusionMatrix(over.svmpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
over.SVM.cM
over.m1<- over.SVM.cM$byClass[c(1, 2, 5, 7, 11)]
over.m1
#plotting confusion matrix
over.SVM.cM$table
fourfoldplot(over.SVM.cM$table, col=rainbow(4), main="Oversampled SVM Confusion Matrix")

# Train an Random Forest model
set.seed(123)
over.RFModel <- train(overweight~., data=over, method="rf", trControl=control)
over.RFModel
over.RFpred=predict(over.RFModel,newdata = test)
over.RF.cM<- confusionMatrix(over.RFpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
over.m2<- over.RF.cM$byClass[c(1, 2, 5, 7, 11)]
over.m2
#plotting confusion matrix
over.RF.cM$table
fourfoldplot(over.RF.cM$table, col=rainbow(4), main="Oversampled RF Confusion Matrix")

# Train a Logisitic Regression model
set.seed(123)
over.lrModel <- train(overweight~., data=over, method="glm", trControl=control)
over.lrModel
over.lrpred=predict(over.lrModel,newdata = test)
over.lr.cM<- confusionMatrix(over.lrpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
over.m3<- over.lr.cM$byClass[c(1, 2, 5, 7, 11)]
over.m3
#plotting confusion matrix
over.lr.cM$table
fourfoldplot(over.lr.cM$table, col=rainbow(4), main="Oversampled LR Confusion Matrix")

# Train an k- Nearest Neigbour model
set.seed(123)
over.knnModel <- train(overweight~., data=over, method="knn", trControl=control)
over.knnModel
over.knnpred=predict(over.knnModel,newdata = test)
over.knn.cM<- confusionMatrix(over.knnpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
over.m4<- over.knn.cM$byClass[c(1, 2, 5, 7, 11)]
over.m4
#plotting confusion matrix
over.knn.cM$table
fourfoldplot(over.knn.cM$table, col=rainbow(4), main="Oversampled KNN Confusion Matrix")

# Train a Neural Net model
set.seed(123)
over.nnModel <- train(overweight~., data=over, method="nnet", trControl=control)
over.nnModel
over.nnpred=predict(over.nnModel,newdata = test)
over.nn.cM<- confusionMatrix(over.nnpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
over.m5<- over.nn.cM$byClass[c(1, 2, 5, 7, 11)]
over.m5
#plotting confusion matrix
over.nn.cM$table
fourfoldplot(over.nn.cM$table, col=rainbow(4), main="Oversampled NN Confusion Matrix")

# Train a Naive Bayes model
set.seed(123)
over.nbModel <- train(overweight~., data=over, method="nb", trControl=control)
over.nbModel
over.nbpred=predict(over.nbModel,newdata = test)
over.nb.cM<- confusionMatrix(over.nbpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
over.m6<- over.nb.cM$byClass[c(1, 2, 5, 7, 11)]
over.m6
#plotting confusion matrix
over.nb.cM$table
fourfoldplot(over.nb.cM$table, col=rainbow(4), main="Oversampled NB Confusion Matrix")

## Train a Linear Discriminant Analysis model
set.seed(123)
ldaModel <- train(overweight~., data=train, method="lda", trControl=control)
ldaModel
ldapred=predict(ldaModel,newdata = test)
lda.cM<- confusionMatrix(ldapred,as.factor(test$overweight), positive = 'Normal', mode='everything')
m7<- lda.cM$byClass[c(1, 2, 5, 7, 11)]
m7
##plotting confusion matrix
lda.cM$table
fourfoldplot(lda.cM$table, col=rainbow(4), main="Imbalanced LDA Confusion Matrix")


# Train a Bagging model
set.seed(123)
over.bagModel <- train(overweight~., data=over, method="treebag", trControl=control)
over.bagModel
over.bagpred=predict(over.bagModel,newdata = test)
over.bag.cM<- confusionMatrix(over.bagpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
over.m8<- over.bag.cM$byClass[c(1, 2, 5, 7, 11)]
over.m8
#plotting confusion matrix
over.bag.cM$table
fourfoldplot(over.bag.cM$table, col=rainbow(4), main="Oversampled Bagging Confusion Matrix")


# Train a Boosting model
set.seed(123)
over.boModel <- train(overweight~., data=over, method="ada", trControl=control)
over.boModel
over.bopred=predict(over.boModel,newdata = test)
over.bo.cM<- confusionMatrix(over.bopred,as.factor(test$overweight), positive = 'Normal', mode='everything')
over.m9<- over.bo.cM$byClass[c(1, 2, 5, 7, 11)]
over.m9
#plotting confusion matrix
over.bo.cM$table
fourfoldplot(over.bo.cM$table, col=rainbow(4), main="Oversampled Boosting Confusion Matrix")

############################### measure #########################################
measure.score <-round(data.frame(SVM= over.m1, RF= over.m2, LR = over.m3, KNN=over.m4, NN=over.m5, NB=over.m6, Bagging = over.m8, Boosting = over.m9), 3)
#table(measure.score)

flextable(measure.score)
#xtable(measure.score, digits = 3)

# collect all resamples and compare
#results <- resamples(list(SVM=SvmModel, Bagging=bagModel,NB=modelnb,
                   #       RF=RfModel))

######################################################################################################################################################################
#######################################################################################################################################################################
########################################################################################################################################################################
# Undersampled data --------------------------------------------------------
###############################################
under <- ovun.sample(overweight~., data = train, method = "under")$data
under
names(under)
#under =under[,-c(13,14)]
#under
#xtable(table(over$class))

# Model building ----------------------------------------------------------
# prepare training scheme for cross-validation
control <- trainControl(method="repeatedcv", number=10, repeats=5)

# Train a SVM model
set.seed(123)
under.svmModel <- train(overweight~., data=under, method="svmRadial", trControl=control)
under.svmModel
under.svmpred=predict(under.svmModel,newdata = test)
under.SVM.cM<- confusionMatrix(under.svmpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
under.SVM.cM
under.m1<- under.SVM.cM$byClass[c(1, 2, 5, 7, 11)]
under.m1
#plotting confusion matrix
under.SVM.cM$table
fourfoldplot(under.SVM.cM$table, col=rainbow(4), main="Undersampled SVM Confusion Matrix")


#Train a Random Forest model
set.seed(123)
under.RFModel <- train(overweight~., data=under, method="rf", trControl=control)
under.RFModel
under.RFpred=predict(under.RFModel,newdata = test)
under.RF.cM<- confusionMatrix(under.RFpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
under.m2<- under.RF.cM$byClass[c(1, 2, 5, 7, 11)]
under.m2
#plotting confusion matrix
under.RF.cM$table
fourfoldplot(under.RF.cM$table, col=rainbow(4), main="Undersampled RF Confusion Matrix")

# Train a Logisitic Regression model
set.seed(123)
under.lrModel <- train(overweight~., data=under, method="glm", trControl=control)
under.lrModel
under.lrpred=predict(under.lrModel,newdata = test)
under.lr.cM<- confusionMatrix(under.lrpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
under.m3<- under.lr.cM$byClass[c(1, 2, 5, 7, 11)]
under.m3
#plotting confusion matrix
under.lr.cM$table
fourfoldplot(under.lr.cM$table, col=rainbow(4), main="Undersampled LR Confusion Matrix")

# Train a k- Nearest Neigbour model
set.seed(123)
under.knnModel <- train(overweight~., data=under, method="knn", trControl=control)
under.knnModel
under.knnpred=predict(under.knnModel,newdata = test)
under.knn.cM<- confusionMatrix(under.knnpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
under.m4<- under.knn.cM$byClass[c(1, 2, 5, 7, 11)]
under.m4
#plotting confusion matrix
under.knn.cM$table
fourfoldplot(under.knn.cM$table, col=rainbow(4), main="Undersampled KNN Confusion Matrix")

# Train a Neural Net model
set.seed(123)
under.nnModel <- train(overweight~., data=under, method="nnet", trControl=control)
under.nnModel
under.nnpred=predict(under.nnModel,newdata = test)
under.nn.cM<- confusionMatrix(under.nnpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
under.m5<- under.nn.cM$byClass[c(1, 2, 5, 7, 11)]
under.m5
#plotting confusion matrix
under.nn.cM$table
fourfoldplot(under.nn.cM$table, col=rainbow(4), main="Undersampled NN Confusion Matrix")

# Train a Naive Bayes model
set.seed(123)
under.nbModel <- train(overweight~., data=under, method="nb", trControl=control)
under.nbModel
under.nbpred=predict(under.nbModel,newdata = test)
under.nb.cM<- confusionMatrix(under.nbpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
under.m6<- under.nb.cM$byClass[c(1, 2, 5, 7, 11)]
under.m6
#plotting confusion matrix
under.nb.cM$table
fourfoldplot(under.nb.cM$table, col=rainbow(4), main="Undersampled NB Confusion Matrix")


## Train a Linear Discriminant Analysis model
set.seed(123)
ldaModel <- train(overweight~., data=train, method="lda", trControl=control)
ldaModel
ldapred=predict(ldaModel,newdata = test)
lda.cM<- confusionMatrix(ldapred,as.factor(test$overweight), positive = 'Normal', mode='everything')
m7<- lda.cM$byClass[c(1, 2, 5, 7, 11)]
m7
##plotting confusion matrix
lda.cM$table
fourfoldplot(lda.cM$table, col=rainbow(4), main="Imbalanced LDA Confusion Matrix")


# Train a Bagging model
set.seed(123)
under.bagModel <- train(overweight~., data=under, method="treebag", trControl=control)
under.bagModel
under.bagpred=predict(under.bagModel,newdata = test)
under.bag.cM<- confusionMatrix(under.bagpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
under.m8<- under.bag.cM$byClass[c(1, 2, 5, 7, 11)]
under.m8
#plotting confusion matrix
under.bag.cM$table
fourfoldplot(under.bag.cM$table, col=rainbow(4), main="Undersampled Bagging Confusion Matrix")


# Train a Boosting model
set.seed(123)
under.boModel <- train(overweight~., data=under, method="ada", trControl=control)
under.boModel
under.bopred=predict(under.boModel,newdata = test)
under.bo.cM<- confusionMatrix(under.bopred,as.factor(test$overweight), positive = 'Normal', mode='everything')
under.m9<- under.bo.cM$byClass[c(1, 2, 5, 7, 11)]
under.m9
#plotting confusion matrix
under.bo.cM$table
fourfoldplot(under.bo.cM$table, col=rainbow(4), main="Undersampled Boosting Confusion Matrix")

############################### TABULATE THE MEASURES #########################################
measure.score <-round(data.frame(SVM= under.m1, RF= under.m2, LR = under.m3, KNN=under.m4, NN=under.m5, NB=under.m6, Bagging = under.m8, Boosting = under.m9), 3)
table(measure.score)
flextable(measure.score)
#xtable(measure.score, digits = 3)'--Latex Table


# collect all resamples and compare
results <- resamples(list(SVM=SvmModel, Bagging=bagModel,LR=lrModel,NB=nbModel,
                          RF=RFModel))


###########################################################################################################################################################################
####################################################################################################################################################################################
########################################################################################################################################################################
# Hybrid Method --------------------------------------------------------
####################################################################
hybrid <- ovun.sample(overweight~., data = train, method = "both")$data
hybrid
#xtable(table(over$class))## for latex table

# Model building ----------------------------------------------------------
# prepare training scheme for cross-validation
control <- trainControl(method="repeatedcv", number=10, repeats=5)

# Train a SVM model
set.seed(123)
both.svmModel <- train(overweight~., data=hybrid, method="svmRadial", trControl=control)
both.svmModel
both.svmpred=predict(both.svmModel,newdata = test)
both.SVM.cM<- confusionMatrix(both.svmpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
both.SVM.cM
both.m1<- both.SVM.cM$byClass[c(1, 2, 5, 7, 11)]
both.m1
#plotting confusion matrix
both.SVM.cM$table
fourfoldplot(both.SVM.cM$table, col=rainbow(4), main="Hybrid SVM Confusion Matrix")

# Train a Random Forest model
set.seed(123)
both.RFModel <- train(overweight~., data=hybrid, method="rf", trControl=control)
both.RFModel
both.RFpred=predict(both.RFModel,newdata = test)
both.RF.cM<- confusionMatrix(both.RFpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
both.m2<- both.RF.cM$byClass[c(1, 2, 5, 7, 11)]
both.m2
#plotting confusion matrix
both.RF.cM$table
fourfoldplot(both.RF.cM$table, col=rainbow(4), main="Hybrid RF Confusion Matrix")

# Train a Logisitic Regression model
set.seed(123)
both.lrModel <- train(overweight~., data=hybrid, method="glm", trControl=control)
both.lrModel
both.lrpred=predict(both.lrModel,newdata = test)
both.lr.cM<- confusionMatrix(both.lrpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
both.m3<- both.lr.cM$byClass[c(1, 2, 5, 7, 11)]
both.m3
#plotting confusion matrix
both.lr.cM$table
fourfoldplot(both.lr.cM$table, col=rainbow(4), main="Hybrid LR Confusion Matrix")

# Train a k- Nearest Neigbour model
set.seed(123)
both.knnModel <- train(overweight~., data=hybrid, method="knn", trControl=control)
both.knnModel
both.knnpred=predict(both.knnModel,newdata = test)
both.knn.cM<- confusionMatrix(both.knnpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
both.m4<- both.knn.cM$byClass[c(1, 2, 5, 7, 11)]
both.m4
#plotting confusion matrix
both.knn.cM$table
fourfoldplot(both.knn.cM$table, col=rainbow(4), main="Hybrid KNN Confusion Matrix")

# Train a Neural Net model
set.seed(123)
both.nnModel <- train(overweight~., data=hybrid, method="nnet", trControl=control)
both.nnModel
both.nnpred=predict(both.nnModel,newdata = test)
both.nn.cM<- confusionMatrix(both.nnpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
both.m5<- both.nn.cM$byClass[c(1, 2, 5, 7, 11)]
both.m5
#plotting confusion matrix
both.nn.cM$table
fourfoldplot(both.nn.cM$table, col=rainbow(4), main="Hybrid NN Confusion Matrix")

# Train a Naive Bayes model
set.seed(123)
both.nbModel <- train(overweight~., data=hybrid, method="nb", trControl=control)
both.nbModel
both.nbpred=predict(both.nbModel,newdata = test)
both.nb.cM<- confusionMatrix(both.nbpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
both.m6<- both.nb.cM$byClass[c(1, 2, 5, 7, 11)]
both.m6
#plotting confusion matrix
both.nb.cM$table
fourfoldplot(both.nb.cM$table, col=rainbow(4), main="Hybrid NB Confusion Matrix")

# train a Linear Discriminant Analysis model
#set.seed(1)
#ldaModel <- train(overweight~., data=train, method="lda", trControl=control)
#ldaModel
#ldapred=predict(ldaModel,newdata = test)
#lda.cM<- confusionMatrix(ldapred,as.factor(test$overweight), positive = 'Normal', mode='everything')
#m7<- lda.cM$byClass[c(1, 2, 5, 7, 11)]
#m7
##plotting confusion matrix
#lda.cM$table
#fourfoldplot(lda.cM$table, col=rainbow(4), main="Imbalanced LDA Confusion Matrix")

# Train a Bagging model
set.seed(123)
both.bagModel <- train(overweight~., data=hybrid, method="treebag", trControl=control)
both.bagModel
both.bagpred=predict(both.bagModel,newdata = test)
both.bag.cM<- confusionMatrix(both.bagpred,as.factor(test$overweight), positive = 'Normal', mode='everything')
both.m8<- both.bag.cM$byClass[c(1, 2, 5, 7, 11)]
both.m8
#plotting confusion matrix
both.bag.cM$table
fourfoldplot(both.bag.cM$table, col=rainbow(4), main="Hybrid Bagging Confusion Matrix")

# Train a Boosting model
set.seed(123)
both.boModel <- train(overweight~., data=hybrid, method="ada", trControl=control)
both.boModel
both.bopred=predict(both.boModel,newdata = test)
both.bo.cM<- confusionMatrix(both.bopred,as.factor(test$overweight), positive = 'Normal', mode='everything')
both.m9<- both.bo.cM$byClass[c(1, 2, 5, 7, 11)]
both.m9
#plotting confusion matrix
both.bo.cM$table
fourfoldplot(both.bo.cM$table, col=rainbow(4), main="Hybrid Boosting Confusion Matrix")

############################### measure #########################################
measure.score <-round(data.frame(SVM= both.m1, RF= both.m2, LR = both.m3, KNN=both.m4, NN=both.m5, NB=both.m6, Bagging =both.m8, Boosting = both.m9), 3)
flextable(measure.score)

