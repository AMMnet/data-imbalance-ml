##AMMNET Training on Handling Imbalanced Malaria Data for Machine Learning-Based Models####
############RESAMPLING METHODS WITH R###############################
#######################INSTRUCTOR:O.OLAWALE AWE, PhD.####################
################AMMnet Annual Meeting, Kigali, 2024#########################################
#############################################################################################

####Install Packages and Libraries############
#install.packages('caret', dependences=TRUE)
#install.packages('tidyverse', dependences=TRUE)
### Load Some Necessary Packages and Libraries
##############################################################
##Or Simply run the following codes to install all packages at once.
####Install Packages and Libraries
#list.of.packages <- c("psych", "ggplot2", "caretEnsemble", "tidyverse", "mlbench", "caret", "flextable", "mltools", "tictoc", "ROSE", "kernlab", "smotefamily", "klaR", "ada")
#new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
#if(length(new.packages)) install.packages(new.packages)


##Load libraries
library(caret) #for machine learning models
library(psych) ##for description of  data
library(ggplot2) ##for data visualization
library(caretEnsemble)##enables the creation of ensemble models
library(tidyverse) ##for data manipulation
library(mlbench)  ## for benchmarking ML Models
library(flextable) ## to create and style tables
library(mltools) #for hyperparameter tuning
library(tictoc) #for determining the time taken for a model to run
library(ROSE)  ## for random oversampling
library(smotefamily) ## for smote sampling
library(ROCR) ##For ROC curve

####Load the Health Data you want to work on 

# Load the Malaria data given
mdata = read.csv("Malaria-Data.csv", header = TRUE)
dim(mdata)
mdata
head(mdata)
names(mdata)
#str(odata)
attach(mdata)
summary(mdata) ###Descriptive Statistics
describe(mdata)###Descriptive Statistics
sum(is.na(mdata))###Check for missing data

###Note: For the purpose of this training, 
#it is assumed that the data is already clean and preprocessed 

###Rename the classes of the Target variable and plot it to determine imbalance
mdata$severe_maleria <- factor(mdata$severe_maleria, 
                           levels = c(0,1), 
                           labels = c('Not Infected', 'Infected'))
###Plot Target Variable
plot(factor(severe_maleria), names= c('Not Infected', 'Infected'), col=c(2,3), ylim=c(0, 600), ylab='Respondent', xlab='Malaria Diagnosis')
box()
#Or use ggplot 
#ggplot(mdata, aes(x = factor(severe_maleria))) + geom_bar() + labs(x = "Malaria Detected", y = "Count")

###DATA PARTITION FOR MACHINE LEARNING
##################################################################
ind=sample(2, nrow(mdata),replace =T, prob=c(0.70,0.30))
train=mdata[ind==1,]
test= mdata[ind==2,]
#Get the dimensions of your train and test data
dim(train)
dim(test)

##### Now Let's train some machine learning models using package caret
#The caret R package (Kuhn et al. 2021) (short for Classification And REgression Training) to carry out machine learning tasks in RStudio.
#The caret package offers a range of tools and models for classification and regression machine learning problems.
#In fact, it offers over 200 different machine learning models from which to choose. 
#Don’t worry, we don’t expect you to use them all!

###VIEW THE MODELS IN CARET
models= getModelInfo()
names(models)
################################################################
#Check for zero variance predictors:
nzv <- nearZeroVar(mdata[,-18], saveMetrics = TRUE)
print(nzv)
#Remove nzv
#mdata1 <- mdata[, !nzv$nzv]
#dim(mdata1)
#################################################################
####Prepare training scheme for cross-validation#################
#################################################################
control <- trainControl(method="repeatedcv", number=10, repeats=5)
##################################################################
#####TRAIN YOUR ML MODELS
# Train a SVM model
set.seed(123)
tic()
SvmModel <- train(factor(severe_maleria)~., data=train, method="svmRadial", trControl=control, na.action = na.omit)
toc()
SvmModel
Svmpred= predict(SvmModel,newdata = test)
SVM.cM<- confusionMatrix(Svmpred,as.factor(test$severe_maleria), positive = 'Infected', mode='everything')
SVM.cM
m1<- SVM.cM$byClass[c(1, 2, 5, 7, 11)]
m1
#plotting confusion matrix
SVM.cM$table
fourfoldplot(SVM.cM$table, col=rainbow(4), main="Imbalanced SVM Confusion Matrix")
plot(varImp(SvmModel, scale=T))
#####################################################################
# Train a Random Forest model
set.seed(123)
tic()
RFModel <- train(factor(severe_maleria)~., data=train, method="rf", trControl=control)
toc()
RFModel
RFpred=predict(RFModel,newdata = test)
RF.cM<- confusionMatrix(RFpred,as.factor(test$severe_maleria), positive = 'Infected', mode='everything')
m2<- RF.cM$byClass[c(1, 2, 5, 7, 11)]
m2
#plotting confusion matrix
RF.cM$table
fourfoldplot(RF.cM$table, col=rainbow(4), main="Imbalanced RF Confusion Matrix")
plot(varImp(RFModel, scale=T))
###################################################################
####CREATE ROC curve for your models-try for all models
# Make predictions on the test set using type='prob'
predrf <- predict(RFModel, newdata = test, type = "prob")
# Create a prediction object needed by ROCR
pred_rf <- prediction(predrf[, "Infected"], test$severe_maleria)
# Calculate performance measures like ROC curve
perf_rf <- performance(pred_rf, "tpr", "fpr")
# Plot the ROC curve
plot(perf_rf, colorize = TRUE, main = "ROC Curve-Random Forest")
# Compute AUC
auc_value <- performance(pred_rf, "auc")@y.values[[1]]
auc_label <- paste("AUC =", round(auc_value, 2))
# Add AUC value as text on the plot
text(0.5, 0.3, auc_label, col = "blue", cex = 1.5)  # Adjust position
#################################################################
# Train an Logisitic Regression model
set.seed(123)
lrModel <- train(factor(severe_maleria)~., data=train, method="glm", trControl=control)
lrModel
lrpred=predict(lrModel,newdata = test)
lr.cM<- confusionMatrix(lrpred,as.factor(test$severe_maleria), positive = 'Infected', mode='everything')
m3<- lr.cM$byClass[c(1, 2, 5, 7, 11)]
m3
#plotting confusion matrix
lr.cM$table
fourfoldplot(lr.cM$table, col=rainbow(4), main="Imbalanced LR Confusion Matrix")
plot(varImp(lrModel, scale=T))
##############################################################
##################################################################
# Make predictions on the test set using type='prob'
predlr <- predict(lrModel, newdata = test, type = "prob")
# Load the ROCR package
library(ROCR)
# Create a prediction object needed by ROCR
pred_lr <- prediction(predlr[, "Infected"], test$severe_maleria)
# Calculate performance measures like ROC curve
perf_lr <- performance(pred_lr, "tpr", "fpr")
# Plot the ROC curve
plot(perf_lr, colorize = TRUE, main = "ROC Curve-Logistic Regression")
# Compute AUC
auc_value <- performance(pred_lr, "auc")@y.values[[1]]
auc_label <- paste("AUC =", round(auc_value, 2))
# Add AUC value as text on the plot
text(0.5, 0.3, auc_label, col = "blue", cex = 1.5)  # Adjust position and other text parameters as needed
##############################################################
# Train a k- Nearest Neigbour model
set.seed(123)
knnModel <- train(factor(severe_maleria)~., data=train, method="knn", trControl=control)
knnModel
knnpred=predict(knnModel,newdata = test)
knn.cM<- confusionMatrix(knnpred,as.factor(test$severe_maleria), positive = 'Infected', mode='everything')
m4<- knn.cM$byClass[c(1, 2, 5, 7, 11)]
m4
#plotting confusion matrix
knn.cM$table
fourfoldplot(knn.cM$table, col=rainbow(4), main="Imbalanced KNN Confusion Matrix")
plot(varImp(knnModel, scale=T))
##############################################################
# Train a Neural Net model
set.seed(123)
nnModel <- train(factor(severe_maleria)~., data=train, method="nnet", trControl=control)
nnModel
nnpred=predict(nnModel,newdata = test)
nn.cM<- confusionMatrix(nnpred,as.factor(test$severe_maleria), positive = 'Infected', mode='everything')
m5<- nn.cM$byClass[c(1, 2, 5, 7, 11)]
m5
#plotting confusion matrix
nn.cM$table
fourfoldplot(nn.cM$table, col=rainbow(4), main="Imbalanced NN Confusion Matrix")
plot(varImp(nnModel, scale=T))
#############################################################
# Train a Naive Bayes model
set.seed(123)
nbModel <- train(factor(severe_maleria)~., data=train, method="nb", trControl=control)
nbModel
nbpred=predict(nbModel,newdata = test)
nb.cM<- confusionMatrix(nbpred,as.factor(test$severe_maleria), positive = 'Infected', mode='everything')
m6<- nb.cM$byClass[c(1, 2, 5, 7, 11)]
m6
#plotting confusion matrix
nb.cM$table
fourfoldplot(nb.cM$table, col=rainbow(4), main="Imbalanced NB Confusion Matrix")
plot(varImp(nbModel, scale=T))
####################################################################
#Train a Linear Discriminant Analysis model
set.seed(123)
ldaModel <- train(factor(severe_maleria)~., data=train, method="lda", trControl=control)
ldaModel
ldapred=predict(ldaModel,newdata = test)
lda.cM<- confusionMatrix(ldapred,as.factor(test$severe_maleria), positive = 'Infected', mode='everything')
m7<- lda.cM$byClass[c(1, 2, 5, 7, 11)]
m7
#plotting confusion matrix
lda.cM$table
fourfoldplot(lda.cM$table, col=rainbow(4), main="Imbalanced LDA Confusion Matrix")
plot(varImp(ldaModel, scale=T))
######################################################################
#Train a Linear Vector Quantization model
set.seed(123)
lvqModel <- train(factor(severe_maleria)~., data=train, method="lvq", trControl=control)
lvqModel
lvqpred=predict(lvqModel,newdata = test)
lvq.cM<- confusionMatrix(lvqpred,as.factor(test$severe_maleria), positive = 'Infected', mode='everything')
m7<- lvq.cM$byClass[c(1, 2, 5, 7, 11)]
m7
#plotting confusion matrix
lvq.cM$table
fourfoldplot(lvq.cM$table, col=rainbow(4), main="Imbalanced LDA Confusion Matrix")
plot(varImp(lvqModel, scale=T))

########################################################################
# Train a Bagging model
set.seed(123)
bagModel <- train(factor(severe_maleria)~., data=train, method="treebag", trControl=control)
bagModel
bagpred=predict(bagModel,newdata = test)
bag.cM<- confusionMatrix(bagpred,as.factor(test$severe_maleria), positive = 'Infected', mode='everything')
m8<- bag.cM$byClass[c(1, 2, 5, 7, 11)]
m8
#plotting confusion matrix
bag.cM$table
fourfoldplot(bag.cM$table, col=rainbow(4), main="Imbalanced Bagging Confusion Matrix")
plot(varImp(bagModel, scale=T))
################################################################
# Train a Boosting model
set.seed(123)
boModel <- train(factor(severe_maleria)~., data=train, method="ada", trControl=control)
boModel
bopred=predict(boModel,newdata = test)
bo.cM<- confusionMatrix(bopred,as.factor(test$severe_maleria), positive = 'Infected', mode='everything')
m9<- bo.cM$byClass[c(1, 2, 5, 7, 11)]
m9
#plotting confusion matrix
bo.cM$table
fourfoldplot(bo.cM$table, col=rainbow(4), main="Imbalanced Boosting Confusion Matrix")
plot(varImp(boModel, scale=T))
############################### TABULATE YOUR RESULTS#########################################
measure <-round(data.frame(SVM= m1, RF= m2, LR = m3, KNN=m4, NN=m5, NB=m6, Bagging = m8, Boosting = m9), 3)
dim(measure)
rownames(measure)=c('Sensitivity', 'Specificity', 'Precision','F1-Score', 'Balanced Accuracy')
flextable(measure)
measure
#xtable(measure.score, digits = 3)

# collect all resamples and compare
results <- resamples(list(SVM=SvmModel, Bagging=bagModel,LR=lrModel,NB=nbModel,
                          RF=RFModel))

# summarize the distributions of the results 
summary(results)
# # boxplots of results
bwplot(results)
# # dot plots of results
 dotplot(results)

### Oversampling 
 # Oversampled data --------------------------------------------------------
 over <- ovun.sample(factor(severe_maleria)~., data = train, method = "over")$data
 over
 #xtable(table(over$class))
 
 # Model building ----------------------------------------------------------
 # prepare training scheme for cross-validation
 control <- trainControl(method="repeatedcv", number=10, repeats=5)
 
 # train an SVM model
 set.seed(123)
 over.svmModel <- train(factor(severe_maleria)~., data=over, method="svmRadial", trControl=control)
 over.svmModel
 over.svmpred=predict(over.svmModel,newdata = test)
 over.SVM.cM<- confusionMatrix(over.svmpred,as.factor(test$severe_maleria), positive = 'Infected', mode='everything')
 over.SVM.cM
 over.m1<- over.SVM.cM$byClass[c(1, 2, 5, 7, 11)]
 over.m1
 #plotting confusion matrix
 over.SVM.cM$table
 fourfoldplot(over.SVM.cM$table, col=rainbow(4), main="Oversampled SVM Confusion Matrix")
 
 
 