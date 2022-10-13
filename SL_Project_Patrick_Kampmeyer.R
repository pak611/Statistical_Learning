

#****************************************************PROJECT SCRIPT *************************************************************
#*
#*This project asks you to develop a classification system for pieces of glass, such as glass that might be found at a crime scene. The final goal is the submission of a written report that 
# 1.	Clearly describes the goals and outcomes for the project.
# 2.	Clearly describes the data set used for the project (with appropriate references of course.)
# 3.	A careful description of your exploration of several different methods that could be used for classification including several tree based and several support vector machine based methods. 
# 4.	A brief summary comparing the results of those methods with a conclusion about their effectiveness.
# 5.	Provides the selection of a single method that you recommend for use for new data that provides the same predictive variables as the Glass Classification data set.

#*************************************************************************************************************************
#*


#install.packages('mlbench')

library(mlbench)

data(Glass)
dim(Glass)
levels(Glass$Type)
head(Glass)

  
  # Goal
  
#  Our goal is to identify the glass types by detecting or analyzing the materials or properties of the broken glass in the crime scene to help the police crack the criminal.

## Data Preprocessing



library(caret)
library(ISLR)
library(class)
library(MASS)
library(splines)



# load the data
glass = read.csv("glass.csv")
colnames(glass) = c("id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type")

### binary classification

#At first, we simplify our problem into binary classification problem, since our first goal is to detect whether the glass is float processed or not.


binary = function(x){
  if((x==1 | x==3)){
    return(1)
  }else{
    return(2)
  }
}
glass$Type = sapply(glass$Type, binary)
glass$Type = as.factor(glass$Type)


glass.type = glass$Type
glass.id = glass$id
glass = glass[,-1]
head(glass)




## Train Set Split

# we split the data
set.seed(1)
glass_idx = sample(nrow(glass), size = trunc(0.8 * nrow(glass)))
glass_trn = glass[glass_idx,]
glass_tst = glass[-glass_idx,]


X_train = glass_trn[,1:9]
y_train = glass_trn$Type
X_test = glass_tst[,1:9]
y_test = glass_tst$Type


### pairplot

par(mfrow = c(4,2))
library(ggplot2)
library(gridExtra)
p1 = qplot(Na,RI,data=glass,colour = glass.type)
p2 = qplot(Mg,RI,data=glass,colour = glass.type)
p3 = qplot(Al,RI,data=glass,colour = glass.type)
p4 = qplot(Si,RI,data=glass,colour = glass.type)
p5 = qplot(K,RI,data=glass,colour = glass.type)
p6 = qplot(Ca,RI,data=glass,colour = glass.type)
p7 = qplot(Ba,RI,data=glass,colour = glass.type)
p8 = qplot(Fe,RI,data=glass,colour = glass.type)
grid.arrange(p1, p2,p3,p4, nrow = 2, ncol=2)
grid.arrange(p5, p6,p7,p8, nrow = 2, ncol=2)




featurePlot(x = glass[,c("Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe","RI")], y = glass$Type)



### PCA


glass.pr = prcomp(glass[,c(2:9)], center = TRUE, scale = TRUE)
summary(glass.pr)
screeplot(glass.pr, type = "l", npcs = 8, main = "Screeplot of the PCs")
abline(h = 1, col="red", lty=5)
legend("topright", legend=c("Eigenvalue = 1"),
       col=c("red"), lty=5, cex=0.6)
cumpro <- cumsum(glass.pr$sdev^2 / sum(glass.pr$sdev^2))
plot(cumpro[0:15], xlab = "PC #", ylab = "Amount of explained variance", main = "Cumulative variance plot")
abline(v = 5, col="blue", lty=5)
abline(h = 0.88759, col="blue", lty=5)
legend("topleft", legend=c("Cut-off @ PC6"),
       col=c("blue"), lty=5, cex=0.6)



library("factoextra")
fviz_pca_ind(glass.pr, geom.ind = "point", pointshape = 21, 
             pointsize = 2, 
             fill.ind = as.factor(glass$Type), 
             col.ind = "black", 
             palette = "jco", 
             addEllipses = TRUE,
             label = "var",
             col.var = "black",
             repel = TRUE,
             legend.title = "Diagnosis") +
  ggtitle("2D PCA-plot from 9 feature dataset") +
  theme(plot.title = element_text(hjust = 0.5))


plot(glass.pr$x[,1],glass.pr$x[,2], col = c(2,3), xlab="PC1", ylab = "PC2 ", main = "PC1 / PC2 - plot")


### LDA


library(MASS)
glass.lda = lda(Type ~ ., data = glass_trn)
glass.lda.predict = predict(glass.lda, newdata = glass_tst)



### CONSTRUCTING ROC AUC PLOT:
# Get the posteriors as a dataframe.
library(ROCR)
glass.lda.predict.posteriors <- as.data.frame(glass.lda.predict$posterior)
# Evaluate the model
pred <- ROCR::prediction(glass.lda.predict.posteriors[,2], y_test)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values
# Plot
plot(roc.perf)
abline(a=0, b= 1)
text(x = .25, y = .65 ,paste("AUC = ", round(auc.train[[1]],3), sep = ""))




### LDA (pca)


glass.pcst = glass.pr$x[,1:4]
glass.pcst <- cbind(glass.pcst, as.numeric(glass.type)-1)
colnames(glass.pcst)[5] <- "type"





set.seed(1996)
num_obs = nrow(glass.pcst)
train_index = sample(num_obs, size = trunc(0.50 * num_obs))
train_data = data.frame(glass.pcst[train_index, ])
test_data = data.frame(glass.pcst[-train_index, ])




library(MASS)
glass.lda = lda(type ~ PC1+PC2+PC3+PC4, data = train_data)
glass.lda.predict = predict(glass.lda, newdata = test_data)




### CONSTRUCTING ROC AUC PLOT:
# Get the posteriors as a dataframe.
library(ROCR)
glass.lda.predict.posteriors <- as.data.frame(glass.lda.predict$posterior)
# Evaluate the model
pred <- ROCR::prediction(glass.lda.predict.posteriors[,2], test_data$type)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values
# Plot
plot(roc.perf)
abline(a=0, b= 1)
text(x = .25, y = .65 ,paste("AUC = ", round(auc.train[[1]],3), sep = ""))


#So, it is worse than the result from what we get from the original training set. So, we give it up.

#Now, let us see the performance of QDA.


glass.qda = qda(Type ~ ., data = glass_trn)
glass.qda.predict = predict(glass.qda, newdata = glass_tst)




### CONSTRUCTING ROC AUC PLOT:
# Get the posteriors as a dataframe.
library(ROCR)
glass.qda.predict.posteriors <- as.data.frame(glass.qda.predict$posterior)
# Evaluate the model
pred <- ROCR::prediction(glass.qda.predict.posteriors[,2], y_test)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values
# Plot
plot(roc.perf)
abline(a=0, b= 1)
text(x = .25, y = .65 ,paste("AUC = ", round(auc.train[[1]],3), sep = ""))




## Supervised Learning for binary classification

### Linear Regression 



{r,warning=FALSE}
lm.model = lm(Type~Mg,data = glass_trn)
sqrt(mean((lm.model$residuals)^2))






for(i in names(glass_trn[1:9])){
  fit = lm(glass_trn$Type ~ glass_trn[,i])
  cat(i,"-->", mean((fit$residuals)^2),"\n")
}
# Mg is chosen for the predictor for linear regression.




plot(Type ~ Mg, data = glass_trn, 
     col = "red", pch = "|", ylim = c(-0.2, 1),
     main = "Using Linear Regression for Classification")
abline(h = 0, lty = 3)
abline(h = 1, lty = 3)
abline(h = 0.5, lty = 2)
abline(lm.model, lwd = 3, col = "green")


#This model is not good since it provides some negative probabilities.


### Generalized Linear Model


{r,warning=FALSE}
# LOOCV
library(boot)
cv.error = rep(0,5)
for(i in 1:5){
  glm.fit = glm(Type~poly(Mg,i),data = glass_trn, family = "binomial")
  cv.error[i] = cv.glm(glass_trn,glm.fit)$delta[1]
  cat("polynomial: ", i, "--> cv error: ",cv.error[i], "\n")
}
polynomial = max(which.min(cv.error))
cat('the best ploynomial is ', polynomial)




{r,warning=FALSE}
glm.model = glm(Type ~ poly(Mg,polynomial), data = glass_trn, family = "binomial")
glm.prob = predict(glm.model, newdata = glass_tst, type = "response")
glm.pred = ifelse(glm.prob>0.5,2,1)


{r,warning=FALSE}
cal_class_err = function(actual, predicted){
  mean(actual!=predicted)
}
cal_class_err(actual = y_test, predicted = glm.pred)



{r}
CM_log = confusionMatrix(y_test, factor(glm.pred))
CM_log


{r}
metrics.log = CM_log$byClass
metrics.log



{r,warning=FALSE}
# k-fold
set.seed(1)
cv.error.10 = rep(0,10)
for(i in 1:10){
  glm.fit = glm(Type ~ poly(Mg,i),data =glass_trn,family = "binomial")
  cv.error.10[i] = cv.glm(glass_trn, glm.fit, K=10)$delta[1]
  cat("polynomial: ", i, "--> cv error: ",cv.error.10[i], "\n")
}
polynomial = max(which.min(cv.error.10))
cat('the best ploynomial is ', polynomial)

#So, we take the polynomial to be 7 so that we can reduce the varibility and have least chance to overfit. Let's see what will happen.

{r,warning=FALSE}
glm.model.10fold = glm(Type ~ poly(Mg,polynomial), data = glass_trn, family = "binomial")
glm.pred.10fold = ifelse(predict(glm.model.10fold, newdata = glass_tst, type = "response")>0.5,2,1)
cal_class_err = function(actual, predicted){
  mean(actual!=predicted)
}
cal_class_err(actual = glass_tst$Type, predicted = glm.pred.10fold)




plot(Type ~ Mg, data = glass_tst, 
     col = "red", pch = "|", ylim = c(-0.2, 1),
     main = "Using Logistic Regression for Classification")
abline(h = 0, lty = 3)
abline(h = 1, lty = 3)
abline(h = 0.5, lty = 2)
curve(predict(glm.model.10fold, data.frame(Mg = x), type = "response"), 
      add = TRUE, lwd = 3, col = "green", )
abline(v = -coef(glm.model.10fold)[1] / coef(glm.model.10fold)[2], lwd = 2)






### K-Nearest Neighbor Classifier




knn.pred = knn(train = scale(X_train), test = scale(X_test), cl = y_train, k = 3)
knn.pred



cal_class_err(actual = y_test, predicted = knn.pred)


#Then, I try to choose k.


set.seed(199)
k_to_try = 1:150
err_k = rep(x = 0, times = length(k_to_try))
for (i in seq_along(k_to_try)) {
  pred = knn(train = scale(X_train), 
             test  = scale(X_test), 
             cl    = y_train, 
             k     = k_to_try[i])
  err_k[i] = cal_class_err(actual = y_test, predicted = pred)
  if(i %% 10 == 0)
    cat('K:', i, "--> error: ", err_k[i], "\n")
}




# plot error vs choice of k
plot(err_k, type = "b", col = "dodgerblue", cex = 1, pch = 20, 
     xlab = "k, number of neighbors", ylab = "classification error",
     main = "(Test) Error Rate vs Neighbors")
# add line for min error seen
abline(h = min(err_k), col = "red", lty = 3)
# add line for minority prevalence in test set
abline(h = mean(y_test == 1), col = "orange", lty = 2)



which(min(err_k) == err_k)
k.best = max(which(min(err_k) == err_k))


In this case, we choose k = 4 since the largest one is the least variable, and has the least chance of overfitting.



{r, warning=FALSE}
knn.pred.best = knn(train = scale(X_train), test = scale(X_test), cl = y_train, k = k.best)



CM_knn = confusionMatrix(factor(y_test), factor(knn.pred.best))
CM_knn




metrics.knn = CM_knn$byClass
metrics.knn



### SVM


my_confusionmatrix = function(pred,truth,lvs = c(1,2,3,5,6,7)){
  lvs = lvs
  truth = factor(truth,levels = lvs)
  prediction = factor(pred,levels = lvs)
  CM = confusionMatrix(truth, prediction)
  return(CM)
}



library(e1071)
mysvm = function(kernel){
  svm.tune=tune(svm ,Type∼.,data=glass_trn ,kernel = kernel,
ranges=list(gamma = 2^(-8:1), cost = 2^(0:4)),
tunecontrol = tune.control(sampling = "fix"))
  
  best_gamma = svm.tune$best.parameters[1]
  best_cost = svm.tune$best.parameters[2]
  
  x.svm <- svm(Type~., data = glass_trn, cost=best_cost, gamma=best_gamma, kernel = kernel, probability = TRUE)
  x.svm.prob <- predict(x.svm, type="prob", newdata=glass_tst[-10], probability = TRUE)
  
  return(list(
    best_model = svm.tune$best.model,
    svm.prob = x.svm.prob
  ))
}


library(pROC)
SVM_pred = function(kernel){
  model = mysvm(kernel = kernel)$best_model
  ypred = predict(model,glass_tst[-10])
  
  CM_svm = my_confusionmatrix(ypred,glass_tst[,10], lvs = c(1,2))
  print(CM_svm)
  
  accuracy = (sum(diag(CM_svm$table)))/sum(CM_svm$table)
  
  predictions <- as.numeric(predict(model, glass_tst[-10], type = 'response'))
  
  roc.multi <- multiclass.roc(glass_tst[,10], predictions, quiet = TRUE)
  
  cat('kernel: ',kernel, '\n')
  cat('accuracy: ',accuracy, '\n')
  cat('AUC: ',auc(roc.multi), '\n')
  cat('\n')
  return(list(
    predictions = ypred,
    accuracy = accuracy
  ))
}


{r,warning=FALSE}
svm1=SVM_pred('linear')
svm2=SVM_pred('polynomial')
svm3=SVM_pred('radial')
svm4=SVM_pred('sigmoid')

#So, I choose the kernel to be "radial".




### Tree


library(party)
x.ct <- ctree(Type ~ ., data=glass_trn)
x.ct.pred <- predict(x.ct, newdata=glass_tst)
x.ct.prob <-  1- unlist(treeresponse(x.ct, glass_tst), use.names=F)[seq(1,nrow(glass_tst)*2,2)]
# To view the decision tree, uncomment this line.
plot(x.ct, main="Decision tree created using condition inference trees")


### Random Forest



x.cf <- cforest(Type ~ ., data=glass_trn, control = cforest_unbiased(mtry = ncol(glass)-2))
x.cf.pred <- predict(x.cf, newdata=glass_tst)
x.cf.prob <-  1- unlist(treeresponse(x.cf, glass_tst), use.names=F)[seq(1,nrow(glass_tst)*2,2)]



### Neural Network


library(nnet)
# creating training and test set
# fit neural network
set.seed(202)
scaler = function(x){
  return(
    (x - min(x)) / (max(x) - min(x))
  )
}
glass_trn[-10] = apply(glass_trn[-10],2,scaler)
glass_tst[-10] = apply(glass_tst[-10],2,scaler)
my_nnet = function(size){
  NN = nnet(Type~.,data = glass_trn, size = size,maxit = 200, decay = 5e-4)
  return(NN)
}




# nnet with 10 fold cv
glass[,1:9] = apply(glass[,1:9],2,scaler)
m = tune.nnet(Type~., data = glass, size = 1:15)
nn.cv = summary(m)




plot(nn.cv$performances[,1:2], type = "b")
abline(h = min(nn.cv$performances[,2]), v = nn.cv$performances[,1][which.min(nn.cv$performances[,2])], col=2)



nn.pred = predict(nn.cv$best.model, glass_tst[-10], type = "class")
table(nn.pred, glass_tst[,10])







NN.prediction = function(size){
  NN = my_nnet(size)
  pred = predict(NN, glass_tst[-10], type = "class")
  tab = table(pred,glass_tst[,10])
  print(tab)
  accuracy = sum(diag(tab))/sum(tab)
  return(accuracy)
}




size = seq(2,20,2)
res = c()
for(i in size){
  res = append(res, NN.prediction(i))
}



best_size = size[which.min(res)]
plot(res, type = "b", ylab = 'test error', xlab = 'size', main = "test error versus the size of hidden layers")
abline(v = which.min(res), h = min(res), col = 2)







### ROC
# ctree
x.ct.prob.rocr <- ROCR::prediction(x.ct.prob, y_test)
x.ct.perf <- performance(x.ct.prob.rocr, "tpr","fpr")
# add=TRUE draws on the existing chart 
plot(x.ct.perf, lty = 3, col=2, main="ROC curves of different machine learning classifier")
# Draw a legend.
legend(0.6, 0.6, c('Decison Tree', 'Random Forest','SVM','XGBoost','QDA', 'Logistic Regression', 'ANN'), 2:8)
# cforest
x.cf.prob.rocr <- ROCR::prediction(x.cf.prob, y_test)
x.cf.perf <- performance(x.cf.prob.rocr, "tpr","fpr")
plot(x.cf.perf, col=3, lty = 4,add=TRUE)
# svm
x.svm <- svm(Type~., data = glass_trn,kernel = "sigmoid", probability = TRUE)
x.svm.prob <- predict(x.svm, type="prob", newdata=glass_tst[-10], probability = TRUE) 
x.svm.prob.rocr <- ROCR::prediction(attr(x.svm.prob, "probabilities")[,2], y_test)
x.svm.perf <- performance(x.svm.prob.rocr, "tpr","fpr")
plot(x.svm.perf, col=4, lty = 5,add=TRUE)
# lda
glass.lda.predict.posteriors <- as.data.frame(glass.lda.predict$posterior)
# Evaluate the model
pred <- ROCR::prediction(glass.lda.predict.posteriors[,2], y_test)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values
round(auc.train[[1]],3)
# Plot
plot(roc.perf, col=5, lty = 6,add=TRUE)
# QDA
glass.qda.predict.posteriors <- as.data.frame(glass.qda.predict$posterior)
# Evaluate the model
pred <- ROCR::prediction(glass.qda.predict.posteriors[,2], y_test)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values
# Plot
plot(roc.perf, col = 6 , lty = 7,add = TRUE)
# Logistic regression
pred = ROCR::prediction(glm.prob, y_test)
roc.perf = performance(pred, measure = "tpr", x.measure = "fpr")
auc.train <- performance(pred, measure = "auc")
auc.train <- auc.train@y.values
plot(roc.perf, col = 7 , lty = 8,add = TRUE)
# nn.prob = predict(nn.cv$best.model, glass_tst[-10], type = "raw")
pred = ROCR::prediction(nn.prob, glass_tst[,10])
perf = performance(pred, "tpr", "fpr")
plot(perf, col = 8, lty = 9, add = TRUE)
abline(a=0,b=1)



### Spline 


# load the data
glass = read.csv("glass.csv")
colnames(glass) = c("id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type")


{r,warning=FALSE}
attach(glass)
set.seed(19)
knots = sample(min(Na):max(Na),3)
               
fit.spline = lm(RI~bs(Na,knots))
summary(fit.spline)


{r,warning=F}
Nalims = range(Na)
Na.grid = seq(Nalims[1],Nalims[2])
plot(Na,RI,col="grey",xlab='Na',ylab='RI')
points(Na.grid,predict(fit.spline,newdata = list(Na = Na.grid)),col="darkgreen",lwd=2,type="l")
#adding cutpoints
abline(v=knots,lty=2,col='dodgerblue')



{r,warning=F}
# smoothing spline
fit.spline.1 = smooth.spline(Na,RI,df = 16)
plot(Na,RI,col="grey",xlab="Na",ylab="RI")
points(Na.grid,predict(fit.spline,newdata=list(Na=Na.grid)),col="darkgreen",lwd=2,type="l")
# adding cut points
abline(v = knots,lty = 2,col = "dodgerblue")
lines(fit.spline.1,col="red",lwd=2)
legend("topright",c('Smoothing Spline with DOF=16','Cubic Spline'),col = c('red','darkgreen'),lwd = 2)



{r,warning=FALSE}
fit.spline.2 = smooth.spline(Na,RI,cv = TRUE)
fit.spline.2



{r,warning=F}
#It selects $\lambda=0.006579777 $ and df = 4.781314 as it is a Heuristic and can take various values for how rough the function is
plot(Na,RI,col="grey",xlab="Na",ylab="RI")
points(Na.grid,predict(fit.spline,newdata=list(Na=Na.grid)),col="darkgreen",lwd=2,type="l")
# adding cut points
abline(v = knots,lty = 2,col = "dodgerblue")
lines(fit.spline.1,col="red",lwd=2,lty=4)
lines(fit.spline.2,col="orange",lwd=2,lty=5)
legend("topright",c('Smoothing Spline with DOF=16','Cubic Spline','Smoothng Splines with DOF=4.78 selected by CV'),col = c('red','darkgreen','orange'),lwd = 2)


## Multiclass 

### LDA


glass = read.csv("glass.csv")
colnames(glass) = c("id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type")
glass = subset(glass, select = -c(id))



barplot(table(glass$Type), col = c(2,3,4,5,6,7))




library(corrplot)
M = cor(glass)
corrplot(M,method="circle")



barplot(M[,10][1:9],col = 2,main = "correlation with the glass type")



# we split the data
library(caTools)
set.seed(123)
glass$Type = as.factor(glass$Type)
split = sample.split(glass$Type, SplitRatio = 0.75)
glass_trn = subset(glass, split == TRUE)
glass_tst = subset(glass, split == FALSE)
table(glass_tst$Type)




X_train = glass_trn[,1:9]
y_train = glass_trn$Type
X_test = glass_tst[,1:9]
y_test = glass_tst$Type










library(MASS)
lda.model = lda(Type~.,data = glass_trn)
lda.model


The LDA output indicates that $\hat \pi_1$ = 0.327 and $\hat \pi_2$ = 0.355, which means 32.7% of the training observations correspond to glass type which is "1".

{r,warning=FALSE}
lda.pred = predict(lda.model,X_test)$class



library(caret)
library(ISLR)
library(class)
library(MASS)
library(splines)
library(pROC)




my_confusionmatrix = function(pred,truth,lvs = c(1,2,3,5,6,7)){
  lvs = lvs
  truth = factor(truth,levels = lvs)
  prediction = factor(pred,levels = lvs)
  CM = confusionMatrix(truth, prediction)
  return(CM)
}
my_confusionmatrix(pred = lda.pred, truth = y_test)




lda.cv = lda(Type~.,CV = TRUE,data = glass)
#table(glass$Type, lda.cv$class, dnn = c('Actual Group','Predicted Group'))
CM_lda_multi = my_confusionmatrix(pred = lda.cv$class, truth = glass$Type)
CM_lda_multi




# QDA


qda.model = qda(Type~Mg+Na+Al,data = glass_trn)
qda.model


The LDA output indicates that $\hat \pi_1$ = 0.327 and $\hat \pi_2$ = 0.355, which means 32.7% of the training observations correspond to glass type which is "1".

{r,warning=FALSE}
qda.pred = predict(qda.model,X_test)$class




my_confusionmatrix(pred = qda.pred, truth = y_test)




qda.cv = qda(Type~Na+Mg+Al,CV = TRUE,data = glass)
#table(glass$Type, lda.cv$class, dnn = c('Actual Group','Predicted Group'))
CM_lda_multi = my_confusionmatrix(pred = qda.cv$class, truth = glass$Type)
CM_lda_multi





###  SVM



# split into trn and tst
library(plspm)
library(e1071)


mysvm = function(kernel){
  # scale
  traindata = glass_trn
  traindata[-10] = scale(traindata[-10])
  testdata = glass_tst
  testdata[-10] = scale(testdata[-10])
  
  # tunning
  svm.tune=tune(svm ,Type∼.,data=traindata ,kernel = kernel,
ranges=list(gamma = 2^(-8:1), cost = 2^(0:4)),
tunecontrol = tune.control(sampling = "fix"))
  
  best_gamma = svm.tune$best.parameters[1]
  best_cost = svm.tune$best.parameters[2]
  
  x.svm <- svm(Type~., data = traindata, cost=best_cost, gamma=best_gamma, kernel = kernel, type = 'C-classification', probability = TRUE)
  x.svm.prob <- predict(x.svm, type="prob", newdata=testdata[-10], probability = TRUE)
  
  return(list(
    best_model = svm.tune$best.model,
    svm.prob = x.svm.prob
  ))
}







SVM_pred_multi = function(kernel){
  
  # scale
  traindata = glass_trn
  traindata[-10] = scale(traindata[-10])
  testdata = glass_tst
  testdata[-10] = scale(testdata[-10])
  
  model = mysvm(kernel = kernel)$best_model
  ypred = predict(model,testdata[-10])
  
  CM_svm = my_confusionmatrix(ypred,testdata$Type)
  
  accuracy = (sum(diag(CM_svm$table)))/nrow(testdata[-10])
  
  predictions <- as.numeric(predict(model, testdata[-10], type = 'response'))
  
  roc.multi <- multiclass.roc(testdata$Type, predictions, quiet = TRUE)
  
  cat('kernel: ',kernel, '\n')
  cat('accuracy: ',accuracy, '\n')
  cat('AUC: ',auc(roc.multi), '\n')
  cat('\n')
  return(list(
    predictions = ypred,
    accuracy = accuracy,
    table = CM_svm$table,
    byclass = CM_svm$byClass
  ))
}




{r,warning=FALSE}
svm1_multi = SVM_pred_multi('linear')
svm2_multi = SVM_pred_multi('polynomial')
svm3_multi = SVM_pred_multi('radial')
svm4_multi = SVM_pred_multi('sigmoid')


#We find that the accuracy is 87.5% and AUC is 97.16% when we set the kernel to be 'polynomial'.



folds = createFolds(glass_trn$Type, k = 10)
cv = lapply(folds, function(x) { 
  
  # scale
  traindata = glass_trn
  traindata[-10] = scale(traindata[-10])
  testdata = glass_tst
  testdata[-10] = scale(testdata[-10])
  
  training_fold = traindata[-x, ] 
  test_fold = traindata[x, ] 
  classifier = svm(formula = Type ~ .,
                   data = training_fold,
                   type = 'C-classification',
                   kernel = 'radial')
  
  y_pred = predict(classifier, newdata = test_fold[-10])
  cm = table(test_fold[, 10], y_pred)
  accuracy = sum(diag(cm))/sum(cm)
  return(accuracy)
})
accuracy = mean(as.numeric(cv))
accuracy







# Decison trees


library(tree)



tree.model = tree(Type∼., glass_trn)
summary(tree.model)




tree.pred = predict(tree.model, glass_tst ,type="class")
table(tree.pred, glass_tst$Type)
cat('accuracy: ', (sum(diag(table(tree.pred, glass_tst$Type))))/nrow(glass_tst))


# pruning the tree model
set.seed(102)
cv.tree.model = cv.tree(tree.model ,FUN=prune.misclass )
cv.tree.model


The tree with 10 terminal nodes results in the lowest
cross-validation error rate.



par(mfrow=c(1,2))
plot(cv.tree.model$size ,cv.tree.model$dev ,type="b", xlab = 'size', ylab = 'dev')
abline(v = 9, h = 50, lty = 5, col = 2)
plot(cv.tree.model$k ,cv.tree.model$dev ,type="b", xlab = 'k', ylab = 'dev')
abline(v = 1.5, h = 50, lty = 5, col = 2)
mtext("50 cross-validation errors", side = 3, line = -1, outer = TRUE)




prune.tree =prune.misclass (tree.model ,best=9)
plot(prune.tree)
text(prune.tree,pretty = 1)



tree.pred.prune=predict(prune.tree ,glass_tst , type="class")
table(tree.pred.prune,glass_tst$Type)
cat('accuracy: ', (sum(diag(table(tree.pred.prune, glass_tst$Type))))/nrow(glass_tst))


After parameter tuning, we improve the accuracy by 10%.

Let us check it out with random forest.


library(randomForest)


# bagging
set.seed(135)
bag.glass= randomForest(Type∼.,data = glass_trn,mtry = 9, importance =TRUE)
bag.glass

yhat.bag = predict(bag.glass,glass_tst, type = 'class')
table(yhat.bag,glass_tst$Type)
cat('accuracy: ', (sum(diag(table(yhat.bag, glass_tst$Type))))/nrow(glass_tst))

mybagging = function(mtry, table = FALSE){
  set.seed(135)
  bag = randomForest(Type∼.,data = glass_trn,mtry = mtry, importance =TRUE)
  
  yhat.bag = predict(bag,glass_tst, type = 'class')
  accuracy = (sum(diag(table(yhat.bag, glass_tst$Type))))/nrow(glass_tst)
  
  if(table == TRUE){
    print(table(yhat.bag, glass_tst$Type))
  }
  return(accuracy)
}

mtry = seq(3,9,0.05)
acc = sapply(mtry, mybagging)

plot(x = mtry, y = acc, type = 'b', lty = 1, xlab = "mtry", ylab = "accuracy")
abline(v = mtry[which.max(acc)], h = max(acc), col = 2, lty = 5)
best_mtry = mtry[which.max(acc)]
mybagging(best_mtry, table = TRUE)

Here, we use gbm package for __boosting__.

library(gbm)

set.seed(1)
boost.glass = gbm(Type∼.,data=glass_trn, 
                  n.trees=5000, 
                  #cv_fold = 5,
                  interaction.depth=4,
                  shrinkage = 0.005)
summary(boost.glass)


There are no significant influencial factors shown above.

yhat.boost=predict(boost.glass, glass_tst,
n.trees=5000)
yhat.boost = apply(yhat.boost, 1, which.max)
boost.pred = sapply(yhat.boost, function(x){
  return(ifelse(x>3, x+1, x))
})
confusionMatrix(factor(boost.pred,levels = c(1,2,3,5,6,7)), factor(y_test))


From the prediction result, the accruacy is 97.5% which is pretty good.

### ANN


# Helper packages
library(dplyr)         # for basic data wrangling
# Modeling packages
library(keras)         # for fitting DNNs
library(tfruns)        # for additional grid search & model training functions
# Modeling helper package - not necessary for reproducibility
library(tfestimators)  # provides grid search & model training interface



glass_x = glass[-10]
glass_y = glass[,10]
# standardize
glass_x = scale(glass_x)
# One-hot encode response
glass_y = to_categorical(as.numeric(as.character(glass_y)))
glass_y = glass_y[,-c(1,5)]
colnames(glass_y) = c("1","2","3","5","6","7")




model = keras_model_sequential() %>%
  
  # network 
  layer_dense(units = 4, activation = 'linear') %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 4, activation = 'linear') %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 4, activation = 'linear') %>%
  layer_dropout(0.5) %>%
  layer_dense(units = 6, activation = "softmax") %>%
  
  # Backpropagation
  compile(
    loss = 'categorical_crossentropy',
    optimizer = "adam",
    metrics = list('accuracy')
  )

fit1 <- model %>%
  fit(
    x = glass_x,
    y = glass_y,
    epochs = 25,
    batch_size = 4,
    validation_split = 0.4,
    verbose = FALSE
  )
fit1

plot(fit1)

library(nnet)
# creating training and test set
# fit neural network
set.seed(202)
scaler = function(x){
  return(
    (x - min(x)) / (max(x) - min(x))
  )
}
glass_trn[-10] = apply(glass_trn[-10],2,scaler)
glass_tst[-10] = apply(glass_tst[-10],2,scaler)
my_nnet = function(size){
  NN = nnet(glass_trn[,1:9],class.ind(glass_trn[,10]), size = size, softmax = TRUE,maxit = 500)
  return(NN)
}

# nnet with 10 fold cv
glass[,1:9] = apply(glass[,1:9],2,scaler)
m = tune.nnet(Type~., data = glass, size = 1:15)
nn.cv = summary(m)

plot(nn.cv$performances[,1:2], type = "b")
abline(h = min(nn.cv$performances[,2]), v = nn.cv$performances[,1][which.min(nn.cv$performances[,2])], col=2)

NN.prediction = function(size){
  NN = my_nnet(size)
  pred = predict(NN, glass_tst[-10], type = "class")
  tab = table(pred,glass_tst[,10])
  print(tab)
  accuracy = sum(diag(tab))/sum(tab)
  return(accuracy)
}

size = seq(2,12,2)
res = c()
for(i in size){
  res = append(res, NN.prediction(i))
}

best_size = size[which.min(res)]
plot(res, type = "b", ylab = 'test error', xlab = 'size', main = "test error versus the size of hidden layers")
abline(v = which.min(res), h = min(res), col = 2)
