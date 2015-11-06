# Practical Machine Learning Course Project

#Overview

This project uses data from personal activity measuring devices to predict which of 5 ways an individual performed a particular exercise (barbell lift). Two machine learning algorithms were used to create a predictive model. The random forest model was found most accurate.

#Setting options for the analysis

Packages were loaded that were used throughout this project, and the **`options`** function was used to turn off scientific notation. Also, the working directory and the seed were set.


```r
library(downloader)
library(knitr)
library(rmarkdown)
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```r
options(scipen = 999)
setwd("~/GoogleDrive/R/MachineLearning")
set.seed(500)
```


#Downloading the data and exploratory analysis

The data files were downloaded to the working directory and then loaded into R.


```r
if (!file.exists("training.csv")) {
    fileurl = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
    download(fileurl, "training.csv", mode = "wb")
}

if (!file.exists("testing.csv")) {
    fileurl2 = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
    download(fileurl2, "testing.csv", mode = "wb")
}


training = read.csv("training.csv", na.strings = c("NA", 
    "#DIV/0!", ""))
testing = read.csv("testing.csv", na.strings = c("NA", 
    "#DIV/0!", ""))
```

A table was created to get an understanding of how many observations were in each **`classe`** of exercise type. Also, the **`str`** function was used to get an understanding of the structure of the variables in the datasets (the results of this function are hidden form this document because of its length).


```r
str(training)
```


```r
table(training$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

The first 7 rows, which were not needed for the analysis, were removed. Also, some columns have a high percentage of missing values. Rows with more than 50% **`NA`** values were removed from the training dataset.


```r
trainingID = training[, 8:length(colnames(training))]
trainingNA = trainingID[, !colSums(is.na(trainingID)) >= 
    (0.5 * nrow(trainingID))]
```

#Creating the models

A subset of the training dataset was created for cross validation. The training dataset was subset into 60% to create the model and 40% to test the prediction model.


```r
inTrain = createDataPartition(y = training$classe, 
    p = 0.6, list = FALSE)
trainingsubset = trainingNA[inTrain, ]
trainingtest = trainingNA[-inTrain, ]
trainingsubset = as.data.frame(trainingsubset)
```

Based on my understanding of types of machine learning regression models, a random forest alogrithm and a logit boosting algorithm were both used to create a predictive model with this data. Both models incorporated preprocessing using Principal Component Analysis, and each was cross-validated against the 40% testing subset (**`trainingtest`**) of the original training dataset.


```r
fitRF = randomForest(classe ~ ., data = trainingsubset, 
    method = "class", preProcOptions = "pca")
fitLB = train(classe ~ ., data = trainingsubset, method = "LogitBoost", 
    preProcess = "pca")
```

```
## Loading required package: caTools
## Loading required namespace: e1071
```

#Model predicitons and cross-validation


```r
predictRF = predict(fitRF, trainingtest, type = "class")
CMRF = confusionMatrix(predictRF, trainingtest$classe)
CMRF
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231    4    0    0    0
##          B    1 1511   13    0    0
##          C    0    3 1353   19    1
##          D    0    0    2 1267    3
##          E    0    0    0    0 1438
## 
## Overall Statistics
##                                                
##                Accuracy : 0.9941               
##                  95% CI : (0.9922, 0.9957)     
##     No Information Rate : 0.2845               
##     P-Value [Acc > NIR] : < 0.00000000000000022
##                                                
##                   Kappa : 0.9926               
##  Mcnemar's Test P-Value : NA                   
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9954   0.9890   0.9852   0.9972
## Specificity            0.9993   0.9978   0.9964   0.9992   1.0000
## Pos Pred Value         0.9982   0.9908   0.9833   0.9961   1.0000
## Neg Pred Value         0.9998   0.9989   0.9977   0.9971   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1926   0.1724   0.1615   0.1833
## Detection Prevalence   0.2849   0.1944   0.1754   0.1621   0.1833
## Balanced Accuracy      0.9994   0.9966   0.9927   0.9922   0.9986
```

The random forest method created a model with **99.413714** percent accuracy.


```r
predictLB = predict(fitLB, trainingtest)
CMLB = confusionMatrix(predictLB, trainingtest$classe)
CMLB
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1311  172  206   91   48
##          B   86  545   73  131  151
##          C   88  106  339   91   52
##          D   38   53   32  440   58
##          E   65  115   89   73  715
## 
## Overall Statistics
##                                                
##                Accuracy : 0.6482               
##                  95% CI : (0.635, 0.6612)      
##     No Information Rate : 0.3073               
##     P-Value [Acc > NIR] : < 0.00000000000000022
##                                                
##                   Kappa : 0.5468               
##  Mcnemar's Test P-Value : < 0.00000000000000022
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8256   0.5499   0.4587  0.53269   0.6982
## Specificity            0.8556   0.8944   0.9239  0.95831   0.9175
## Pos Pred Value         0.7172   0.5527   0.5015  0.70853   0.6764
## Neg Pred Value         0.9171   0.8934   0.9110  0.91511   0.9248
## Prevalence             0.3073   0.1918   0.1430  0.15983   0.1981
## Detection Rate         0.2537   0.1055   0.0656  0.08514   0.1384
## Detection Prevalence   0.3537   0.1908   0.1308  0.12016   0.2045
## Balanced Accuracy      0.8406   0.7222   0.6913  0.74550   0.8079
```

The logit boosting method created a model with **64.8219814** percent accuracy. 

#Out-of sample error

By subtracting this accuracy from the value of **`100`** , the out-of-sample percent error can be estimated. For the random forest method, the out-of-sample error can be estimated to be **0.586286** percent. The out-of-sample error for the logit boosting method can be estimated to be **35.1780186** percent. From this analysis, the random forest method is the most accurate. 

#Conclusions

Since the random forest model has **99.413714** percent accuracy, it can be used to accurately predict the type of class of barbell exercise an individual performed.


#Code for problem online submission


```r
testinganswers = predict(fitRF, testing, type = "class")

pml_write_files = function(x) {
    n = length(x)
    for (i in 1:n) {
        filename = paste0("problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, 
            row.names = FALSE, col.names = FALSE)
    }
}

pml_write_files(testinganswers)
```
