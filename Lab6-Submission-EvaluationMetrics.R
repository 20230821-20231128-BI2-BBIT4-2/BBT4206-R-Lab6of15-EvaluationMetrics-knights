.libPaths()

lapply(.libPaths(), list.files)

if (require("languageserver")) {
  require("languageserver")
} else {
  install.packages("languageserver", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}


# STEP 1. Install and Load the Required Packages ----
## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## pROC ----
if (require("pROC")) {
  require("pROC")
} else {
  install.packages("pROC", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# 1. Accuracy and Cohen's Kappa ----
## 1.a. Load the dataset ----
library(readr)
census <- read_csv("data/census.csv")
View(census)

## 1.b. Determine the Baseline Accuracy ----
census_freq <- census$income
cbind(frequency =
        table(census_freq),
      percentage = prop.table(table(census_freq)) * 100)

## 1.c. Split the dataset ----
train_index <- createDataPartition(census$income,
                                   p = 0.75,
                                   list = FALSE)
census_train <- census[train_index, ]
census_test <- census[-train_index, ]

## 1.d. Train the Model ----
# We apply the 5-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 5)

set.seed(7)
income_model_glm <-
  train(income ~ ., data = census_train, method = "glm",
        metric = "Accuracy", trControl = train_control)

## 1.e. Display the Model's Performance ----
print(income_model_glm)

predictions <- predict(income_model_glm, census_test[, 1:13])
confusion_matrix <-
  caret::confusionMatrix(predictions,as.factor(
                         census_test[, 1:14]$income))
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")

# 2. RMSE, R Squared, and MAE ----

## 2.a. Load the dataset ----
Student_Marks <- read_csv("data/Student_Marks.csv")
View(Student_Marks)
student_no_na <- na.omit(Student_Marks)

## 2.b. Split the dataset ----
set.seed(7)

train_index <- sample(1:dim(Student_Marks)[1], 10) # nolint: seq_linter.
Student_Marks_train <- Student_Marks[train_index, ]
Student_Marks_test <- Student_Marks[-train_index, ]

## 2.c. Train the Model ----
train_control <- trainControl(method = "boot", number = 1000)

Student_Marks_model_lm <-
  train(Marks ~ ., data = Student_Marks_train,
        na.action = na.omit, method = "lm", metric = "RMSE",
        trControl = train_control)

## 2.d. Display the Model's Performance ----
print(Student_Marks_model_lm)

predictions <- predict(Student_Marks_model_lm, Student_Marks_test[, 1:2])

print(predictions)

#### RMSE ----
rmse <- sqrt(mean((Student_Marks_test$Marks - predictions)^2))
print(paste("RMSE =", rmse))

#### SSR ----
ssr <- sum((Student_Marks_test$Marks - predictions)^2)
print(paste("SSR =", ssr))

#### SST ----

sst <- sum((Student_Marks_test$Marks - mean(Student_Marks_test$Marks))^2)
print(paste("SST =", sst))

#### R Squared ----
r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", r_squared))

#### MAE ----
absolute_errors <- abs(predictions - Student_Marks_test$Marks)
mae <- mean(absolute_errors)
print(paste("MAE =", mae))

# 3. Area Under ROC Curve ----
## 3.a. Load the dataset ----
library(readr)
Customer_Churn <- read_csv("data/Customer Churn.csv")
Customer_Churn$Churn <- ifelse(Customer_Churn$Churn == 0, "No", "Yes")

View(Customer_Churn)
## 3.b. Determine the Baseline Accuracy ----

Customer_Churn_freq <- Customer_Churn$Churn
cbind(frequency =
        table(Customer_Churn_freq),
      percentage = prop.table(table(Customer_Churn_freq)) * 100)

## 3.c. Split the dataset ----.
train_index <- createDataPartition(Customer_Churn$Churn,
                                   p = 0.8,
                                   list = FALSE)
Customer_Churn_train <- Customer_Churn[train_index, ]
Customer_Churns_test <- Customer_Churn[-train_index, ]

## 3.d. Train the Model ----
train_control <- trainControl(method = "cv", number = 10,
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)

set.seed(7)
churn_model_knn <-
  train(Churn ~ ., data = Customer_Churn_train, method = "knn",
        metric = "ROC", trControl = train_control)

## 3.e. Display the Model's Performance ----
print(churn_model_knn)

predictions <- predict(churn_model_knn, Customer_Churns_test[, 1:13])

print(predictions)
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         as.factor(Customer_Churns_test[, 1:14]$Churn))


print(confusion_matrix)

predictions <- predict(churn_model_knn, Customer_Churns_test[, 1:13],
                       type = "prob")


print(predictions)


roc_curve <- roc(Customer_Churns_test$Churn, predictions$No)

plot(roc_curve, main = "ROC Curve for KNN Model", print.auc = TRUE,
     print.auc.x = 0.6, print.auc.y = 0.6, col = "blue", lwd = 2.5)

# 4. Logarithmic Loss (LogLoss) ----
## 4.a. Load the dataset ----
library(readr)
Crop_recommendation <- read_csv("data/Crop_recommendation.csv")
View(Crop_recommendation)

## 4.b. Train the Model ----

train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3,
                              classProbs = TRUE,
                              summaryFunction = mnLogLoss)
set.seed(7)

crop_model_cart <- train(label ~ ., data = Crop_recommendation, method = "rpart",
                         metric = "logLoss", trControl = train_control)

## 4.c. Display the Model's Performance ----

print(crop_model_cart)

# References ----
##UCI MACHINE LEARNING. (2016). Adult Census Income. Predict whether income exceeds $50K/yr based on census data. Retrieved from: https://www.kaggle.com/datasets/uciml/adult-census-income
##Yasser H, M. (2021). Student Marks Dataset. Student Marks Prediction - Regression Problem. Retrieved from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset/
##Ingle, Atharva. (2020). Crop Recommendation Dataset. Maximize agricultural yield by recommending appropriate crops. Retrieved from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset/