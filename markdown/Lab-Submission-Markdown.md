Business Intelligence Project
================
Peter David Aringo
17/10/2023

- [Student Details](#student-details)

# Student Details

|                                                   |                                                              |
|---------------------------------------------------|--------------------------------------------------------------|
| **Student ID Numbers and Names of Group Members** | 135230 Peter Aringo                                          |
|                                                   | 135356 Ann Kigera                                            |
|                                                   | 122883 Michelle Guya                                         |
|                                                   | 134834 Kasio Emmanuel                                        |
|                                                   | 136301 Ian Nyameta                                           |
| **BBIT 4.2 Group**                                | Group B                                                      |
| **Course Code**                                   | BBT4206                                                      |
| **Course Name**                                   | Business Intelligence II                                     |
| **Program**                                       | Bachelor of Business Information Technology                  |
| **Semester Duration**                             | 21<sup>st</sup> August 2023 to 28<sup>th</sup> November 2023 |

## Installing and Loading the required packages

```{r Installing}
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

if (require("pROC")) {
  require("pROC")
} else {
  install.packages("pROC", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

```

# 1. Accuracy and Cohen's Kappa ----
This R code segment performs a series of steps for building and evaluating a predictive model for a census dataset. It begins by loading the data, calculating the baseline accuracy of the target variable ("income"), splitting the data into training and testing sets, training a generalized linear model (GLM) using 5-fold cross-validation, and displaying model performance metrics such as accuracy. It also generates and visualizes a confusion matrix to assess the model's predictive capabilities. The focus is on classification, and the code emphasizes accuracy and Cohen's Kappa as evaluation measures, making it a typical workflow for assessing the performance of a predictive model in a classification context.

```{r Accuracy and Cohen's Kappa}
# Your R code for Accuracy and Cohen's Kappa goes here
library(readr)
census <- read_csv("data/census.csv")
View(census)

census_freq <- census$income
cbind(frequency =
        table(census_freq),
      percentage = prop.table(table(census_freq)) * 100)

train_index <- createDataPartition(census$income,
                                   p = 0.75,
                                   list = FALSE)
census_train <- census[train_index, ]
census_test <- census[-train_index, ]

train_control <- trainControl(method = "cv", number = 5)

set.seed(7)
income_model_glm <-
  train(income ~ ., data = census_train, method = "glm",
        metric = "Accuracy", trControl = train_control)

print(income_model_glm)

predictions <- predict(income_model_glm, census_test[, 1:13])
confusion_matrix <-
  caret::confusionMatrix(predictions,as.factor(
                         census_test[, 1:14]$income))
print(confusion_matrix)

fourfoldplot(as.table(confusion_matrix), color = c("grey", "lightblue"),
             main = "Confusion Matrix")
```

# 2. RMSE, R Squared, and MAE ----
This R code segment focuses on the evaluation of a regression model's performance using key metrics. It starts by loading a dataset containing student marks, splits the data into training and testing sets, and trains a linear regression model using RMSE as the evaluation metric. The code then calculates and prints several important metrics, including RMSE (indicating the model's prediction accuracy), SSR (measuring unexplained variation), SST (representing total variation), R-squared (quantifying how well the model fits the data), and MAE (indicating average prediction error). These metrics collectively provide a thorough assessment of the model's accuracy, goodness of fit, and its ability to explain the variance in the data, making it a comprehensive evaluation of the regression model's performance.
```{r RMSE, R Squared, and MAE}
Student_Marks <- read_csv("data/Student_Marks.csv")
View(Student_Marks)
student_no_na <- na.omit(Student_Marks)

set.seed(7)

train_index <- sample(1:dim(Student_Marks)[1], 10) # nolint: seq_linter.
Student_Marks_train <- Student_Marks[train_index, ]
Student_Marks_test <- Student_Marks[-train_index, ]

train_control <- trainControl(method = "boot", number = 1000)

Student_Marks_model_lm <-
  train(Marks ~ ., data = Student_Marks_train,
        na.action = na.omit, method = "lm", metric = "RMSE",
        trControl = train_control)

print(Student_Marks_model_lm)

predictions <- predict(Student_Marks_model_lm, Student_Marks_test[, 1:2])

print(predictions)


rmse <- sqrt(mean((Student_Marks_test$Marks - predictions)^2))
print(paste("RMSE =", rmse))


ssr <- sum((Student_Marks_test$Marks - predictions)^2)
print(paste("SSR =", ssr))


sst <- sum((Student_Marks_test$Marks - mean(Student_Marks_test$Marks))^2)
print(paste("SST =", sst))


r_squared <- 1 - (ssr / sst)
print(paste("R Squared =", r_squared))


absolute_errors <- abs(predictions - Student_Marks_test$Marks)
mae <- mean(absolute_errors)
print(paste("MAE =", mae))
```

# 3. Area Under ROC Curve ----
This R code segment is dedicated to evaluating a classification model's performance using the Area Under the ROC Curve (AUC-ROC) metric. It starts by loading a dataset related to customer churn, converting the churn status to binary labels, and determining baseline accuracy. The dataset is split into training and testing sets, and a k-nearest neighbors (KNN) classification model is trained with a focus on ROC as the evaluation metric. The code displays the model's performance, including the ROC curve and AUC-ROC value, to assess the model's ability to differentiate between customers who churn and those who do not. It provides a comprehensive evaluation of the classification model's accuracy and its power to distinguish between churn and non-churn cases.
```{r Area Under ROC Curve}
library(readr)
Customer_Churn <- read_csv("data/Customer Churn.csv")
Customer_Churn$Churn <- ifelse(Customer_Churn$Churn == 0, "No", "Yes")

View(Customer_Churn)

Customer_Churn_freq <- Customer_Churn$Churn
cbind(frequency =
        table(Customer_Churn_freq),
      percentage = prop.table(table(Customer_Churn_freq)) * 100)

train_index <- createDataPartition(Customer_Churn$Churn,
                                   p = 0.8,
                                   list = FALSE)
Customer_Churn_train <- Customer_Churn[train_index, ]
Customer_Churns_test <- Customer_Churn[-train_index, ]

train_control <- trainControl(method = "cv", number = 10,
                              classProbs = TRUE,
                              summaryFunction = twoClassSummary)

set.seed(7)
churn_model_knn <-
  train(Churn ~ ., data = Customer_Churn_train, method = "knn",
        metric = "ROC", trControl = train_control)

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
```

# 4. Logarithmic Loss (LogLoss) ----
This R code segment is dedicated to evaluating a classification model using the Logarithmic Loss (LogLoss) metric. It begins by loading a dataset related to crop recommendations and proceeds to train a classification model using a decision tree method. The training employs repeated cross-validation with 5 folds and 3 repetitions while optimizing for LogLoss as the evaluation metric. The code then displays the model's performance, likely presenting LogLoss values to assess the model's accuracy in making probabilistic predictions for crop recommendations. LogLoss measures the quality of the model's predictions, with lower values indicating better performance, and this code segment provides an evaluation of the model's suitability for crop recommendation.
```{r Logarithmic Loss}
library(readr)
Crop_recommendation <- read_csv("data/Crop_recommendation.csv")
View(Crop_recommendation)


train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3,
                              classProbs = TRUE,
                              summaryFunction = mnLogLoss)
set.seed(7)

crop_model_cart <- train(label ~ ., data = Crop_recommendation, method = "rpart",
                         metric = "logLoss", trControl = train_control)

print(crop_model_cart)
```