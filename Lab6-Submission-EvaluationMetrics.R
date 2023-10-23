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
data(PimaIndiansDiabetes)
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

# [OPTIONAL] **Deinitialization: Create a snapshot of the R environment ----
# Lastly, as a follow-up to the initialization step, record the packages
# installed and their sources in the lockfile so that other team-members can
# use renv::restore() to re-install the same package version in their local
# machine during their initialization step.
# renv::snapshot() # nolint

# References ----

## Kuhn, M., Wing, J., Weston, S., Williams, A., Keefer, C., Engelhardt, A., Cooper, T., Mayer, Z., Kenkel, B., R Core Team, Benesty, M., Lescarbeau, R., Ziem, A., Scrucca, L., Tang, Y., Candan, C., & Hunt, T. (2023). caret: Classification and Regression Training (6.0-94) [Computer software]. https://cran.r-project.org/package=caret # nolint ----

## Leisch, F., & Dimitriadou, E. (2023). mlbench: Machine Learning Benchmark Problems (2.1-3.1) [Computer software]. https://cran.r-project.org/web/packages/mlbench/index.html # nolint ----

## National Institute of Diabetes and Digestive and Kidney Diseases. (1999). Pima Indians Diabetes Dataset [Dataset]. UCI Machine Learning Repository. https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database # nolint ----

## Robin, X., Turck, N., Hainard, A., Tiberti, N., Lisacek, F., Sanchez, J.-C., Müller, M., Siegert, S., Doering, M., & Billings, Z. (2023). pROC: Display and Analyze ROC Curves (1.18.4) [Computer software]. https://cran.r-project.org/web/packages/pROC/index.html # nolint ----

## Wickham, H., François, R., Henry, L., Müller, K., Vaughan, D., Software, P., & PBC. (2023). dplyr: A Grammar of Data Manipulation (1.1.3) [Computer software]. https://cran.r-project.org/package=dplyr # nolint ----

## Wickham, H., Chang, W., Henry, L., Pedersen, T. L., Takahashi, K., Wilke, C., Woo, K., Yutani, H., Dunnington, D., Posit, & PBC. (2023). ggplot2: Create Elegant Data Visualisations Using the Grammar of Graphics (3.4.3) [Computer software]. https://cran.r-project.org/package=ggplot2 # nolint ----

# **Required Lab Work Submission** ----
## Part A ----
# Create a new file called
# "Lab6-Submission-EvaluationMetrics.R".
# Provide all the code you have used to demonstrate the classification and
# regression evaluation metrics we have gone through in this lab.
# This should be done on any datasets of your choice except the ones used in
# this lab.

## Part B ----
# Upload *the link* to your
# "Lab6-Submission-EvaluationMetrics.R" hosted
# on Github (do not upload the .R file itself) through the submission link
# provided on eLearning.

## Part C ----
# Create a markdown file called "Lab-Submission-Markdown.Rmd"
# and place it inside the folder called "markdown". Use R Studio to ensure the
# .Rmd file is based on the "GitHub Document (Markdown)" template when it is
# being created.

# Refer to the following file in Lab 1 for an example of a .Rmd file based on
# the "GitHub Document (Markdown)" template:
#     https://github.com/course-files/BBT4206-R-Lab1of15-LoadingDatasets/blob/main/markdown/BIProject-Template.Rmd # nolint

# Include Line 1 to 14 of BIProject-Template.Rmd in your .Rmd file to make it
# displayable on GitHub when rendered into its .md version

# It should have code chunks that explain all the steps performed on the
# datasets.

## Part D ----
# Render the .Rmd (R markdown) file into its .md (markdown) version by using
# knitR in RStudio.

# You need to download and install "pandoc" to render the R markdown.
# Pandoc is a file converter that can be used to convert the following files:
#   https://pandoc.org/diagram.svgz?v=20230831075849

# Documentation:
#   https://pandoc.org/installing.html and
#   https://github.com/REditorSupport/vscode-R/wiki/R-Markdown

# By default, Rmd files are open as Markdown documents. To enable R Markdown
# features, you need to associate *.Rmd files with rmd language.
# Add an entry Item "*.Rmd" and Value "rmd" in the VS Code settings,
# "File Association" option.

# Documentation of knitR: https://www.rdocumentation.org/packages/knitr/

# Upload *the link* to "Lab-Submission-Markdown.md" (not .Rmd)
# markdown file hosted on Github (do not upload the .Rmd or .md markdown files)
# through the submission link provided on eLearning.
