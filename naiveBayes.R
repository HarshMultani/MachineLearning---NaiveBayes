# Naive _ Bayes Algorithm
install.packages('caTools')
library('caTools')
install.packages('e1071')
library(e1071)

# Importing the dataset
dataset <- read.csv('Social_Network_Ads.csv')
dataset <- dataset[3:5]


# Encoding the target variable
dataset$Purchased = factor(dataset$Purchased, levels = c(0,1))


# Split the dataset into train and test sets
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Feature Scaling the data
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])


# Fitting the Model here
model = naiveBayes(x = training_set[-3], y = training_set$Purchased)


# Predicting the test set results
Y_pred = predict(model, newdata = test_set[-3])


# Making the confusion matrix
cm = table(test_set[ ,3], Y_pred)



