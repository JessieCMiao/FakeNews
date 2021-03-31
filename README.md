# FakeNews

a. What is the overall purpose of this project?

The purpose of this notbook is to build a system to identify unreliable news articles so when we see an article, we can have a machine learning program to identify if the article might be fake news. 


b. What do each file in your repository do?

test.csv and train.csv are the dataset from the Kaggle competition (https://www.kaggle.com/c/fake-news/data). CleanFakeNews.csv is the file contain the data after Dr. Heaton did the cleaning. 
DataCleaning.R include what Professor Heaton did in class to help us start off this competition by cleaning the dataset. gbm.R is my first submission for this Kaggle competition which received a score of 0.9474. xgbtree.R is my second submission for this Kaggle competition which received a score of 0.9612. submission.csv is the prediction result from xgbtree.R file and 2submission.csv is the prediction result from gbm.R file. 

c. What methods did you use to clean the data or do feature engineering?

I followed Dr. Heaton by first create a language variable, use term frequency–inverse document frequency to calculate most common use words, and make those tf-idf become explanatory variables. 


d. What methods did you use to generate predictions?

For the gbm.R, I used gradient boosting for the prediction. I choose the response variable as isFake to determine if the article is fake or not, method is gbm and Trcontrol is the controls for the function and predict our test data from our training data results. For the xgbtree.R, I use XGBTree prediction method. With the same response variable but the tuning parameters were chosen by trying different numbers to try to obtain the lowest RMSE(The values used in grid_default gave the lowest RMSE score). And using XGBTree model to get the prediction.
