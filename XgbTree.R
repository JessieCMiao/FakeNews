
library(dplyr)
library(DataExplorer)
library(caret)
library(readr)




news <-  vroom::vroom("./CleanFakeNews.csv")
names(news)
head(news)
tail(news)


news.train <- news %>% filter(Set == "train") %>% select(-Set)
news.test <- news %>% filter(Set == "test") %>% select(-Set)

news.train$isFake <- as.factor(news.train$isFake)
str(news.train$isFake)
dim(news.test)
names(news.train)

#10 folds repeat 3 times
control <- trainControl(method='repeatedcv', 
                        number=3, 
                        repeats=2)
grid_default <- expand.grid(
  nrounds = 250,
  max_depth = 10,
  eta = 0.3,
  gamma = 15,
  colsample_bytree = .5,
  min_child_weight = 25,
  subsample = 1)

#Metric compare model is Accuracy
metric <- "Accuracy"
set.seed(123)
xgb_default <- train(isFake ~ ., 
                     data=news.train, 
                     method='xgbTree', 
                     trControl=control,
                     tuneGrid = grid_default)

names(xgb_default)
#plot(xgb_default)
xgb_default$bestTune


predictions <- data.frame(id=news.test$Id, label = (predict(xgb_default, newdata=news.test)))
predictions

## write to a csv
write_csv(predictions, 'submission.csv') 
