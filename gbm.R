library(dplyr)
library(caret)

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

library(caret)
#use gbm
gbm <- train(form=isFake~.,
             data=news.train,
             method="gbm",
             trControl=trainControl(method="repeatedcv",
                                    number=3, #Number of pieces of your data
                                    repeats=1)) #repeats=1 = "cv"



gbm$results

gbm.preds <- data.frame(id=news.test$Id, label=predict(gbm, newdata=news.test))

write.csv(gbm.preds, "2submission.csv")
