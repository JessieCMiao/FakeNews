install.packages('wactor')

# Load libraries
library(readr)
library(dplyr)
library(cld2)
library(stringr)
library(ggplot2)
library(tidyr)
library(tm)       ## For text mining
library(textstem)     ## For lemmatization
library(tidytext)
library(wordcloud2)
library(pROC)
library(ROCR)
library(caret)
library(DataExplorer)
library(tidyverse)
library(wactor)
library(xgboost)


# Read in data

news.train <- read.csv("./train.csv")
news.test <- read.csv("./test.csv")
news <- bind_rows(train=news.train, test=news.test, .id="Set")

# Fuse title with text
news$text <- with(news, paste(title, text))
# Analyze languages
news$lan <- cld2::detect_language(text = news$text, plain_text = FALSE)
# Group languages
for (i in 1:length(news$lan)) {
  if ((is.na(news$lan[i]) == TRUE) | (news$lan[i] %in% c('ar','ca','el','id','it',
                                                         'nl','no','pl','pt','sr',
                                                         'tr','zh'))) {
    news$lan[i] = 'other'
  }
}
# Count the number of quotation marks in the text, which could indicate legitimacy.
for (i in 1:length(news$text)) {
  x <- news$text[i]
  news$quotenum[i] <- lengths(regmatches(x, gregexpr("\"", x)))
}
# Count length of title
for (i in 1:length(news$title)) {
  if (is.na(news$title[i]) == TRUE) {
    news$title[i] = 0
  } else {
    news$title[i] = str_length(news$title[i])
  }
  x <- news$text[i]
  news$quotenum[i] <- lengths(regmatches(x, gregexpr("\"", x)))
}


glimpse(news)



# Create VCorpus for cleaning
doc <- VCorpus(VectorSource(news$text))
# Convert text to lower case
doc <- tm_map(doc, content_transformer(tolower))
# Remove numbers
doc <- tm_map(doc, removeNumbers)
# Remove Punctuations
doc <- tm_map(doc, removePunctuation)
# Remove Stopwords
doc <- tm_map(doc, removeWords, stopwords('english'))
doc <- tm_map(doc, content_transformer(str_remove_all), "[[:punct:]]")
# Remove Whitespace
doc <- tm_map(doc, stripWhitespace)
# Lemmatization
doc <- tm_map(doc, content_transformer(lemmatize_strings))
# inspect output
writeLines(as.character(doc[[45]]))



# Create Tidy data
df.tidy <- tidy(doc)

# Convert dtm to matrix
data.mat <- as.matrix(df.tidy)
dim(data.mat)

# Convert matrix to data frame
news.clean <- as.data.frame(data.mat)
# Replace news$text with cleaned data
news$text <- as.character(news.clean$text)


# Split into training and test sets
train <- news %>% filter(Set == "train") %>% select(-Set)
test <- news %>% filter(Set == "test") %>% select(-Set)
dim(train)
dim(test)


# Create validation set from train
data <- split_test_train(train, .p = 0.85)

# Create wactor environment object
acc_v <- wactor(data$train$text, max_words = 10000)

# Create tf-idf matrices
data.train <- tfidf(acc_v, data$train$text)
data.test  <- tfidf(acc_v, data$test$text)

# Convert to data frames
df.train <- as.data.frame(data.train)
df.test <- as.data.frame(data.train)



# Convert to DMatrix objects
xgb.data <- list(train = xgb_mat(data.train, y = data$train$label),
                 test = xgb_mat(data.test, y = data$test$label))

#####################
## Build XGB Model ##
#####################

params <- list(max_depth = 2,
               eta = 0.4,
               nthread = 60,
               objective = 'binary:logistic'
)

xgb.model <- xgb.train(params,
                       xgb.data$train,
                       nrounds = 100,
                       print_every_n = 50,
                       watchlist = xgb.data
)


# Increase nrounds and train on same model
# xgb.model <- xgb.train(params,
#                        xgb.data$train,
#                        xgb_model = xgb.model,
#                        nrounds = 500,
#                        print_every_n = 50,
#                        watchlist = xgb.data
# )
# 
# formatC(head(predict(xgb.model, xgb.data$test)), format = 'f', digits = 10)



# For actual predictions, prepare the test dataset
test.data <- tfidf(acc_v, test$text)
# Convert to DMatrix
test.xgb.data <- xgb_mat(test.data)
# Predict response using model
test$label <- as.integer(predict(xgb.model, test.xgb.data) >= 0.5)

# Create submission file
write_csv(select(test, id, label), 'wactor-xgb.csv') 
