install.packages("fastDummies")
install.packages("ranger")
install.packages("randomForest")
install.packages("ISLR")
install.packages("plotROC")
install.packages("MASS")
install.packages("corpcor")

library(corpcor)
library(ggplot2)
library(lattice)
library(caret)
library(fastDummies)
library(dplyr)
library(tidyverse)
library(ranger)
library(randomForest)
library(broom)
library(ISLR)
library(plotROC)
library(MASS)
library(gbm)

#Data from 1979 and before has little to no spread data
spread_data <- nfl[nfl$schedule_season>1979,]

nfl$spread_favorite_result <- if_else(
  nfl$spread_favorite_cover_result == "Cover", 1, 0)

#Splitting the data into training and testing data sets
training <- nfl[nfl$schedule_season>1979 & nfl$schedule_season<=2012,]
testing <- nfl[nfl$schedule_season>2012,]

#First Logistic model GLM
spread_glm <- glm(spread_favorite_result ~ 
                    division_matchup + 
                    schedule_sunday + schedule_playoff +
                    spread_outlier + spread_favorite +
                    score_avg_pts_for_roll_lag.x + score_avg_pts_against_roll_lag.x +
                    score_avg_pts_for_roll_lag.y + score_avg_pts_against_roll_lag.y + 
                    weather_rain + weather_snow,
                  family = "binomial",
                  data = training)
summary(spread_glm)


spread_lda <- lda(spread_favorite_result ~ 
                    division_matchup + 
                    schedule_sunday + schedule_playoff +
                    spread_outlier + spread_favorite +
                    score_avg_pts_for_roll_lag.x + score_avg_pts_against_roll_lag.x +
                    score_avg_pts_for_roll_lag.y + score_avg_pts_against_roll_lag.y + 
                    weather_rain + weather_snow,
                  data = training)
fits_lda <- predict(spread_lda, newdata = testing)
confusionMatrix(table(fits_lda$class, testing$spread_favorite_result))

set.seed(1982)
training$spread_favorite_result <- as.factor(training$spread_favorite_result)
testing$spread_favorite_result <- as.factor(testing$spread_favorite_result)
tune_grid_rf <- expand.grid(mtry = 2:11,
                            splitrule = "gini",
                            min.node.size = 10)
train_control_rf <- trainControl(method = "cv",
                                 number = 10,
                                 search = "grid")
spread_rf <- train(spread_favorite_result ~ 
                     division_matchup + 
                     schedule_sunday + schedule_playoff +
                     spread_outlier + spread_favorite +
                     score_avg_pts_for_roll_lag.x + score_avg_pts_against_roll_lag.x +
                     score_avg_pts_for_roll_lag.y + score_avg_pts_against_roll_lag.y + 
                     weather_rain + weather_snow,
                   data = training,
                   method = "ranger",
                   num.trees = 500,
                   importance ="impurity",
                   trControl = train_control_rf,
                   tuneGrid = tune_grid_rf)

spread_rf
plot(spread_rf)

test_preds_rf <- predict(spread_rf, newdata = testing)
confusionMatrix(table(test_preds_rf, testing$spread_favorite_result))

set.seed(99)
grid <- expand.grid(interaction.depth = c(1, 3, 5), 
                    n.trees = seq(100, 500, by = 100),
                    shrinkage = c(.01, 0.001),
                    n.minobsinnode = 10)

trainControl <- trainControl(method = "cv", number = 5)

gbm_boston <- train(spread_favorite_result ~ 
                      division_matchup + 
                      schedule_sunday + schedule_playoff +
                      spread_outlier + spread_favorite +
                      score_avg_pts_for_roll_lag.x + score_avg_pts_against_roll_lag.x +
                      score_avg_pts_for_roll_lag.y + score_avg_pts_against_roll_lag.y + 
                      weather_rain + weather_snow,
                    data = training, 
                    distribution = "bernoulli", 
                    method = "gbm",
                    trControl = trainControl, 
                    tuneGrid = grid,
                    verbose = FALSE)
gbm_boston

test_pred_gbm <- predict(gbm_boston, newdata = testing)
confusionMatrix(test_pred_gbm, testing$spread_favorite_result)
