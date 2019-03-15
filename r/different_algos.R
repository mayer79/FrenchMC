# LIBRARIES

library(CASdatasets)  # Download zip from http://cas.uqam.ca/ and install from local package source
library(rpart)
library(ranger)
library(xgboost)
library(glmnet)
library(lightgbm)
library(DALEX) # For model interpretation
library(ceterisParibus)
library(tidyverse) # For data prep
library(caret) # For data split

# FUNCTIONS
deviance_poisson <- function(y, pred) {
  2 * mean(pred - y + log((y / pred)^y))
}

deviance_gamma <- function(y, pred) {
  2 * mean((y - pred) / pred - log(y / pred))
}

# Transformer for XGBoost
prep_mat <- function(data, response = TRUE) {
  data[["Power"]] <- match(data[["Power"]], letters)
  data[["Gas"]] <- data[["Gas"]] == "Diesel"
  data.matrix(data[c(if (response) y_name, x_name, w_name)])
}

# DATA PREPARATION

# Load the data
data(freMTPLfreq) 

# Some column names
x_name <- c("CarAge", "DriverAge", "Power", "Gas", "Density")
y_name <- "ClaimNb"
w_name <- "Exposure"

# Train/test split
set.seed(3928272)
ind <- caret::createDataPartition(freMTPLfreq[[y_name]], p = 0.80, list = FALSE) %>% c

trainDF <- freMTPLfreq[ind, ]
validDF <- freMTPLfreq[-ind, ]

trainMat <- prep_mat(freMTPLfreq[ind, ])
validMat <- prep_mat(freMTPLfreq[-ind, ])

# Use cross-validation to find best alpha and lambda, the two penalization parameters of elastic net
steps <- 10

for (i in seq_len(steps)) {
  fit_glm <- cv.glmnet(x = trainMat[, x_name], 
                       y = trainMat[, y_name], 
                       offset = log(trainMat[, w_name]),
                       alpha = i / steps, 
                       nfolds = 5, 
                       type.measure = "deviance")
  if (i == 0) cat("\n alpha\t deviance (CV)")
  cat("\n", i / steps, "\t", sqrt(min(fit_glm$cvm)))
}

# Use CV to find best lambda given optimal alpha of 0.5
set.seed(342)
fit_glm <- cv.glmnet(x = trainMat[, x_name], 
                     y = trainMat[, y_name], 
                     alpha = 0.2, 
                     nfolds = 5, 
                     type.measure = "deviance")
cat("Best deviance (CV):", sqrt(min(fit_glm$cvm))) # 2.101321

# # On test sample (always with best lambda)
# pred <- predict(fit_glm, validMat[, x_name], type = "response", newoffset = validMat[, w_name])
# deviance_poisson(validMat[, y_name], pred)

#======================================================================
# One single tree (just for illustration)
#======================================================================

fit_tree <- rpart(reformulate(x_name, y_name), data = trainDF, method = "poisson", cp = 0.001)
plot(fit_tree)
text(fit_tree)
# pred <- predict(fit_tree, testDF)
# perf(test$y, pred) # 0.2929

#======================================================================
# random forest (without tuning - that is their strength ;))
#======================================================================

# Use OOB estimates to find optimal mtry (small number of trees)
for (m in seq_along(x_name)) {
  fit_rf <- ranger(reformulate(x_name, y_name), data = trainDF, num.trees = 50, mtry = m, seed = 37 + m * 3)
  if (m == 1) cat("m\t rmse (OOB)")
  cat("\n", m, "\t", sqrt(fit_rf$prediction.error))
}

# Use optimal mtry to fit 500 trees
m <- 1
fit_rf <- ranger(reformulate(x_name, y_name), data = trainDF, 
                 importance = "impurity", num.trees = 500, 
                 mtry = m, seed = 837363)
cat("Best rmse (OOB):", sqrt(fit_rf$prediction.error)) # 0.002568853 

object.size(fit_rf) # 24 MB

# Log-impurity gains
barplot(fit_rf %>% importance %>% sort %>% log)

# perf(test$y, predict(fit_rf, testDF)$predictions) 

#======================================================================
# gradient boosting with "XGBoost"
#======================================================================

trainXgb <- xgb.DMatrix(trainMat[, x_name], 
                        label = trainMat[, y_name] / trainMat[, w_name], 
                        weight = trainMat[, w_name])

watchlist <- list(train = trainXgb)

# Grid search CV (vary different parameters together first to narrow reasonable range)
paramGrid <- expand.grid(iteration = NA_integer_, # filled by algorithm
                         score = NA_real_,     # "
                         learning_rate = 0.2, #c(0.02, 0.05), # c(0.2, 0.1, 0.05, 0.02, 0.01),
                         max_depth = 5:7, # 1:10, -> 5:6
                         min_child_weight = c(0, 1e-04, 1e-2), # c(0, 10^-(-1:4)) -> 1e-04
                         colsample_bytree = c(0.5, 0.7, 0.9), # seq(0.5, 1, by = 0.1), # 
                         subsample = c(0.8, 1), # seq(0.5, 1, by = 0.1), # ok
                         lambda = 0:2, # c(0, 0.1, 0.5, 1, 2, 3), # l2 penalty
                         alpha = 0:2, # c(0, 0.1, 0.5, 1, 2, 3), # l1 penalty
                         min_split_loss = c(0, 1e-04, 1e-02), # c(0, 10^-(-1:4)), 
                         nthread = 4)

(n <- nrow(paramGrid))
set.seed(342267)
paramGrid <- paramGrid[sample(n, 10), ]
(n <- nrow(paramGrid)) # 100

for (i in seq_len(n)) { # i = 1
  print(i)
  cvm <- xgb.cv(as.list(paramGrid[i, -(1:2)]), 
                trainXgb,     
                nrounds = 5000, # we use early stopping
                nfold = 5,
                objective = "count:poisson",
                showsd = FALSE,
                early_stopping_rounds = 20,
                verbose = 0L)
  paramGrid[i, 1] <- bi <- cvm$best_iteration
  paramGrid[i, 2] <- as.numeric(cvm$evaluation_log[bi, "test_poisson_nloglik_mean"])
  save(paramGrid, file = "paramGrid_xgb.RData")
}

# load("paramGrid_xgb.RData", verbose = TRUE)
head(paramGrid <- paramGrid[order(paramGrid$score), ])

# Best only (no ensembling)
cat("Best deviance (CV):", paramGrid[1, "score"]) # 0.2767028
fit_xgb <- xgb.train(paramGrid[1, -(1:2)], 
                     data = trainXgb, 
                     nrounds = paramGrid[1, "iteration"],
                     objective = "count:poisson")
# pred <- predict(fit_xgb, validMat[, x_name]) * validMat[, w_name]
# deviance_poisson(validMat[, y_name], pred)  

#======================================================================
# gradient boosting with "lightGBM"
#======================================================================

trainLgb <- lgb.Dataset(trainMat[, x_name], 
                        label = trainMat[, y_name] / trainMat[, w_name], 
                        weight = trainMat[, w_name])

# A grid of possible parameters
paramGrid <- expand.grid(iteration = NA_integer_, # filled by algorithm
                         score = NA_real_,     # "
                         learning_rate = c(0.05, 0.02), # c(1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01), # -> 0.02
                         num_leaves = c(31, 63, 127), # c(31, 63, 127, 255),
                         # max_depth = 14,
                         min_data_in_leaf = c(10, 20, 50), # this paramter cannot be tuned without calling lgb.Dataset
                         lambda_l1 = c(0, 0.5, 1),
                         lambda_l2 = 2:6, #  c(0, 0.1, 0.5, 1:6),
                         min_sum_hessian_in_leaf = c(0.001, 0.1), # c(0, 1e-3, 0.1),
                         feature_fraction = c(0.7, 1), # seq(0.5, 1, by = 0.1),
                         bagging_fraction = c(0.8, 1), # seq(0.5, 1, by = 0.1),
                         bagging_freq = 4,
                         nthread = 4,
                         max_bin = 255)

(n <- nrow(paramGrid)) # 2160
set.seed(34234)
paramGrid <- paramGrid[sample(n, 100), ]
(n <- nrow(paramGrid)) # 100

for (i in seq_len(n)) {
  print(i)
  cvm <- lgb.cv(as.list(paramGrid[i, -(1:2)]), 
                trainLgb,     
                nrounds = 5000, # we use early stopping
                nfold = 5,
                objective = "poisson",
                showsd = FALSE,
                early_stopping_rounds = 20,
                verbose = -1,
                metric = "poisson")
  paramGrid[i, 1:2] <- as.list(cvm)[c("best_iter", "best_score")]
  save(paramGrid, file = "paramGrid_lgb.RData") # if lgb crashes
}

# load("paramGrid_lgb.RData", verbose = TRUE)
head(paramGrid <- paramGrid[order(-paramGrid$score), ])

# Use best only (no ensembling)
cat("Best deviance (CV):", -paramGrid[1, "score"]) # 0.2523963

system.time(fit_lgb <- lgb.train(paramGrid[1, -(1:2)], 
                                 data = trainLgb, 
                                 nrounds = paramGrid[1, "iteration"],
                                 objective = "poisson"))

# Interpretation
imp_lgb <- lgb.importance(fit_lgb)
print(imp_lgb)
lgb.plot.importance(imp_lgb, top_n = length(x))

# # Select best and test 
# pred <- predict(fit_lgb, validMat[, x_name]) * validMat[, w_name]
# deviance_poisson(validMat[, y_name], pred) # 0.09463408

# Now use an average of top 3 models
m <- 3

# keep test predictions, no model
predList <- vector(mode = "list", length = m)

for (i in seq_len(m)) {
  print(i)
  fit_temp <- lgb.train(paramGrid[i, -(1:2)], 
                        data = trainLgb, 
                        nrounds = paramGrid[i, "iteration"],
                        objective = "poisson",
                        verbose = -1)
  predList[[i]] <- predict(fit_temp, validMat[, x_name]) * validMat[, w_name]
}
pred <- rowMeans(do.call(cbind, predList))
# # Test
# deviance_poisson(validMat[, y_name], pred) # 0.99132587


#======================================================================
# gradient boosting with "catboost". Strong with categorical input
#======================================================================

# Untuned example
param_list <- list(loss_function = 'RMSE',
                   learning_rate = 0.05,
                   iterations = 10000, 
                   l2_leaf_reg = 0,
                   #bootstrap_type = "Bernoulli",
                   # subsample = 0.8,
                   rsm = 1,
                   eval_metric = 'RMSE',
                   od_wait = 20,
                   od_type = "Iter",
                   use_best_model = TRUE,
                   depth = 4,
                   thread_count = 7,
                   border_count = 128,
                   metric_period = 100)

# With five-fold CV
folds <- caret::createFolds(train$y, k = 5)
pred_oof <- numeric(length(train$y))
best_iter <- numeric(length(folds))

for (f in folds) { # f <- folds[[1]]
  learn_pool <- catboost.load_pool(train$X[-f, ], label = train$y[-f])
  test_pool <- catboost.load_pool(train$X[f, ], label = train$y[f])
  
  fit_cb <- catboost.train(learn_pool, test_pool, params = param_list)  
  pred_oof[f] <- catboost.predict(fit_cb, test_pool)
}

perf(train$y, pred_oof)


# Grid search
paramGrid <- expand.grid(best_iter = NA_integer_, # filled by algorithm
                         score = NA_real_, 
                         learning_rate = 0.05, #c(0.05, 0.02)
                         depth = 3:10,
                         l2_leaf_reg = 0:5,
                         rsm = c(0.6, 0.8, 1),
                         thread_count = 7,
                         logging_level = "Silent",
                         loss_function = 'RMSE',
                         eval_metric = 'RMSE',
                         iterations = 10000, # early stopping, see next param
                         od_wait = 20,
                         od_type = "Iter",
                         use_best_model = TRUE,
                         border_count = 128,
                         stringsAsFactors = FALSE)

(n <- nrow(paramGrid)) # 2160
set.seed(34234)
paramGrid <- paramGrid[sample(n, 30), ]
(n <- nrow(paramGrid)) # 100

# Create CV folds
folds <- caret::createFolds(train$y, k = 5)

rmse <- function(y, pred) {
  sqrt(mean((y - pred)^2))
}

for (i in seq_len(n)) {
  print(i)
  pred_oof <- numeric(length(folds))
  best_iter <- numeric(length(folds))
  
  for (j in 1:length(folds)) {
    cat(".")
    f <- folds[[j]]
    test_pool = catboost.load_pool(train$X[f, ], label = train$y[f])
    
    fit <- catboost.train(catboost.load_pool(train$X[-f, ], label = train$y[-f]), 
                          test_pool, 
                          params = as.list(paramGrid[i, -(1:2)]))  
    
    pred_oof[j] <- rmse(train$y[f], catboost.predict(fit, test_pool))
    best_iter[j] <- fit$tree_count
  }
  
  paramGrid[i, 1:2] <- c(mean(best_iter), mean(pred_oof))
  save(paramGrid, file = "paramGrid_cb.RData") # if lgb crashes
}


# load("paramGrid_cb.RData", verbose = TRUE)
head(paramGrid <- paramGrid[order(paramGrid$score), ])

# Use best only (no ensembling)
cat("Best rmse (CV):", paramGrid[1, "score"]) # 0.09608951

best_params <- paramGrid[1, ] %>% 
  select(-logging_level, -use_best_model, -iterations, -score, -od_type, -od_wait) %>% 
  rename(iterations = best_iter) %>%
  mutate(metric_period = 100,
         #    task_type = "GPU",
         iterations = round(iterations),
         border_count = 32) %>% 
  unclass

fit_cb <- catboost.train(catboost.load_pool(train$X, label = train$y), 
                         params = best_params)

pred <- catboost.predict(fit_cb, catboost.load_pool(test$X))
perf(test$y, pred) # 0.9905
