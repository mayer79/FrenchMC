#===============================================
#  Data Analysis French MTPL2
#  GLM, Regression Trees and Boosting
#===============================================

# Noll, Alexander and Salzmann, Robert and Wuthrich, Mario V., Case Study: French Motor Third-Party Liability Claims (November 8, 2018). Available at SSRN: https://ssrn.com/abstract=3164764 or http://dx.doi.org/10.2139/ssrn.3164764 

## install CASdatasets
# missing_packages <- setdiff(c("xts", "sp", "lattice"), installed.packages()[, 1])
# if (length(missing_packages)) {
#   install.packages(missing_packages)
# }
# install.packages("CASdatasets", repos = "http://dutangc.free.fr/pub/RRepos/", type = "source")
library(CASdatasets)
library(tidyverse)
# Install lightgbm
# git clone --recursive https://github.com/microsoft/LightGBM
# cd LightGBM
# Rscript build_r.R
library(lightgbm)
library(ranger)
library(MetricsWeighted)
library(flashlight)

data(freMTPL2freq)
str(freMTPL2freq)

#===============================================
# Data preparation
#===============================================

# Covariables & response & weight
x <- c("VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand", 
       "VehGas", "log_density", "Region", "Area")
y <- "Freq"
w <- "Exposure"

# Common preprocessing (slightly different to published version in order to improve interpretability)
dat <- freMTPL2freq %>% 
  mutate(VehGas = factor(VehGas),
         Area = as.integer(Area),
         VehPower = pmin(VehPower, 12),
         VehAge = pmin(VehAge, 20),
         BonusMalus = pmin(BonusMalus, 100),
         log_density = log(Density),
         VehBrand = fct_reorder(VehBrand, parse_number(as.character(VehBrand))), 
         ClaimNb = pmin(ClaimNb, 4),
         Exposure = pmin(Exposure, 1),
         Freq = ClaimNb / Exposure)
str(dat)

# Preprocessing wrapper function for GBM
prep_lgb <- function(dat, x) {
  dat %>% 
    select_at(x) %>% 
    mutate_if(Negate(is.numeric), as.integer) %>% 
    data.matrix()
}

# Data split (80% train, 10% valid, 10% reserve)
set.seed(100)
ind <- caret::createFolds(dat[[y]], k = 10, list = FALSE)
train <- dat[ind <= 8, ]
valid <- dat[ind == 9, ]
test <- dat[ind == 10, ]


#===============================================
# Modelling
#===============================================

# GLM (ca. 1 minute to fit. Can switch to glmnet)
form <- reformulate(x, "Freq")
fit_glm <- glm(form, 
               data = train, 
               family = quasipoisson(), 
               weights = train[[w]])
summary(fit_glm)

# RF (mse as objective, takes 3 minutes to fit)
fit_rf <- ranger(form, 
                 data = train, 
                 num.trees = 500, 
                 case.weights = train[[w]], 
                 min.node.size = ceiling(nrow(train) / 100))

# LGB
dtrain <- lgb.Dataset(prep_lgb(train, x), 
                      label = train[[y]], 
                      weight = train[[w]])

# # Grid search CV (vary different parameters together first to narrow reasonable range)
# paramGrid <- expand.grid(iteration = NA_integer_, # filled by algorithm
#                          score = NA_real_,
#                          learning_rate = 0.2, 
#                          num_leaves = c(16, 31, 63), 
#                          min_data_in_leaf = c(10, 100, 1000), 
#                          lambda_l1 = 2, # 0:3,
#                          lambda_l2 = 2, # 0:5, 
#                          min_sum_hessian_in_leaf = 0.001, # c(0, 0.001, 0.1),
#                          feature_fraction = 1, # c(0.7, 1), 
#                          bagging_fraction = 1, # c(0.8, 1), 
#                          # bagging_freq = 1,
#                          nthread = 7)
# 
# (n <- nrow(paramGrid))
# # set.seed(342267)
# # paramGrid <- paramGrid[sample(n, 10), ]
# # (n <- nrow(paramGrid)) # 100
# 
# pb <- txtProgressBar(0, n, style = 3)
# 
# for (i in seq_len(n)) { # i = 1
#   cvm <- lgb.cv(as.list(paramGrid[i, -(1:2)]), 
#                 dtrain,     
#                 nrounds = 5000, # we use early stopping
#                 nfold = 5,
#                 objective = "poisson",
#                 showsd = FALSE,
#                 early_stopping_rounds = 20,
#                 verbose = -1)
#   paramGrid[i, 1:2] <- as.list(cvm)[c("best_iter", "best_score")]
#   setTxtProgressBar(pb, i)
#   save(paramGrid, file = "paramGrid_freq.RData")
# }

load("paramGrid_freq.RData", verbose = TRUE)
head(paramGrid <- paramGrid[order(paramGrid$score), ])

# 8 seconds to fit
fit_gbm <- lgb.train(paramGrid[1, -(1:2)], 
                     data = dtrain, 
                     nrounds = paramGrid[1, "iteration"],
                     objective = "poisson")


#===============================================
#  Interpretation
#===============================================

# Setting up flashlights on validation data
fl_glm <- flashlight(model = fit_glm, label = "glm", 
                     predict_function = function(fit, X) predict(fit, X, type = "response"))
fl_gbm <- flashlight(model = fit_gbm, label = "gbm", 
                     predict_function = function(fit, X) predict(fit, prep_lgb(X, x)))
fl_rf <- flashlight(model = fit_rf, label = "rf", 
                    predict_function = function(fit, X) predict(fit, X)$predictions)
fls <- multiflashlight(list(fl_glm, fl_rf, fl_gbm), 
                       data = valid, y = y, w = w,
                       metrics = list(Deviance = deviance_poisson, Pseudo_R_squared = r_squared))

# 1) Performance
perf <- light_performance(fls, deviance_function = deviance_poisson)
plot(perf, fill = "darkred")

# 2) Permutation importance with respect to first metric (= Poisson deviance)
imp <- light_importance(fls, v = x, n_max = 10000)
plot(imp, fill = "darkred")

# 3) Effects (click on "Zoom" in R-Studio if plot does not seem to be proper)
# In a R Markdown file, we can loop over most_important(imp, 3) variables
v <- "BonusMalus" # -> use monotonicity constraints in lgb?
# v <- "VehAge"
# v <- "VehBrand"
eff <- light_effects(fls, v = v, counts_weighted = TRUE)
eff$response$counts <- trunc(eff$response$counts) # to avoid ugly count formatting
p <- plot(eff) +
  scale_color_viridis_d(begin = 0.1, end = 0.9)
p
plot_counts(p, eff, alpha = 0.3)

# 4) Additive variable breakdown for first obs
bd <- light_breakdown(fls, new_obs = valid[1, ], n_max = 10000, top_m = 5)
plot(bd, facet_scales = "free_y") # (zoom)
