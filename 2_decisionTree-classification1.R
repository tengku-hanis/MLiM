##=============================================================================##
## Title: Decision tree using tidymodels (classification) - basic
## Author: Tengku Muhammad Hanis Mokhtar
## Date: May16, 2024
##=============================================================================##

# Basic workflow for tidymodels 

# Packages ----------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(mlbench)


# Data --------------------------------------------------------------------

data("PimaIndiansDiabetes2")
pima <- PimaIndiansDiabetes2

## Balanced data 
set.seed(123)
pima2 <- 
  pima %>% 
  filter(diabetes == "neg") %>% 
  slice_sample(n = 268) %>% 
  bind_rows(
    pima %>% 
      filter(diabetes == "pos")
    )

## Explore data ----
skimr::skim(pima2)

# Edit ootcome level
pima2 <- 
  pima2 %>% 
  mutate(diabetes = relevel(diabetes, ref = "pos"))

## Split data ----
set.seed(123)
split_ind <- initial_split(pima2)
pima_train <- training(split_ind)
pima_test <- testing(split_ind)

## Preprocessing ----
pima_rc <- 
  recipe(diabetes~., data = pima_train) %>% 
  step_impute_knn(all_predictors())

pima_train_process <- 
  pima_rc %>% 
  prep() %>% 
  bake(new_data = NULL)

pima_test_process <- 
  pima_rc %>% 
  prep() %>% 
  bake(new_data = pima_test)

## 10-fold CV ----
set.seed(123)
pima_cv <- vfold_cv(pima_train_process, v = 10)


# Tuning ------------------------------------------------------------------

## Specify model ----
dt_mod <- 
  decision_tree(
    cost_complexity = tune(), #default range of parameters will be used
    tree_depth = tune(),
    min_n = tune()
    ) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")
# cost_complexity()
# tree_depth()

## Specify workflow ----
dt_wf <- workflow() %>% 
  add_model(dt_mod) %>% 
  add_recipe(pima_rc)

## tune_grid ----
set.seed(123)

ctrl <- control_resamples(save_pred = TRUE)
dt_tune <- 
  dt_wf %>% 
  tune_grid(resamples = pima_cv,
            metrics = metric_set(accuracy, roc_auc),
            grid = 10, #10 sets of parameter combination randomly developed from the tuning parameters
            #space-filling design automatically applied here
            control = ctrl)

## Explore tuning result ----
autoplot(dt_tune) + theme_light()
dt_tune %>% collect_metrics() 

dt_tune %>% show_best(metric = "accuracy")
dt_tune %>% show_best(metric = "roc_auc")

dt_tune %>% 
  collect_predictions() %>% 
  group_by(id) %>% # tree_depth also can
  roc_curve(diabetes, .pred_pos) %>% 
  autoplot()

best_tune <- 
  dt_tune %>% 
  select_best(metric = "roc_auc")

## Finalize workflow ----
dt_wf_final <- 
  dt_wf %>% 
  finalize_workflow(best_tune)


# Re-fit on training data -------------------------------------------------

dt_train <- 
  dt_wf_final %>% 
  fit(data = pima_train_process)

# Visualise (not practical)
dt_fit <- 
  dt_train %>% 
  extract_fit_parsnip()
rpart.plot::rpart.plot(dt_fit$fit, roundint=FALSE)

# Another way to visualise
vip::vip(dt_fit)


# Assess on testing data --------------------------------------------------

## Fit on test data ----
pima_pred <- 
  pima_test_process %>% 
  bind_cols(predict(dt_train, new_data = pima_test_process)) %>% 
  bind_cols(predict(dt_train, new_data = pima_test_process, type = "prob"))

## Performance metrics ----
# 1) custom metric set to evaluate performance
test_performance <- metric_set(accuracy, sens, spec, precision, recall, f_meas) #f_meas = F1-score
test_performance(pima_pred, truth = diabetes, estimate = .pred_class)

# 2) Specific metrics
## Accuracy
pima_pred %>% 
  accuracy(truth = diabetes, estimate = .pred_class)

## Plot ROC
pima_pred %>% 
  roc_curve(diabetes, .pred_pos) %>% 
  autoplot()

pima_pred %>% 
  roc_auc(diabetes, .pred_pos)

# 3) Confusion matrix
conf_mat(pima_pred, truth = diabetes, estimate = .pred_class) %>% 
  autoplot("heatmap")

# 4) All available metrics
conf_mat(pima_pred, truth = diabetes, estimate = .pred_class) %>% 
  summary()

