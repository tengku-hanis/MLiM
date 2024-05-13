##=============================================================================##
## Title: Decision tree using tidymodels (classification) - intermediate
## Author: Tengku Muhammad Hanis Mokhtar
## Date: May16, 2024
##=============================================================================##

# Compare grid_regular(), grid_random(), grid_latin_hypercube(), and grid_max_entropy()

# Packages ----------------------------------------------------------------

library(tidyverse)
library(tidymodels)
library(mlbench)


# Compare grid approach ---------------------------------------------------

## Regular grid search ----
param_reg_base <- expand.grid(cost_complexity = c(-13, -12, -11), tree_depth = c(10, 20), min_n = c(2, 3))
set.seed(123)
param_reg <- grid_regular(tree_depth(), min_n(), levels = 20)
#levels determine how many values to pick from the range of parameters

## Random grid search ----
set.seed(123)
param_rand <- grid_random(tree_depth(), min_n(), size = 20)
#size determine how many values to pick from the range of parameters

## Grid latin hypercube search ----
set.seed(123)
param_latin <- grid_latin_hypercube(tree_depth(), min_n(), size = 20)
#size determine how many values to pick from the range of parameters

## Grid maximum entropy search ----
set.seed(123)
param_max <- grid_max_entropy(tree_depth(), min_n(), size = 20)
#size determine how many values to pick from the range of parameters

## Compare
param_reg %>% 
  mutate(grid = "regular") %>% 
  bind_rows(
    param_rand %>% 
      mutate(grid = "random")
  ) %>% 
  bind_rows(
    param_latin %>% 
      mutate(grid = "latin")
  ) %>% 
  bind_rows(
    param_max %>% 
      mutate(grid = "max_entropy")
  ) %>% 
  ggplot(aes(tree_depth, min_n)) +
  geom_point() +
  facet_grid(cols = vars(grid))


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
  ) %>% 
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
    cost_complexity = tune(),
    tree_depth = tune(),
    min_n = tune()
  ) %>% 
  set_engine("rpart") %>% 
  set_mode("classification")

## Max entropy search ----
set.seed(123)
param_max <- grid_max_entropy(cost_complexity(), tree_depth(), min_n(), size = 100)

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
            grid = param_max,
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


# Assess on testing data --------------------------------------------------

## Fit on test data ----
pima_pred <- 
  pima_test_process %>% 
  bind_cols(predict(dt_train, new_data = pima_test_process)) %>% 
  bind_cols(predict(dt_train, new_data = pima_test_process, type = "prob"))

## Performance metrics ----
## Accuracy
pima_pred %>% 
  accuracy(truth = diabetes, estimate = .pred_class)

## Plot ROC
pima_pred %>% 
  roc_curve(diabetes, .pred_pos) %>% 
  autoplot()

pima_pred %>% 
  roc_auc(diabetes, .pred_pos)

# Previous result:
# Accuracy: 0.694
# ROC_AUC: 0.819

