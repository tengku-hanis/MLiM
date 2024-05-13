##=============================================================================##
## Title: Deep neural network for classification - basic
## Author: Tengku Muhammad Hanis Mokhtar
## Date: May16, 2024
##=============================================================================##

# Basic workflow for deep neural network for keras and tensorflow 

# Packages ----------------------------------------------------------------

library(dplyr)
library(recipes)
library(rsample)
library(keras)
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

## Split data ----
set.seed(123)
split_ind <- initial_split(pima2)
pima_train <- training(split_ind)
pima_test <- testing(split_ind)

## Preprocessing ----
pima_rc <- 
  recipe(diabetes~., data = pima_train) %>% 
  step_impute_knn(all_predictors()) %>% 
  step_normalize(all_predictors()) %>% 
  step_dummy(all_nominal_predictors())


pima_train_processed <- 
  pima_rc %>% 
  prep() %>% 
  bake(new_data = NULL)

pima_test_processed <- 
  pima_rc %>% 
  prep() %>% 
  bake(new_data = pima_test)

## Data for DL ----

# Validation set
set.seed(123)
split_ind2 <- initial_split(pima_train_processed)
pima_train2 <- training(split_ind2)
pima_val <- testing(split_ind2)

# Split features and outcome
pima_x_train_full <- 
  pima_train_processed %>% 
  select(-diabetes) %>% 
  as.matrix()
pima_y_train_full <- 
  pima_train_processed %>% 
  select(diabetes) %>% 
  mutate(diabetes = ifelse(diabetes == "pos", 1, 0)) %>% 
  as.matrix()

pima_x_train_partial <- 
  pima_train2 %>% 
  select(-diabetes) %>% 
  as.matrix()
pima_y_train_partial <- 
  pima_train2 %>% 
  select(diabetes) %>% 
  mutate(diabetes = ifelse(diabetes == "pos", 1, 0)) %>% 
  as.matrix()

pima_x_val <- 
  pima_val %>% 
  select(-diabetes) %>% 
  as.matrix()
pima_y_val <- 
  pima_val %>% 
  select(diabetes) %>% 
  mutate(diabetes = ifelse(diabetes == "pos", 1, 0)) %>% 
  as.matrix()

pima_x_test <-
  pima_test_processed %>% 
  select(-diabetes) %>% 
  as.matrix()
pima_y_test <- 
  pima_test_processed %>% 
  select(diabetes) %>% 
  mutate(diabetes = ifelse(diabetes == "pos", 1, 0)) %>% 
  as.matrix()


# Build the model --------------------------------------------------------

model <- 
  keras_model_sequential() %>% 
  layer_dense(units = 128, 
              activation = "relu") %>% 
  layer_dense(units = 64, 
              activation = "relu") %>% 
  layer_dense(units = 1, 
              activation = "sigmoid")


# Compile the model -------------------------------------------------------

model %>% 
  compile(optimizer = "rmsprop",
          loss = "binary_crossentropy",
          metrics = list("accuracy", metric_auc()))


# Fit the model -----------------------------------------------------------

history <- 
  model %>% 
  fit(pima_x_train_partial,
      pima_y_train_partial,
      epochs = 50,
      batch_size = 50,
      validation_data = list(pima_x_val, pima_y_val))

plot(history)


# Re-train the model ------------------------------------------------------

# Build the model
model <- 
  keras_model_sequential() %>% 
  layer_dense(units = 128, 
              activation = "relu") %>% 
  layer_dense(units = 64, 
              activation = "relu") %>% 
  layer_dense(units = 1, 
              activation = "sigmoid")

# Compile
model %>% 
  compile(optimizer = "rmsprop",
          loss = "binary_crossentropy",
          metrics = list("accuracy", metric_auc()))

# Fit
model %>% 
  fit(pima_x_train_full,
      pima_y_train_full,
      epochs = 20,
      batch_size = 16)

# Evaluate the model ------------------------------------------------------

results <- 
  model %>% 
  evaluate(pima_x_test, pima_y_test)
results

# ROC plot
roc_dat <- 
  model %>% 
  predict(pima_x_test) %>% 
  as_tibble(.name_repair = NULL) %>% 
  rename(prob = V1) %>% 
  bind_cols(pima_y_test)

roc_dat %>% 
  mutate(diabetes = as.factor(diabetes)) %>% 
  yardstick::roc_curve(diabetes, prob, event_level = "second") %>% 
  tune::autoplot()


# Prediction --------------------------------------------------------------

# New data
new_data <- 
  pima %>% 
  slice(1:10) %>% 
  select(-diabetes)

# Apply preprocessing
new_data_preprocessed <- 
  pima_rc %>% 
  prep() %>% 
  bake(new_data = new_data) %>% 
  as.matrix()

# Predict a new data
pred <- 
  model %>% 
  predict(new_data_preprocessed) %>% 
  as_tibble()

pred %>% 
  mutate(diabetes = ifelse(V1 > 0.5, "pos", "neg")) %>% 
  rename(prob = V1)

