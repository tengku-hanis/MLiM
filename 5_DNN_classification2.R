##=============================================================================##
## Title: Deep neural network for classification - intermediate
## Author: Tengku Muhammad Hanis Mokhtar
## Date: May16, 2024
##=============================================================================##

# Apply cross-validation to DNN for small to intermediate data

# Packages ----------------------------------------------------------------

library(dplyr)
library(ggplot2)
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
pima_x_train_full <- 
  pima_train_processed %>% 
  select(-diabetes) %>% 
  as.matrix()
pima_y_train_full <- 
  pima_train_processed %>% 
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


# Define the model --------------------------------------------------------

build_model <- function() {
  model <- 
    keras_model_sequential() %>%
    layer_dense(units = 128, 
                activation = "relu") %>%
    layer_dense(units = 64, 
                activation = "relu") %>%
    layer_dense(units = 1,
                activation = "sigmoid")
  
  model %>% compile(
    optimizer = "rmsprop",
    loss = "binary_crossentropy",
    metrics = c("accuracy", metric_auc())
  )
}


# Cross-validation --------------------------------------------------------

# Define 10-fold CV
k <- 10
fold_id <- sample(rep(1:k, length.out = nrow(pima_x_train_full)))
num_epochs <- 50

# Create a blank list
all_scores_loss <- vector(mode = "list", length = length(1:k))
names(all_scores_loss) <- paste0("fold_", seq_along(1:k))

all_scores_acc <- vector(mode = "list", length = length(1:k))
names(all_scores_acc) <- paste0("fold_", seq_along(1:k))

all_scores_auc <- vector(mode = "list", length = length(1:k))
names(all_scores_auc) <- paste0("fold_", seq_along(1:k))

# Loop
set.seed(123)
for (i in 1:k) {
  cat("Processing fold #", i, "\n")
  
  # Prepare the validation data
  val_indices <- which(fold_id == i)
  val_data <- pima_x_train_full[val_indices, ]
  val_targets <- pima_y_train_full[val_indices]
  
  # Prepare the partial training data
  partial_train_data <- pima_x_train_full[-val_indices, ]
  partial_train_targets <- pima_y_train_full[-val_indices]
  
  # Predefined keras model
  model <- build_model()
  
  # Fit the model, (silent mode, verbose = 0)
  history <- 
    model %>% fit(
    partial_train_data,
    partial_train_targets,
    validation_data = list(val_data, val_targets),
    epochs = num_epochs,
    batch_size = 16, 
    verbose = 0
    )

  # Result of validation data
  all_scores_loss[[i]] <- history$metrics[4]
  all_scores_acc[[i]] <- history$metrics[5]
  all_scores_auc[[i]] <- history$metrics[6]
  }

# Result of 10-fold CV
scores_dat <- 
  all_scores_loss %>%  
  unlist() %>% 
  as_tibble() %>% 
  mutate(metric = "loss") %>%
  bind_rows(
    all_scores_acc %>%  
      unlist() %>% 
      as_tibble() %>% 
      mutate(metric = "acc")
  ) %>% 
  bind_rows(
    all_scores_auc %>%  
      unlist() %>% 
      as_tibble() %>% 
      mutate(metric = "auc")
    )
  
scores_dat
scores_dat %>% 
  group_by(metric) %>% 
  summarise(across(everything(), list(mean = mean, min = min, max = max)))

# Plot
scores_dat %>%
  mutate(epoch = rep(1:50, times = 30)) %>% 
  ggplot(aes(epoch, value)) +
  geom_point(alpha = 0.3) +
  geom_smooth() +
  geom_vline(xintercept = 10, colour = "red") +
  facet_wrap(~metric, scales = "free_y") +
  theme_bw()
  

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
      epochs = 10,
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
