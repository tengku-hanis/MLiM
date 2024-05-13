##=============================================================================##
## Title: Convolutional Neural Network (CNN)
## Author: Tengku Muhammad Hanis Mokhtar
## Date: May16, 2024
##=============================================================================##

# Basic workflow for CNN

# Packages ----------------------------------------------------------------

library(keras) 
library(dplyr)
library(yardstick)


# Specify path ------------------------------------------------------------

train_dir <- "Data/train"
validation_dir <- "Data/validation"
test_dir <- "Data/test"

train_benign_dir <- file.path(train_dir, "benign")
train_malignant_dir <- file.path(train_dir, "malignant")

validation_benign_dir <- file.path(validation_dir, "benign")
validation_malignant_dir <- file.path(validation_dir, "malignant")

test_benign_dir <- file.path(test_dir, "benign")
test_malignant_dir <- file.path(test_dir, "malignant")


# Check number of samples -------------------------------------------------

num_train_benign <- length(list.files(train_benign_dir))
num_train_malignant <- length(list.files(train_malignant_dir))

num_validation_benign <- length(list.files(validation_benign_dir))
num_validation_malignant <- length(list.files(validation_malignant_dir))

num_test_benign <- length(list.files(test_benign_dir))
num_test_malignant <- length(list.files(test_malignant_dir))

total_train <- num_train_benign + num_train_malignant
total_validation <- num_validation_benign + num_validation_malignant
total_test <- num_test_benign + num_test_malignant


# Check the image ---------------------------------------------------------

img <- imager::load.image("Data/train/benign/384.jpg")
plot(img)

# Data generator ----------------------------------------------------------

# Data generator
image_gen <- image_data_generator()

# Define batch size and image size
batch_size <- 100
IMG_HEIGHT <- 224  
IMG_WIDTH <- 224

# Data generator for each set
train_data_gen <- flow_images_from_directory(train_dir,
                                             generator = image_gen,
                                             batch_size = batch_size,
                                             target_size = c(IMG_HEIGHT,
                                                             IMG_WIDTH),
                                             class_mode = "binary")

validation_data_gen <- flow_images_from_directory(validation_dir,
                                                  generator = image_gen,
                                                  batch_size = batch_size,
                                                  target_size = c(IMG_HEIGHT,
                                                                  IMG_WIDTH),
                                                  class_mode = "binary")

test_data_gen <- flow_images_from_directory(test_dir,
                                            generator = image_gen,
                                            batch_size = batch_size,
                                            shuffle = FALSE,
                                            target_size = c(IMG_HEIGHT,
                                                            IMG_WIDTH),
                                            class_mode = "binary")


# Deep learning layer -----------------------------------------------------

# DL layers
inputs <- layer_input(shape = c(224, 224, 3))

outputs <- inputs %>%
  layer_rescaling(1 / 255) %>%
  layer_conv_2d(filters = 32, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_conv_2d(filters = 256, kernel_size = 3, activation = "relu") %>%
  layer_flatten() %>%
  layer_dropout(0.5) %>%
  layer_dense(1, activation = "sigmoid")

model <- keras_model(inputs, outputs)

# Compile
model %>% compile(loss = "binary_crossentropy",
                  optimizer = "adam",
                  metrics = "accuracy")


# Callbacks ---------------------------------------------------------------

callbacks <- list(
  callback_model_checkpoint(
    filepath = "dl_scratch_raw_images.h5",
    save_best_only = TRUE,
    monitor = "val_loss"
  ),
  callback_early_stopping(
    monitor = "val_loss",
    min_delta = 0.01,
    patience = 4)
)


# Fit the model -----------------------------------------------------------

# This will run about ~ 10-15 min
history <- model %>% fit(
  train_data_gen,
  epochs = 50,
  validation_data = validation_data_gen,
  callbacks = callbacks
)
plot(history)


# Performance on the testing set ------------------------------------------

test_model <- load_model_tf("dl_scratch_raw_images.h5")
result <- evaluate(test_model, test_data_gen)
cat(sprintf("Test accuracy: %.3f\n", result["accuracy"])) 

# Other metrics - ROC plot
metric_dat <- 
  test_model %>% 
  predict(test_data_gen) %>% 
  as_tibble(.name_repair = NULL) %>% 
  rename(prob = V1) %>% 
  mutate(label = test_data_gen$labels,
         label = as.factor(label))

metric_dat %>% 
  roc_curve(label, prob, event_level = "second") %>% 
  tune::autoplot()

# Other metrics - multiset
multi_metric <- metric_set(roc_auc, pr_auc)
multi_metric2 <- metric_set(sens, spec)

metric_dat %>% 
  multi_metric(label, prob, event_level = "second") %>% 
  bind_rows(
    metric_dat %>% 
      mutate(pred = ifelse(prob > 0.5, 1, 0),
             pred = as.factor(pred)) %>% 
      multi_metric2(truth = label, estimate = pred, event_level = "second")
  )

# Prediction --------------------------------------------------------------

# New image
imager::load.image("Data/train/benign/384.jpg") %>% 
  plot()

# Change the new image into a proper format
new_img <- 
  image_load("Data/train/benign/384.jpg", target_size = c(224, 224)) %>% 
  image_to_array() %>% 
  array_reshape(c(1, 224, 224, 3))

# Predict the new image
pred <- 
  test_model %>% 
  predict(new_img) 

pred %>% 
  as_tibble(.name_repair = NULL) %>% 
  mutate(diabetes = ifelse(V1 > 0.5, "malignant", "benign")) %>% 
  rename(prob = V1)
