library(keras)
use_implementation("tensorflow")

library(tensorflow)
library(tfdatasets)
library(tfprobability)
library(pbapply)
library(readr)
library(abind)

config <- tf$ConfigProto(log_device_placement = TRUE)
tfe_enable_eager_execution(device_policy = "silent", config = config)

n_spec <- 9945
n_site <- 1026
n_traits <- 5
n_env <- 20
random_data <- matrix(rbinom(n_site * n_spec, 1, 0.2), nrow = n_site, ncol = n_spec)
random_traits <- matrix(rnorm(n_spec * n_traits), nrow = n_spec, ncol = n_traits)
random_env <- matrix(rnorm(n_site * n_env), nrow = n_site, ncol = n_env)

block_size <- 100
n_spec_blocks <- floor(n_spec / block_size)
n_site_blocks <- floor(n_site / block_size)

div_k <- kronecker(matrix(seq_len(n_spec_blocks*n_site_blocks), n_spec_blocks, byrow = TRUE), 
                   matrix(1, block_size, block_size))
block_num <- 1
get_rows_cols <- function(block_num) {
  grab_num <- apply(div_k, 2, function(x) which(x == block_num))
  non_zero <- which(sapply(grab_num, function(x) length(x) > 0))
  cols <- grab_num[non_zero[1]] %>% unlist
  rows <- non_zero
  return(list(rows = rows, cols = cols))
}

#row_cols <- pblapply(1:max(div_k), get_rows_cols)
#write_rds(row_cols, "data/row_col_blocks.rds")
row_cols <- read_rds("data/row_col_blocks.rds")

total_blocks <- max(div_k)

## randomize rows and columns of data
generate_batches <- function() {
  random_specs <- sample.int(n_spec)
  random_sites <- sample.int(n_site)
  
  random_data_shuffled <- random_data[random_sites, random_specs]
  random_traits_shuffled <- random_traits[random_specs, ]
  random_env_shuffled <- random_env[random_sites, ]
  
  random_data_list <- lapply(row_cols, function(x) list(data = random_data_shuffled[x$rows, x$cols],
                                                        traits = random_traits_shuffled[x$cols, ],
                                                        env = random_env_shuffled[x$rows, ]))
  random_data_array <- lapply(random_data_list, function(x) x$data) %>%
    abind(along = -1)
  random_traits_array <- lapply(random_data_list, function(x) x$traits) %>%
    abind(along = -1)
  random_env_array <- lapply(random_data_list, function(x) x$env) %>%
    abind(along = -1)
  
  data_to_slice <- list(random_data_array, random_traits_array, random_env_array)
  data_to_slice
}

data_to_slice <- generate_batches()

with(tf$device("cpu:0"), {
  data_to_slice <- list(tf$convert_to_tensor(data_to_slice[[1]]), 
                        tf$convert_to_tensor(data_to_slice[[2]], dtype = tf$float32),
                        tf$convert_to_tensor(data_to_slice[[3]], dtype = tf$float32))
  test <- tensor_slices_dataset(data_to_slice) 
  iter <- make_iterator_one_shot(test)
})

iterator_get_next(iter)

epochs <- 1000

n_latent <- 75

create_feature_encoder <-
  function(n_latent, name = NULL) {
    
    keras_model_custom(name = name, function(self) {
      
      self$dens_1 <- 
        layer_dense(units = 32, activation = "relu")
      
      self$dens_2 <- 
        layer_dense(units = 64, activation = "relu")
      
      self$dens_3 <- 
        layer_dense(units = 128, activation = "relu")
      
      self$fc_mean <- layer_dense(units = n_latent, activation = "softplus")
      self$fc_var <- layer_dense(units = n_latent, activation = "softplus")
      
      function(inputs, mask = NULL, training = TRUE) {
        x <- inputs %>% 
          self$dens_1() %>%
          self$dens_2() %>%
          self$dens_3()
        
        list(x %>% self$fc_mean(), x %>% self$fc_var())
      }
    })
  }

trait_encoder <- create_feature_encoder(n_latent)
env_encoder <- create_feature_encoder(n_latent)

weibull_transform <- function(x, lambda, k) {
  lambda * tf$math$pow(-tf$math$log(x), (1 / k))
}

lambda_1 <- 1
k_1 <- 1
lambda_2 <- trait_params[[1]]
k_2 <- trait_params[[2]]
weibull_KL <- function(lambda_1, lambda_2, k_, k_2) {
  tf$reduce_sum(tf$log(k_1 / tf$pow(lambda_1, k_1)) - tf$log(k_2 / tf$pow(lambda_2, k_2)) +
    (k_1 - k_2) * (tf$log(lambda_1) - (0.57721566490153286061 / k_1)) +
    (tf$pow((lambda_1 / lambda_2), k_2) * tf$exp(tf$lgamma((k_2 / k_1) + 1))) - 1)
}

trait_lambda_prior <- 1
trait_k_prior <- 1
env_lambda_prior <- 1
env_k_prior <- 1

optimizer <- tf$train$AdamOptimizer(1e-4)

for(i in seq_len(epochs)) {
  batch <- iterator_get_next(iter)
  trait_noise <- k_random_uniform(c(block_size, n_latent))
  env_noise <- k_random_uniform(c(block_size, n_latent))
  
  with(tf$GradientTape() %as% grad, {
    trait_params <- trait_encoder(batch[[2]])
    trait_latent <- weibull_transform(trait_noise, trait_params[[1]], trait_params[[2]])
    
    env_params <- env_encoder(batch[[3]])
    env_latent <- weibull_transform(env_noise, env_params[[1]], env_params[[2]])
    
    latent_rep <- tf$matmul(trait_latent, tf$transpose(env_latent))
    probs <- (tf$sigmoid(latent_rep) - 0.5) * 2
    
    nll <- tf$reduce_sum(tf$keras$losses$binary_crossentropy(batch[[1]], probs))
    kl <- weibull_KL(trait_lambda_prior, trait_k_prior, trait_params[[1]], trait_params[[2]]) +
      weibull_KL(env_lambda_prior, env_k_prior, env_params[[1]], env_params[[2]])
    loss <- nll + kl
  })
  
  petecosm_gradient <- grad$gradient(loss, c(trait_encoder$variables, env_encoder$variables))
  
  optimizer$apply_gradients(purrr::transpose(
    list(petecosm_gradient, c(trait_encoder$variables, env_encoder$variables))
  ))
}
