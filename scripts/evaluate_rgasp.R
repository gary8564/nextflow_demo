#!/usr/bin/env Rscript

# =============================================================================
# RGaSP (RobustGaSP) Evaluation Script
# =============================================================================
# This script follows the standard GP workflow:
# 1. Parse arguments
# 2. Load data from HDF5
# 3. Train RGaSP model
# 4. Make predictions
# 5. Calculate metrics and save results
# =============================================================================

suppressPackageStartupMessages({
  library(hdf5r)
  library(RobustGaSP)
  library(jsonlite)
  library(optparse)
})

# =============================================================================
# 1. ARGUMENT PARSING
# =============================================================================

parse_arguments <- function() {
  # Parse command line arguments
  
  option_list <- list(
    make_option(c("--input-dir"), type="character", default=NULL,
                help="Input directory containing data.h5", metavar="character"),
    make_option(c("--output-dir"), type="character", default=NULL,
                help="Output directory for metrics.json", metavar="character")
  )
  
  opt_parser <- OptionParser(option_list=option_list)
  opt <- parse_args(opt_parser)
  
  if (is.null(opt$`input-dir`) || is.null(opt$`output-dir`)) {
    print_help(opt_parser)
    stop("Both --input-dir and --output-dir arguments are required.")
  }
  
  return(opt)
}

# =============================================================================
# 2. DATA LOADING
# =============================================================================

load_h5_data <- function(filepath) {
  # Load and validate data from HDF5 file
  
  if (!file.exists(filepath)) {
    stop(paste("HDF5 file not found:", filepath))
  }
  
  cat("[RGaSP] Loading data from:", filepath, "\n")
  
  h5_file <- H5File$new(filepath, mode = "r")
  
  tryCatch({
    # Read datasets
    X_train_raw <- h5_file[["train_X"]]$read()
    y_train_raw <- h5_file[["train_y"]]$read()
    X_test_raw <- h5_file[["test_X"]]$read()
    y_test_raw <- h5_file[["test_y"]]$read()
    
    # Read standardization parameters
    scaler_mean_raw <- h5_file[["scaler_mean"]]$read()
    scaler_scale_raw <- h5_file[["scaler_scale"]]$read()
    
    # Convert to matrices/vectors
    X_train_raw <- as.matrix(X_train_raw)
    y_train_raw <- as.vector(y_train_raw)
    X_test_raw <- as.matrix(X_test_raw)
    y_test_raw <- as.vector(y_test_raw)
    
    # Align X rows with y length: rows must equal number of responses
    y_len <- length(y_train_raw)
    X_train <- X_train_raw
    X_test <- X_test_raw
    if (nrow(X_train) != y_len) {
      if (ncol(X_train) == y_len) {
        cat("[RGaSP] Aligning rows: transposing X matrices to match y length\n")
        X_train <- t(X_train)
        X_test <- t(X_test)
      } else {
        stop(sprintf(
          "Cannot align X and y: y length=%d, X_train dims=%s",
          y_len, paste(dim(X_train_raw), collapse=" x ")
        ))
      }
    }
    
    y_train <- y_train_raw
    y_test <- y_test_raw
    
    # Validate dimensions
    cat(sprintf("[RGaSP] Data dimensions:\n"))
    cat(sprintf("  X_train: %s\n", paste(dim(X_train), collapse=" x ")))
    cat(sprintf("  y_train: %d\n", length(y_train)))
    cat(sprintf("  X_test:  %s\n", paste(dim(X_test), collapse=" x ")))
    cat(sprintf("  y_test:  %d\n", length(y_test)))
    
    # Validate consistency
    if (nrow(X_train) != length(y_train)) {
      stop(sprintf("Dimension mismatch: X_train has %d rows but y_train has %d elements", 
                   nrow(X_train), length(y_train)))
    }
    if (nrow(X_test) != length(y_test)) {
      stop(sprintf("Dimension mismatch: X_test has %d rows but y_test has %d elements", 
                   nrow(X_test), length(y_test)))
    }
    if (ncol(X_train) != ncol(X_test)) {
      stop(sprintf("Feature mismatch: X_train has %d features but X_test has %d features", 
                   ncol(X_train), ncol(X_test)))
    }
    
    data <- list(
      X_train = X_train,
      y_train = y_train,
      X_test = X_test,
      y_test = y_test,
      scaler_mean = scaler_mean_raw,
      scaler_scale = scaler_scale_raw
    )
    
    cat("[RGaSP] Data loading completed successfully\n")
    return(data)
    
  }, finally = {
    h5_file$close_all()
  })
}

# =============================================================================
# 3. MODEL TRAINING
# =============================================================================

train_rgasp_model <- function(X_train, y_train) {
  # Train RGaSP model with timing
  
  cat("[RGaSP] Training RGaSP model...\n")
  start_time <- Sys.time()
  
  tryCatch({
    model <- rgasp(
      design = X_train, 
      response = y_train, 
      kernel_type = "matern_5_2",
      alpha = 1.9,  # For robustness
      optimization = "lbfgs"
    )
    
    end_time <- Sys.time()
    training_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
    
    cat(sprintf("[RGaSP] Training completed in %.2f seconds\n", training_time))
    
    return(list(model = model, training_time = training_time))
    
  }, error = function(e) {
    cat("[RGaSP] Error during training:", e$message, "\n")
    stop(e)
  })
}

# =============================================================================
# 4. PREDICTION
# =============================================================================

make_predictions <- function(model, X_test) {
  # Make predictions with timing
  
  cat("[RGaSP] Making predictions...\n")
  start_time <- Sys.time()
  
  tryCatch({
    predictions <- predict(model, X_test, interval_data = TRUE)
    
    end_time <- Sys.time()
    infer_time <- as.numeric(difftime(end_time, start_time, units = "secs"))
    
    cat(sprintf("[RGaSP] Inference completed in %.2f seconds\n", infer_time))
    
    return(list(predictions = predictions, infer_time = infer_time))
    
  }, error = function(e) {
    cat("[RGaSP] Error during prediction:", e$message, "\n")
    stop(e)
  })
}

extract_prediction_components <- function(predictions) {
  # Extract and validate prediction components
  
  cat("[RGaSP] Extracting prediction components...\n")
  
  # Debug prediction structure
  cat(sprintf("[RGaSP] Prediction structure:\n"))
  cat(sprintf("  Class: %s\n", class(predictions)))
  cat(sprintf("  Names: %s\n", paste(names(predictions), collapse=", ")))
  
  # Extract mean predictions
  if (is.null(predictions$mean)) {
    stop("Prediction object does not contain 'mean' component")
  }
  mean_scaled <- as.numeric(predictions$mean)
  
  # Extract standard deviation - RobustGaSP returns 'sd' with interval_data=TRUE
  if (is.null(predictions$sd)) {
    stop("Prediction object does not contain 'sd' component. This should not happen with interval_data=TRUE")
  }
  std_scaled <- as.numeric(predictions$sd)
  
  # Extract confidence intervals - RobustGaSP always provides these with interval_data=TRUE
  if (is.null(predictions$lower95) || is.null(predictions$upper95)) {
    stop("Prediction object does not contain 'lower95' or 'upper95' components. This should not happen with interval_data=TRUE")
  }
  lower95_scaled <- as.numeric(predictions$lower95)
  upper95_scaled <- as.numeric(predictions$upper95)
  
  cat(sprintf("[RGaSP] Prediction dimensions: mean=%d, std=%d, lower95=%d, upper95=%d\n", 
              length(mean_scaled), length(std_scaled), length(lower95_scaled), length(upper95_scaled)))
  
  return(list(
    mean_scaled = mean_scaled,
    std_scaled = std_scaled,
    lower95_scaled = lower95_scaled,
    upper95_scaled = upper95_scaled
  ))
}

# =============================================================================
# 5. METRICS CALCULATION
# =============================================================================

descale_predictions <- function(test_y, mean_scaled, std_scaled, lower95_scaled, upper95_scaled, 
                               scaler_mean, scaler_scale) {
  # Descale predictions to original scale
  
  # Get scaling parameters for y (last column)
  mu <- scaler_mean[length(scaler_mean)]
  sigma <- scaler_scale[length(scaler_scale)]
  
  cat(sprintf("[RGaSP] Descaling with mu=%.6f, sigma=%.6f\n", mu, sigma))
  
  return(list(
    ground_truth = test_y * sigma + mu,
    mean = mean_scaled * sigma + mu,
    std = std_scaled * sigma,
    lower95 = lower95_scaled * sigma + mu,
    upper95 = upper95_scaled * sigma + mu
  ))
}

calculate_rmse <- function(predictions, ground_truth) {
  # Calculate RMSE between predictions and ground truth
  sqrt(mean((predictions - ground_truth)^2))
}

save_results <- function(metrics, output_dir) {
  # Save metrics to JSON file
  
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  output_file <- file.path(output_dir, "metrics.json")
  write_json(metrics, output_file, auto_unbox = TRUE, digits = 6)
  
  cat(sprintf("[RGaSP] Metrics saved to %s\n", output_file))
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main <- function() {
  # Main execution function following standard GP workflow
  
  cat("[RGaSP] Starting evaluation...\n")
  
  # 1. Parse arguments
  opt <- parse_arguments()
  
  # 2. Load data
  hdf5_file <- file.path(opt$`input-dir`, "data.h5")
  data <- load_h5_data(hdf5_file)
  
  cat(sprintf("[RGaSP] Loaded data: %d training samples, %d test samples, %d features\n", 
              nrow(data$X_train), nrow(data$X_test), ncol(data$X_train)))
  
  # 3. Train model
  train_result <- train_rgasp_model(data$X_train, data$y_train)
  
  # 4. Make predictions
  pred_result <- make_predictions(train_result$model, data$X_test)
  
  # Extract prediction components
  pred_components <- extract_prediction_components(pred_result$predictions)
  
  # 5. Calculate metrics
  cat("[RGaSP] Calculating metrics...\n")
  
  # Descale predictions
  descaled <- descale_predictions(
    data$y_test, 
    pred_components$mean_scaled, 
    pred_components$std_scaled, 
    pred_components$lower95_scaled, 
    pred_components$upper95_scaled,
    data$scaler_mean, 
    data$scaler_scale
  )
  
  # Calculate RMSE
  rmse <- calculate_rmse(descaled$mean, descaled$ground_truth)
  cat(sprintf("[RGaSP] RMSE: %.6f\n", rmse))
  
  # Prepare and save results
  metrics <- list(
    name = "RGaSP",
    ground_truth = as.numeric(descaled$ground_truth),
    predictions_mean = as.numeric(descaled$mean),
    predictions_std = as.numeric(descaled$std),
    predictions_lower95 = as.numeric(descaled$lower95),
    predictions_upper95 = as.numeric(descaled$upper95),
    rmse = rmse,
    train_time = train_result$training_time,
    infer_time = pred_result$infer_time
  )
  
  save_results(metrics, opt$`output-dir`)
  cat("[RGaSP] Evaluation completed successfully!\n")
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

# Execute main function with comprehensive error handling
tryCatch({
  main()
}, error = function(e) {
  cat("Error during RGaSP evaluation:", e$message, "\n")
  cat("Call stack:\n")
  print(sys.calls())
  quit(status = 1)
}, warning = function(w) {
  cat("Warning during RGaSP evaluation:", w$message, "\n")
}) 