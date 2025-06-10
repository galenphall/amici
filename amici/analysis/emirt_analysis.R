# emIRT Analysis of Supreme Court Amicus Brief Data
# This script loads the prepared data and runs bootstrapped binary IRT

# Load required libraries
library(emIRT)
library(dplyr)
library(ggplot2)
library(corrplot)

# Set working directory and data path
data_dir <- "emirt_data"

# Set small noise for starting values
noise <- 0.01

# Function to load and prepare data
load_amicus_data <- function(data_dir) {
  cat("Loading amicus brief data...\n")
  
  # Load vote matrix
  vote_matrix <- read.csv(file.path(data_dir, "vote_matrix.csv"), row.names = 1)
  
  # Load metadata
  group_metadata <- read.csv(file.path(data_dir, "group_metadata.csv"))
  case_metadata <- read.csv(file.path(data_dir, "case_metadata.csv"))
  summary_stats <- read.csv(file.path(data_dir, "summary_stats.csv"))
  
  cat(sprintf("Loaded data: %d groups, %d cases\n", 
              nrow(vote_matrix), ncol(vote_matrix)))
  cat(sprintf("Data density: %.2f%%\n", summary_stats$density * 100))
  
  return(list(
    votes = as.matrix(vote_matrix),
    groups = group_metadata,
    cases = case_metadata,
    summary = summary_stats
  ))
}

# Function to convert data to emIRT format
prepare_emirt_format <- function(vote_matrix) {
  cat("Converting to emIRT format...\n")
  
  # emIRT expects a specific list format
  # votes should be 1 for "yea", -1 for "nay", 0 for missing
  rc_data <- list(
    votes = vote_matrix,
    n = nrow(vote_matrix),  # number of legislators/groups
    m = ncol(vote_matrix)   # number of bills/cases
  )
  
  return(rc_data)
}

# Function to run binary IRT with multiple starting values
run_binary_irt <- function(rc_data, n_starts = 3) {
  cat("Running binary IRT estimation...\n")

  n_groups <- rc_data$n
  n_cases <- rc_data$m
  
  # Very conservative priors for sparse data
  priors <- makePriors(n_groups, n_cases, 1)
  
  # Make priors much more informative to help convergence
  priors$x$sigma <- priors$x$sigma * 0.25      # Very tight priors on ideal points
  priors$beta$sigma <- priors$beta$sigma * 0.1  # Very tight priors on item parameters
  
  results <- list()
  
  for (i in 1:n_starts) {
    cat(sprintf("  Starting value set %d/%d...\n", i, n_starts))
        
    # Generate starting values closer to zero
    starts <- getStarts(n_groups, n_cases, 1, .type = "zeros")
    
    # Add small random noise to starting values
    starts$x <- starts$x + rnorm(n_groups, 0, noise)
    starts$alpha <- starts$alpha + rnorm(n_cases, 0, noise)
    starts$beta <- starts$beta + rnorm(n_cases, 0, noise)
    
    # Run binary IRT
    result <- tryCatch({
      binIRT(.rc = rc_data,
             .starts = starts,
             .priors = priors,
             .control = list(
               threads = 1,
               verbose = FALSE,
               thresh = 1e-6,
               maxit = 500
             ))
    }, error = function(e) {
      cat(sprintf("    Error in run %d: %s\n", i, e$message))
      return(NULL)
    })
    
    if (!is.null(result) && result$runtime$conv == 1) {
      results[[i]] <- result
      cat(sprintf("    Converged in %d iterations\n", result$runtime$iters))
    } else {
      cat(sprintf("    Did not converge\n"))
    }
  }
  
  if (length(results) == 0) {
    stop("No successful IRT runs. Check data quality or adjust parameters.")
  }
  
  # Return the result with the highest likelihood (if available)
  # For simplicity, return the first successful result
  cat(sprintf("Successfully completed %d/%d runs\n", length(results), n_starts))
  return(results[[1]])
}

# Function to run bootstrapped standard errors
run_bootstrap <- function(irt_result, rc_data, n_bootstrap = 50) {
  cat(sprintf("Running bootstrap with %d iterations...\n", n_bootstrap))
  
  n_groups <- rc_data$n
  n_cases <- rc_data$m

  # Very conservative priors for sparse data
  priors <- makePriors(n_groups, n_cases, 1)
  
  # Make priors much more informative to help convergence
  priors$x$sigma <- priors$x$sigma * 0.25      # Very tight priors on ideal points
  priors$beta$sigma <- priors$beta$sigma * 0.1  # Very tight priors on item parameters

  # Generate starting values closer to zero
  starts <- getStarts(n_groups, n_cases, 1, .type = "zeros")

  # Add small random noise to starting values
  starts$x <- starts$x + rnorm(n_groups, 0, noise)
  starts$alpha <- starts$alpha + rnorm(n_cases, 0, noise)
  starts$beta <- starts$beta + rnorm(n_cases, 0, noise)
  
  # Run bootstrap
  boot_result <- tryCatch({
    boot_emIRT(
      emIRT.out = irt_result,
      .data = rc_data,
      .starts = starts,
      .priors = priors,
      .control = list(
        threads = 1,
        verbose = FALSE,
        thresh = 1e-6,
        maxit = 500
      ),
      Ntrials = n_bootstrap,
      verbose = max(1, floor(n_bootstrap / 10))  # Progress every 10%
    )
  }, error = function(e) {
    cat(sprintf("Bootstrap error: %s\n", e$message))
    return(NULL)
  })
  
  return(boot_result)
}

# Function to analyze and visualize results
analyze_results <- function(irt_result, boot_result, group_metadata, case_metadata) {
  cat("Analyzing results...\n")
  
  # Extract ideal points
  ideal_points <- irt_result$means$x[, 1]
  names(ideal_points) <- group_metadata$group_names
  
  # Extract standard errors if bootstrap was successful
  if (!is.null(boot_result) && !is.null(boot_result$bse)) {
    std_errors <- boot_result$bse$x
  } else {
    std_errors <- rep(NA, length(ideal_points))
  }
  
  # Create results data frame
  results_df <- data.frame(
    group = group_metadata$group_names,
    ideal_point = ideal_points,
    std_error = std_errors,
    lower_ci = ideal_points - 1.96 * std_errors,
    upper_ci = ideal_points + 1.96 * std_errors,
    stringsAsFactors = FALSE
  )
  
  # Sort by ideal point
  results_df <- results_df[order(results_df$ideal_point), ]
  
  # Print top and bottom groups
  cat("\nMost Conservative Groups:\n")
  print(head(results_df, 10))
  
  cat("\nMost Liberal Groups:\n")
  print(tail(results_df, 10))
  
  return(results_df)
}

# Function to create visualizations
create_visualizations <- function(results_df, irt_result, case_metadata) {
  cat("Creating visualizations...\n")
  
  # 1. Ideal point distribution
  p1 <- ggplot(results_df, aes(x = ideal_point)) +
    geom_histogram(bins = 30, alpha = 0.7, fill = "skyblue") +
    labs(title = "Distribution of Group Ideal Points",
         x = "Ideal Point (Conservative → Liberal)",
         y = "Number of Groups") +
    theme_minimal()
  
  ggsave("group_ideal_points_distribution.png", p1, width = 10, height = 6)
  
  # 2. Top/bottom groups with confidence intervals
  n_show <- min(20, nrow(results_df))
  top_bottom <- rbind(head(results_df, n_show/2), tail(results_df, n_show/2))
  top_bottom$group <- factor(top_bottom$group, levels = top_bottom$group)
  
  p2 <- ggplot(top_bottom, aes(x = ideal_point, y = group)) +
    geom_point(size = 2) +
    geom_errorbarh(aes(xmin = lower_ci, xmax = upper_ci), height = 0.2) +
    labs(title = "Most Conservative and Liberal Groups",
         x = "Ideal Point (Conservative → Liberal)",
         y = "Interest Group") +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 8))
  
  ggsave("top_bottom_groups.png", p2, width = 12, height = 8)
  
  # 3. Case parameters (discrimination and difficulty)
  case_params <- data.frame(
    case = case_metadata$case_names,
    year = case_metadata$year,
    alpha = irt_result$means$beta[, 1],  # difficulty
    beta = irt_result$means$beta[, 2]    # discrimination
  )
  
  p3 <- ggplot(case_params, aes(x = alpha, y = beta)) +
    geom_point(alpha = 0.6) +
    labs(title = "Case Parameters",
         x = "Difficulty (α)",
         y = "Discrimination (β)") +
    theme_minimal()
  
  ggsave("case_parameters.png", p3, width = 8, height = 6)
  
  cat("Visualizations saved:\n")
  cat("  - group_ideal_points_distribution.png\n")
  cat("  - top_bottom_groups.png\n")
  cat("  - case_parameters.png\n")
  
  return(list(p1, p2, p3))
}

# Function to save results
save_results <- function(results_df, irt_result, case_metadata, output_file = "amicus_irt_results.csv") {
  cat("Saving results...\n")
  
  # Save group results
  write.csv(results_df, output_file, row.names = FALSE)
  
  # Save case parameters
  case_results <- data.frame(
    case = case_metadata$case_names,
    year = case_metadata$year,
    difficulty = irt_result$means$beta[, 1],
    discrimination = irt_result$means$beta[, 2]
  )
  
  case_output_file <- gsub(".csv", "_cases.csv", output_file)
  write.csv(case_results, case_output_file, row.names = FALSE)
  
  cat(sprintf("Results saved to:\n"))
  cat(sprintf("  - %s (group ideal points)\n", output_file))
  cat(sprintf("  - %s (case parameters)\n", case_output_file))
}

# Main analysis function
main_analysis <- function() {
  cat("=== emIRT Analysis of Supreme Court Amicus Brief Data ===\n\n")
  
  # Load data
  data <- load_amicus_data(data_dir)
  
  # Convert to emIRT format
  rc_data <- prepare_emirt_format(data$votes)
  
  # Run binary IRT
  irt_result <- run_binary_irt(rc_data, n_starts = 3)
  
  # Run bootstrap (optional - set to smaller number for faster execution)
  boot_result <- run_bootstrap(irt_result, rc_data, n_bootstrap = 1000)
  
  # Analyze results
  results_df <- analyze_results(irt_result, boot_result, data$groups, data$cases)
  
  # Create visualizations
  create_visualizations(results_df, irt_result, data$cases)
  
  # Save results
  save_results(results_df, irt_result, data$cases)
  
  cat("\n=== Analysis Complete ===\n")
  
  return(list(
    irt_result = irt_result,
    boot_result = boot_result,
    results_df = results_df,
    data = data
  ))
}

# Run the analysis
if (interactive() || !exists("skip_analysis")) {
  final_results <- main_analysis()
}