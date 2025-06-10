# Sparse Data Diagnostics and Solutions for emIRT
# This script diagnoses why IRT is failing and provides solutions

library(emIRT)
library(dplyr)
library(ggplot2)
library(Matrix)

# Function to diagnose data sparsity issues
diagnose_sparsity <- function(vote_matrix, strategy_name = "unknown") {
  cat(sprintf("\n=== SPARSITY DIAGNOSIS: %s ===\n", toupper(strategy_name)))
  
  n_groups <- nrow(vote_matrix)
  n_cases <- ncol(vote_matrix)
  total_cells <- n_groups * n_cases
  
  # Basic statistics
  clear_votes <- sum(abs(vote_matrix) == 1)
  missing_votes <- sum(vote_matrix == 0)
  density <- clear_votes / total_cells
  
  cat(sprintf("Matrix size: %d groups × %d cases (%d total cells)\n", 
              n_groups, n_cases, total_cells))
  cat(sprintf("Clear votes: %d (%.2f%% density)\n", clear_votes, density * 100))
  cat(sprintf("Missing votes: %d (%.2f%%)\n", missing_votes, missing_votes/total_cells * 100))
  
  # Participation analysis
  group_votes <- rowSums(abs(vote_matrix) == 1)
  case_votes <- colSums(abs(vote_matrix) == 1)
  
  cat(sprintf("\nGroup participation:\n"))
  cat(sprintf("  Mean votes per group: %.1f\n", mean(group_votes)))
  cat(sprintf("  Median votes per group: %.1f\n", median(group_votes)))
  cat(sprintf("  Groups with 0 votes: %d\n", sum(group_votes == 0)))
  cat(sprintf("  Groups with 1 vote: %d\n", sum(group_votes == 1)))
  cat(sprintf("  Groups with 2+ votes: %d\n", sum(group_votes >= 2)))
  cat(sprintf("  Groups with 5+ votes: %d\n", sum(group_votes >= 5)))
  cat(sprintf("  Groups with 10+ votes: %d\n", sum(group_votes >= 10)))
  
  cat(sprintf("\nCase participation:\n"))
  cat(sprintf("  Mean votes per case: %.1f\n", mean(case_votes)))
  cat(sprintf("  Median votes per case: %.1f\n", median(case_votes)))
  cat(sprintf("  Cases with 0 votes: %d\n", sum(case_votes == 0)))
  cat(sprintf("  Cases with 1 vote: %d\n", sum(case_votes == 1)))
  cat(sprintf("  Cases with 2+ votes: %d\n", sum(case_votes >= 2)))
  cat(sprintf("  Cases with 5+ votes: %d\n", sum(case_votes >= 5)))
  cat(sprintf("  Cases with 10+ votes: %d\n", sum(case_votes >= 10)))
  
  # Problem identification
  problems <- c()
  
  if (density < 0.01) problems <- c(problems, "EXTREMELY sparse (<1% density)")
  if (density < 0.05) problems <- c(problems, "Very sparse (<5% density)")
  if (sum(group_votes == 0) > 0) problems <- c(problems, "Groups with no votes")
  if (sum(case_votes == 0) > 0) problems <- c(problems, "Cases with no votes")
  if (sum(group_votes == 1) > n_groups * 0.5) problems <- c(problems, "Many groups with only 1 vote")
  if (sum(case_votes == 1) > n_cases * 0.5) problems <- c(problems, "Many cases with only 1 vote")
  if (mean(group_votes) < 2) problems <- c(problems, "Very low average group participation")
  if (mean(case_votes) < 3) problems <- c(problems, "Very low average case participation")
  
  if (length(problems) > 0) {
    cat(sprintf("\n🚨 PROBLEMS IDENTIFIED:\n"))
    for (p in problems) cat(sprintf("  - %s\n", p))
  }
  
  return(list(
    density = density,
    group_votes = group_votes,
    case_votes = case_votes,
    problems = problems,
    n_groups = n_groups,
    n_cases = n_cases
  ))
}

# Function to aggressively filter data for IRT
filter_for_irt <- function(vote_matrix, min_group_votes = 5, min_case_votes = 5, 
                          min_density = 0.05) {
  cat(sprintf("\n=== AGGRESSIVE FILTERING FOR IRT ===\n"))
  cat(sprintf("Filters: min_group_votes=%d, min_case_votes=%d, min_density=%.1f%%\n", 
              min_group_votes, min_case_votes, min_density * 100))
  
  original_size <- c(nrow(vote_matrix), ncol(vote_matrix))
  cat(sprintf("Original size: %d × %d\n", original_size[1], original_size[2]))
  
  # Iteratively filter until stable
  max_iterations <- 10
  for (i in 1:max_iterations) {
    old_size <- c(nrow(vote_matrix), ncol(vote_matrix))
    
    # Filter groups with too few votes
    group_votes <- rowSums(abs(vote_matrix) == 1)
    keep_groups <- group_votes >= min_group_votes
    
    if (sum(keep_groups) == 0) {
      cat("❌ No groups meet minimum vote requirement\n")
      return(NULL)
    }
    
    vote_matrix <- vote_matrix[keep_groups, , drop = FALSE]
    
    # Filter cases with too few votes
    case_votes <- colSums(abs(vote_matrix) == 1)
    keep_cases <- case_votes >= min_case_votes
    
    if (sum(keep_cases) == 0) {
      cat("❌ No cases meet minimum vote requirement\n")
      return(NULL)
    }
    
    vote_matrix <- vote_matrix[, keep_cases, drop = FALSE]
    
    new_size <- c(nrow(vote_matrix), ncol(vote_matrix))
    
    # Check if size changed
    if (all(new_size == old_size)) {
      cat(sprintf("✓ Converged after %d iterations\n", i))
      break
    }
    
    if (i == max_iterations) {
      cat(sprintf("⚠️ Reached max iterations (%d)\n", max_iterations))
    }
  }
  
  # Check final density
  clear_votes <- sum(abs(vote_matrix) == 1)
  total_cells <- nrow(vote_matrix) * ncol(vote_matrix)
  final_density <- clear_votes / total_cells
  
  cat(sprintf("Final size: %d × %d (%.1f%% of original)\n", 
              new_size[1], new_size[2], 
              (new_size[1] * new_size[2]) / (original_size[1] * original_size[2]) * 100))
  cat(sprintf("Final density: %.2f%%\n", final_density * 100))
  
  if (final_density < min_density) {
    cat(sprintf("⚠️ Still below target density (%.1f%%)\n", min_density * 100))
  }
  
  if (new_size[1] < 20 || new_size[2] < 10) {
    cat("⚠️ Very small final matrix - results may be unreliable\n")
  }
  
  return(vote_matrix)
}

# Function to run IRT with very conservative settings for sparse data
run_sparse_data_irt <- function(vote_matrix, max_attempts = 10) {
  cat(sprintf("\n=== RUNNING SPARSE DATA IRT ===\n"))
  
  if (is.null(vote_matrix) || nrow(vote_matrix) == 0 || ncol(vote_matrix) == 0) {
    cat("❌ No valid data for IRT\n")
    return(NULL)
  }
  
  n_groups <- nrow(vote_matrix)
  n_cases <- ncol(vote_matrix)
  
  # Create emIRT format
  rc_data <- list(
    votes = vote_matrix,
    n = n_groups,
    m = n_cases
  )
  
  # Very conservative priors for sparse data
  priors <- makePriors(n_groups, n_cases, 1)
  
  # Make priors much more informative to help convergence
  priors$x$sigma <- priors$x$sigma * 0.25      # Very tight priors on ideal points
  priors$beta$sigma <- priors$beta$sigma * 0.1  # Very tight priors on item parameters
  
  results <- list()
  
  for (attempt in 1:max_attempts) {
    cat(sprintf("  Attempt %d/%d...\n", attempt, max_attempts))
    
    # Generate starting values closer to zero
    starts <- getStarts(n_groups, n_cases, 1, .type = "zeros")
    
    # Add small random noise to starting values
    starts$x <- starts$x + rnorm(n_groups, 0, 0.1)
    starts$alpha <- starts$alpha + rnorm(n_cases, 0, 0.1)
    starts$beta <- starts$beta + rnorm(n_cases, 0, 0.1)
    
    result <- tryCatch({
      binIRT(.rc = rc_data,
             .starts = starts,
             .priors = priors,
             .control = list(
               threads = 1,
               verbose = FALSE,
               thresh = 1e-4,      # Less strict convergence
               maxit = 2000        # More iterations
             ))
    }, error = function(e) {
      cat(sprintf("    Error: %s\n", e$message))
      return(NULL)
    })
    
    if (!is.null(result)) {
      if (result$runtime$conv == 1) {
        cat(sprintf("    ✓ SUCCESS! Converged in %d iterations\n", result$runtime$iters))
        return(result)
      } else {
        cat(sprintf("    ⚠️ Reached max iterations (%d)\n", result$runtime$iters))
        results[[length(results) + 1]] <- result  # Keep non-converged results as backup
      }
    }
  }
  
  # If no converged results, return best non-converged result if available
  if (length(results) > 0) {
    cat(sprintf("⚠️ No converged results, returning best attempt\n"))
    return(results[[1]])
  }
  
  cat("❌ All attempts failed\n")
  return(NULL)
}

# Alternative: Simple aggregation approach
aggregate_by_year <- function(vote_matrix, case_metadata) {
  cat(sprintf("\n=== TRYING YEAR AGGREGATION ===\n"))
  
  if (is.null(case_metadata$year) || all(is.na(case_metadata$year))) {
    cat("No year information available\n")
    return(NULL)
  }
  
  # Aggregate cases by year
  years <- case_metadata$year[!is.na(case_metadata$year)]
  unique_years <- sort(unique(years))
  
  cat(sprintf("Aggregating %d cases into %d years\n", ncol(vote_matrix), length(unique_years)))
  
  aggregated_matrix <- matrix(0, nrow = nrow(vote_matrix), ncol = length(unique_years))
  rownames(aggregated_matrix) <- rownames(vote_matrix)
  colnames(aggregated_matrix) <- paste0("Year_", unique_years)
  
  for (i in seq_along(unique_years)) {
    year <- unique_years[i]
    year_cases <- which(case_metadata$year == year)
    if (length(year_cases) > 0) {
      year_votes <- vote_matrix[, year_cases, drop = FALSE]
      
      # For each group, aggregate their positions in this year
      for (g in 1:nrow(vote_matrix)) {
        group_year_votes <- year_votes[g, ]
        
        # Simple majority rule: if more petitioner than respondent votes, code as petitioner
        petitioner_votes <- sum(group_year_votes == 1)
        respondent_votes <- sum(group_year_votes == -1)
        
        if (petitioner_votes > respondent_votes) {
          aggregated_matrix[g, i] <- 1
        } else if (respondent_votes > petitioner_votes) {
          aggregated_matrix[g, i] <- -1
        } else {
          aggregated_matrix[g, i] <- 0  # Tie or no votes
        }
      }
    }
  }
  
  # Show results
  clear_votes <- sum(abs(aggregated_matrix) == 1)
  total_cells <- nrow(aggregated_matrix) * ncol(aggregated_matrix)
  density <- clear_votes / total_cells
  
  cat(sprintf("Aggregated matrix: %d × %d (density: %.1f%%)\n", 
              nrow(aggregated_matrix), ncol(aggregated_matrix), density * 100))
  
  return(aggregated_matrix)
}

# Main diagnostic and solution function
solve_sparsity_problem <- function(data_dir = "emirt_data", strategy = "conservative") {
  cat(sprintf("=== SOLVING SPARSITY PROBLEM FOR %s STRATEGY ===\n", toupper(strategy)))
  
  # Load data
  vote_file <- file.path(data_dir, sprintf("vote_matrix_%s.csv", strategy))
  group_file <- file.path(data_dir, sprintf("group_metadata_%s.csv", strategy))
  case_file <- file.path(data_dir, sprintf("case_metadata_%s.csv", strategy))
  
  if (!file.exists(vote_file)) {
    cat(sprintf("❌ Data file not found: %s\n", vote_file))
    return(NULL)
  }
  
  vote_matrix <- as.matrix(read.csv(vote_file, row.names = 1))
  group_metadata <- read.csv(group_file)
  case_metadata <- read.csv(case_file)
  
  # Step 1: Diagnose the problem
  diagnosis <- diagnose_sparsity(vote_matrix, strategy)
  
  # Step 2: Try increasingly aggressive solutions
  solutions <- list()
  
  # Solution 1: Aggressive filtering
  cat(sprintf("\n%s\n", paste(rep("=", 50), collapse = "")))
  cat(sprintf("SOLUTION 1: AGGRESSIVE FILTERING\n"))
  cat(sprintf("%s\n", paste(rep("=", 50), collapse = "")))
  
  filtered_matrix <- filter_for_irt(vote_matrix, min_group_votes = 5, min_case_votes = 5)
  if (!is.null(filtered_matrix)) {
    result1 <- run_sparse_data_irt(filtered_matrix)
    if (!is.null(result1)) {
      solutions[["aggressive_filter"]] <- list(
        result = result1, 
        matrix = filtered_matrix,
        method = "Aggressive filtering (5+ votes)"
      )
    }
  }
  
  # Solution 2: Even more aggressive filtering
  if (length(solutions) == 0) {
    cat(sprintf("\n%s\n", paste(rep("=", 50), collapse = "")))
    cat(sprintf("SOLUTION 2: VERY AGGRESSIVE FILTERING\n"))
    cat(sprintf("%s\n", paste(rep("=", 50), collapse = "")))
    
    filtered_matrix2 <- filter_for_irt(vote_matrix, min_group_votes = 10, min_case_votes = 10)
    if (!is.null(filtered_matrix2)) {
      result2 <- run_sparse_data_irt(filtered_matrix2)
      if (!is.null(result2)) {
        solutions[["very_aggressive_filter"]] <- list(
          result = result2, 
          matrix = filtered_matrix2,
          method = "Very aggressive filtering (10+ votes)"
        )
      }
    }
  }
  
  # Solution 3: Year aggregation
  if (length(solutions) == 0) {
    cat(sprintf("\n%s\n", paste(rep("=", 50), collapse = "")))
    cat(sprintf("SOLUTION 3: YEAR AGGREGATION\n"))
    cat(sprintf("%s\n", paste(rep("=", 50), collapse = "")))
    
    year_matrix <- aggregate_by_year(vote_matrix, case_metadata)
    if (!is.null(year_matrix)) {
      result3 <- run_sparse_data_irt(year_matrix)
      if (!is.null(result3)) {
        solutions[["year_aggregation"]] <- list(
          result = result3, 
          matrix = year_matrix,
          method = "Year aggregation"
        )
      }
    }
  }
  
  # Report results
  if (length(solutions) > 0) {
    cat(sprintf("\n✅ SUCCESS! Found %d working solution(s):\n", length(solutions)))
    for (name in names(solutions)) {
      sol <- solutions[[name]]
      cat(sprintf("  - %s: %d groups × %d cases\n", 
                  sol$method, nrow(sol$matrix), ncol(sol$matrix)))
    }
    
    # Return the first successful solution
    return(solutions[[1]])
  } else {
    cat(sprintf("\n❌ No solutions worked. Data may be too sparse for IRT.\n"))
    cat(sprintf("Consider:\n"))
    cat(sprintf("  - Using network analysis instead of IRT\n"))
    cat(sprintf("  - Aggregating data further (by issue area, decade, etc.)\n"))
    cat(sprintf("  - Using simpler scaling methods (correspondence analysis, etc.)\n"))
    return(NULL)
  }
}

# Run the diagnostic and solution
if (interactive() || !exists("skip_sparse_analysis")) {
  sparse_solution <- solve_sparsity_problem(strategy = "conservative")
}