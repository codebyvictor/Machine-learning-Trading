## functions/05_cluster_betas.R
## =============================================================================
## Clustering utilities for beta series
## - gap_statistic():      choose k via Tibshirani et al.'s Gap Statistic
## - compute_distance_matrix(): build distance among tickers' feature vectors
## - make_beta_feature_matrix(): build (tickers x time) matrix of beta features
## - cluster_betas_static():  one-shot hierarchical clustering on a window
## - cluster_betas_rolling(): rolling (weekly/monthly) hierarchical clustering
## - cluster_betas_kmeans():  simple K-means on average betas (level clustering)
##
## Conventions:
##   * betas_df is a data.frame with columns: date, TICKER1, TICKER2, ..., SPY
##   * For hierarchical clustering we use correlation distance on beta paths
##   * For K-means we cluster on averaged beta levels (k fixed = 5 by default)
## =============================================================================



# Gap Statistic on a feature matrix (rows = items, cols = features)
# Purpose: estimate an appropriate number of clusters k by comparing the
#          observed within-cluster dispersion to reference (uniform) datasets.
# Inputs:
#   X        : numeric matrix (rows = items, cols = features)
#   max_k    : max number of clusters to evaluate
#   n_refs   : number of reference datasets for Monte Carlo
#   n_start  : kmeans nstart (robustness)
#   seed     : RNG seed for reproducibility
# Returns:
#   numeric vector gap[1..max_k]; higher is better; usually choose which.max(gap)


gap_statistic <- function(X, max_k = 12, n_refs = 10, n_start = 10, seed = 123) {
  set.seed(seed)
  
  X <- as.matrix(X)
  X <- X[stats::complete.cases(X), , drop = FALSE]
  if (nrow(X) < 2) return(NA_real_)  # not enough items to cluster
  
  # collapse duplicate rows (distinct paths)
  Xu <- unique(X, MARGIN = 1)
  k_cap <- min(max_k, nrow(Xu))
  if (k_cap < 1) return(NA_real_)
  if (k_cap == 1) return(0)  # only one cluster possible; arbitrary gap value
  
  mins <- apply(X, 2, min, na.rm = TRUE)
  maxs <- apply(X, 2, max, na.rm = TRUE)
  
  gap <- rep(NA_real_, k_cap)
  for (k in 1:k_cap) {
    # for k=1, kmeans is defined but trivial; keep for completeness
    km   <- stats::kmeans(X, centers = k, nstart = n_start, iter.max = 100)
    W_k  <- km$tot.withinss
    
    W_ref <- replicate(n_refs, {
      X_ref <- sapply(seq_along(mins), function(j) stats::runif(nrow(X), mins[j], maxs[j]))
      X_ref <- matrix(X_ref, nrow = nrow(X))
      stats::kmeans(X_ref, centers = k, nstart = n_start, iter.max = 100)$tot.withinss
    })
    
    gap[k] <- mean(log(W_ref)) - log(W_k)
  }
  gap
}

# -----------------------------------------------------------------------------
# Distance matrix helper for assets x features matrix
# Purpose: build a distance object for hierarchical clustering.
# Inputs:
#   M       : numeric matrix with rows = items (tickers), cols = features (time)
#   metric  : "correlation" (default) or any method accepted by dist()
# Returns:
#   stats::dist object
# Notes:
#   - For "correlation", we compute distance = sqrt(2 * (1 - corr))
#     on corr across features (time). This emphasizes *shape/co-movement*.
# -----------------------------------------------------------------------------
# ========= FAST + CACHED DISTANCE (robust) =========
.dist_cache <- new.env(parent = emptyenv())

compute_distance_matrix_fast <- function(M, metric = "correlation", cache_key = NULL) {
  # Must be a numeric matrix with ≥2 rows
  if (is.null(M)) return(stats::as.dist(matrix(numeric(0), 0, 0)))
  M <- as.matrix(M)
  storage.mode(M) <- "double"
  if (nrow(M) < 2L) return(stats::as.dist(matrix(numeric(0), 0, 0)))
  
  # cache
  if (!is.null(cache_key) && exists(cache_key, envir = .dist_cache, inherits = FALSE)) {
    return(get(cache_key, envir = .dist_cache, inherits = FALSE))
  }
  
  if (metric == "correlation") {
    Z <- M
    Z[!is.finite(Z)] <- 0
    
    Tn <- ncol(Z)
    if (Tn < 2L) return(stats::as.dist(matrix(numeric(0), 0, 0)))
    
    # All pairwise correlations in one BLAS call; coerce to plain matrix
    C <- tcrossprod(Z) / (Tn - 1)
    C <- as.matrix(C)
    storage.mode(C) <- "double"
    
    # Safety: clamp, symmetrize, set diagonal
    C[!is.finite(C)] <- 0
    C <- pmax(-1, pmin(1, C))
    # ensure it's square before touching diag
    if (is.null(dim(C)) || length(dim(C)) != 2L || nrow(C) != ncol(C)) {
      return(stats::as.dist(matrix(numeric(0), 0, 0)))
    }
    diag(C) <- 1
    C <- (C + t(C)) / 2
    
    Dmat <- sqrt(2 * pmax(0, 1 - C))
    Dmat <- as.matrix(Dmat)
    if (nrow(Dmat) != ncol(Dmat)) {
      return(stats::as.dist(matrix(numeric(0), 0, 0)))
    }
    
    D <- stats::as.dist(Dmat)
  } else {
    # fallback Euclidean/etc.
    D <- stats::dist(M, method = metric)
  }
  
  if (!is.null(cache_key)) assign(cache_key, D, envir = .dist_cache)
  D
}

safe_hclust <- function(D, method = "average") {
  if (!inherits(D, "dist")) return(NULL)
  sz <- attr(D, "Size")
  if (is.null(sz) || !is.finite(sz) || sz < 2L) return(NULL)
  stats::hclust(D, method = method)
}


# -----------------------------------------------------------------------------
# Build (tickers x time) beta-feature matrix for a date window
# Purpose: take betas_df (date + tickers), slice to [start_date, end_date],
#          transpose to (tickers x time), enforce coverage and standardize.
# Inputs:
#   betas_df   : data.frame with columns date + tickers' rolling betas
#   start_date : window start (Date)
#   end_date   : window end   (Date)
#   min_non_na : required fraction of non-NA points per ticker in window
# Returns:
#   numeric matrix M (n_tickers_kept x n_days), z-scored by row; or NULL if
#   not enough tickers remain after coverage filtering
# -----------------------------------------------------------------------------
    make_beta_feature_matrix <- function(betas_df, start_date, end_date, min_non_na = 0.9) {
      sub <- betas_df %>% dplyr::filter(date >= start_date, date <= end_date)
      
      if (nrow(sub) == 0) return(NULL)
      
      B <- sub %>% tibble::column_to_rownames("date") %>% as.matrix()
      
      # drop time columns (dates) that are all NA
      keep_cols <- colSums(!is.na(B)) > 0
      B <- B[, keep_cols, drop = FALSE]
      if (ncol(B) < 2L) return(NULL)  # need ≥2 time points to compute correlation
      
      M <- t(B)
      
      keep_rows <- apply(M, 1, function(x) mean(!is.na(x)) >= min_non_na)
      M <- M[keep_rows, , drop = FALSE]
      if (nrow(M) < 1L) return(NULL)
      
      # z-score each row; constant rows become NA -> we’ll fix below
      M <- t(scale(t(M)))
      
      # replace any remaining NA in a row with that row mean (0 after z-score)
      row_means <- apply(M, 1, function(x) mean(x, na.rm = TRUE))
      for (i in seq_len(nrow(M))) {
        nas <- is.na(M[i, ])
        if (any(nas)) M[i, nas] <- row_means[i]
      }
      
      # If only one ticker remains, caller will handle it
      M
    }
    

# -----------------------------------------------------------------------------
# One-shot hierarchical clustering on a recent lookback window
# Purpose: cluster tickers by *co-movement* of their beta paths (not levels).
# Inputs:
#   betas_df : date + tickers beta table
#   end_date : last date of window
#   lookback : number of trading days to include (e.g., 252 ~ 1Y)
#   method   : hclust linkage ("average" default)
#   metric   : "correlation" (default) or dist() metric
#   max_k    : max clusters for gap statistic selection
#   seed     : RNG seed
# Returns:
#   list(cluster_df, hc, k_opt, gap)
#     - cluster_df: ticker + cluster
#     - hc       : hclust object
#     - k_opt    : chosen number of clusters
#     - gap      : gap vector for diagnostics
# -----------------------------------------------------------------------------
    cluster_betas_static <- function(betas_df, end_date, lookback = 252,
                                     method = "average", metric = "correlation",
                                     max_k = 10, seed = 123) {
      start_date <- end_date - lookback
      
      # Try stricter to looser coverage thresholds until we have ≥2 tickers
      try_thresholds <- c(0.90, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50)
      M <- NULL
      for (thr in try_thresholds) {
        M <- make_beta_feature_matrix(betas_df, start_date, end_date, min_non_na = thr)
        if (!is.null(M) && nrow(M) >= 2L) break
      }
      if (is.null(M) || nrow(M) < 2L) return(NULL)
      
      k_cap <- min(max_k, nrow(unique(M, MARGIN = 1)))
      if (k_cap < 1L) return(NULL)
      
      if (k_cap == 1L) {
        lbl <- rep(1L, nrow(M)); hc <- NULL; gap <- NA_real_; k_opt <- 1L
      } else {
        gap   <- gap_statistic(M, max_k = k_cap, seed = seed)
        k_opt <- suppressWarnings(which.max(gap))
        if (!length(k_opt) || is.na(k_opt) || k_opt < 1L) k_opt <- min(3L, k_cap)
        
        ckey <- sprintf("static|%s|%d×%d", as.integer(end_date), nrow(M), ncol(M))
        D  <- compute_distance_matrix_fast(M, metric = metric, cache_key = ckey)
        hc <- safe_hclust(D, method)
        if (is.null(hc)) { lbl <- rep(1L, nrow(M)); k_opt <- 1L } else { lbl <- stats::cutree(hc, k = k_opt) }
      }
      
      
      list(
        cluster_df = tibble::tibble(ticker = rownames(M), cluster = as.integer(lbl)) |>
          dplyr::arrange(cluster),
        hc   = hc, k_opt = k_opt, gap = gap
      )
    }
    
    # -----------------------------------------------------------------------------
    # Rolling hierarchical clustering on beta paths (robust)
    # -----------------------------------------------------------------------------
    # betas_df   : data.frame with columns date + tickers' rolling betas
    # rebalance  : "weeks" or "months"
    # lookback   : trailing window length in *calendar days* (e.g., 180)
    # method     : hclust linkage (e.g., "average")
    # metric     : "correlation" (default) or any dist() metric supported by compute_distance_matrix
    # max_k      : upper bound for number of clusters to try (capped per-window)
    # min_non_na : required fraction of non-NA values per ticker in window
    # seed       : RNG seed
    #
    # Returns tibble(date, ticker, cluster) with one row per ticker at each rebalance date
    # -----------------------------------------------------------------------------
    cluster_betas_rolling <- function(betas_df, rebalance = "weeks", lookback = 180,
                                      method = "average", metric = "correlation",
                                      max_k = 10, min_non_na = 0.9, seed = 123) {
      all_dates <- sort(unique(betas_df$date))
      rb_dates <- if (rebalance == "weeks") unique(lubridate::floor_date(all_dates, "week"))
      else                       unique(lubridate::floor_date(all_dates, "month"))
      rb_dates <- rb_dates[rb_dates >= (min(all_dates) + lookback)]
      
      out <- vector("list", length(rb_dates))
      names(out) <- as.character(rb_dates)
      
      for (i in seq_along(rb_dates)) {
        end_date   <- rb_dates[i]
        start_date <- end_date - lookback
        
        # Adaptive coverage: relax until we have ≥2 tickers
        try_thresholds <- c(min_non_na, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50)
        M <- NULL
        for (thr in try_thresholds) {
          M <- make_beta_feature_matrix(betas_df, start_date, end_date, min_non_na = thr)
          if (!is.null(M) && nrow(M) >= 2L) break
        }
        if (is.null(M) || nrow(M) < 2L) next
        
        k_cap <- min(max_k, nrow(unique(M, MARGIN = 1)))
        if (k_cap < 1L) next
        
        if (k_cap == 1L) {
          lbl <- rep(1L, nrow(M))
        } else {
          gap   <- gap_statistic(M, max_k = k_cap, seed = seed)
          k_opt <- suppressWarnings(which.max(gap))
          if (!length(k_opt) || is.na(k_opt) || k_opt < 1L) k_opt <- min(3L, k_cap)
          
          ckey <- sprintf("%s|%s|%d×%d", as.integer(start_date), as.integer(end_date), nrow(M), ncol(M))
          D  <- compute_distance_matrix_fast(M, metric = metric, cache_key = ckey)
          hc <- safe_hclust(D, method)
          if (is.null(hc)) { lbl <- rep(1L, nrow(M)) } else { lbl <- stats::cutree(hc, k = k_opt) }
        }
    
        
        out[[i]] <- tibble::tibble(
          date    = as.Date(end_date),
          ticker  = rownames(M),
          cluster = as.integer(lbl)
        )
      }
      
      valid <- Filter(Negate(is.null), out)
      if (length(valid) == 0L) {
        return(tibble::tibble(date = as.Date(character()), ticker = character(), cluster = integer()))
      }
      
      dplyr::bind_rows(valid) |>
        dplyr::arrange(.data[["date"]], .data[["cluster"]], .data[["ticker"]])
    }
    
## ------------------------------------------------------------------
## Simple K-means clustering on betas (level clustering)
## Purpose: cluster by *level* of beta (averaged over a lookback window),
##          rather than co-movement of the entire beta time-series.
## Typical use: quick 5-bucket partition resembling beta-quantiles,
##              but data-driven with K-means.
## Inputs:
##   betas_df   : data.frame with 'date' + ticker columns of rolling betas
##   k          : number of clusters (default 5)
##   lookback   : trailing days to average beta over (default 180)
##   scale_vars : if TRUE, z-score the average betas before K-means
## Returns:
##   list(cluster_df, kmeans_model)
##     - cluster_df: ticker, avg_beta, cluster
##     - kmeans_model: raw kmeans fit (centers, withinss, etc.)
## ------------------------------------------------------------------
    cluster_betas_kmeans <- function(betas_df,
                                     k = 5,
                                     lookback = 180,
                                     scale_vars = TRUE,
                                     end_date = max(betas_df$date)) {

      
      start_date <- as.Date(end_date) - lookback
      sub <- betas_df %>%
        dplyr::filter(date >= start_date & date <= end_date)
      
      if (nrow(sub) == 0L) stop("No beta data found in this window. Check your dates/lookback.")
      
      # Average beta per ticker over the window
      avg_betas <- sub %>%
        dplyr::select(-date) %>%
        summarise_all(~ mean(.x, na.rm = TRUE)) %>%
        t() %>%
        as.data.frame()
      
      colnames(avg_betas) <- "avg_beta"
      avg_betas$ticker <- rownames(avg_betas)
      rownames(avg_betas) <- NULL
      
      avg_betas <- avg_betas %>%
        dplyr::filter(!is.na(avg_beta), ticker != "SPY")
      
      X <- avg_betas$avg_beta
      if (scale_vars) X <- scale(X)
      
      set.seed(123)
      km <- stats::kmeans(X, centers = k, nstart = 25)
      
      cluster_df <- avg_betas %>%
        dplyr::mutate(cluster = km$cluster) %>%
        dplyr::arrange(cluster)
      
      list(cluster_df = cluster_df, kmeans_model = km)
    }
    
    
    
# ------------------------------------------------------------------
# Rolling K-means on beta paths (tickers x time) with label tracking
# ------------------------------------------------------------------
# betas_df   : data.frame with columns: date + tickers' rolling betas
# k          : number of clusters (e.g., 5)
# rebalance  : "weeks" or "months"
# lookback   : trailing window length in trading days (e.g., 180)
# min_non_na : required fraction of non-NA values per ticker in the window
# nstart     : kmeans nstart
# seed       : RNG seed for reproducibility
# track      : "greedy" = greedily match centers across time to keep labels stable
#              (works well for modest k; no extra packages)
#
# Returns tibble(date, ticker, cluster) with a row per ticker at each rebalance date
    cluster_betas_kmeans_rolling <- function(betas_df,
                                             k = 5,
                                             rebalance = "weeks",
                                             lookback = 180,
                                             min_non_na = 0.9,
                                             nstart = 25,
                                             seed = 123,
                                             track = "greedy") {
      set.seed(seed)
      
      # Rebalance endpoints
      all_dates <- sort(unique(betas_df$date))
      rb_dates <- if (rebalance == "weeks") {
        unique(lubridate::floor_date(all_dates, "week"))
      } else {
        unique(lubridate::floor_date(all_dates, "month"))
      }
      rb_dates <- rb_dates[rb_dates >= (min(all_dates) + lookback)]
      
      out <- vector("list", length(rb_dates))
      names(out) <- as.character(rb_dates)
      
      prev_centers <- NULL
      
      # Helper: greedily match rows of A to rows of B to minimize total distance
      greedy_match <- function(A, B) {
        # A: k x p, B: k x p
        kA <- nrow(A); kB <- nrow(B)
        stopifnot(kA == kB)
        unused_B <- seq_len(kB)
        perm <- integer(kA)
        for (i in seq_len(kA)) {
          d <- colSums((t(B[unused_B, , drop = FALSE]) - A[i, ])^2)
          j_local <- which.min(d)
          perm[i] <- unused_B[j_local]
          unused_B <- unused_B[-j_local]
        }
        perm
      }
      
      for (i in seq_along(rb_dates)) {
        end_date   <- rb_dates[i]
        start_date <- end_date - lookback
        
        # Build features (tickers x time), z-scored by row
        M <- make_beta_feature_matrix(betas_df, start_date, end_date, min_non_na = min_non_na)
        if (is.null(M) || nrow(M) < k) next
        
        # K-means on the row-vectors (beta paths)
        km <- stats::kmeans(M, centers = k, nstart = nstart, iter.max = 100)
        
        # Label tracking: align current centers to previous centers to stabilize IDs
        if (!is.null(prev_centers) && identical(track, "greedy")) {
          # Compute a permutation that matches current centers to previous ones
          # Distances in the same feature space (Euclidean on beta paths)
          # We want perm such that "current_cluster j" maps to "previous_cluster perm[j]"
          # We'll compute using prev->current mapping then invert it for relabeling.
          perm_prev_to_curr <- greedy_match(prev_centers, km$centers)
          # Invert mapping to get current->previous label indices
          inv_perm <- integer(length(perm_prev_to_curr))
          inv_perm[perm_prev_to_curr] <- seq_along(perm_prev_to_curr)
          # Relabel current cluster IDs
          relabeled <- inv_perm[km$cluster]
          km$cluster <- relabeled
          # Reorder centers to be consistent with previous label order
          km$centers <- km$centers[inv_perm, , drop = FALSE]
        }
        
        prev_centers <- km$centers
        
        out[[i]] <- tibble::tibble(
          date    = as.Date(end_date),
          ticker  = rownames(M),
          cluster = as.integer(km$cluster)
        )
      }
      
      dplyr::bind_rows(out) %>%
        dplyr::arrange(date, cluster, ticker)
    }
    
