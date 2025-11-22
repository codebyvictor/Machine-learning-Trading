#########################################
#    cluster_run.R 
#########################################

# ==== Packages  ====
required_pkgs <- c(
  "dplyr","tidyr","lubridate","tibble","readxl","janitor",
  "quantmod","xts","zoo","here","writexl","readr"
)
missing <- setdiff(required_pkgs, installed.packages()[,"Package"])
if (length(missing) > 0) install.packages(missing, repos = "https://cloud.r-project.org")
suppressPackageStartupMessages(lapply(required_pkgs, library, character.only = TRUE))

# ==== Parameters ====
CLUSTER_MODE        <- "hclust"   # "hclust", "kmeans", "kmeans_rolling", "static_hclust"
BETA_LOOKBACK_DAYS  <- 180        # rolling beta window
CL_LOOKBACK_DAYS    <- 180        # clustering lookback
REB_FREQ            <- "weeks"    # for rolling modes: "weeks" or "months"
FIRST_TEST_YEAR     <- 2020       # only used to anchor static k-means window (pre-test end)
SEED                <- 1234
set.seed(SEED)

# ==== Source clustering functions ====
source(here::here("Functions/ClusterBetas.R"), local = FALSE)

# ==== Helper: download returns for 1 symbol ====
get_ret_xts <- function(sym, from, to) {
  sym_y <- gsub("\\.", "-", sym) 
  tryCatch({
    x <- quantmod::getSymbols(sym_y, auto.assign = FALSE, from = from, to = to)
    r <- quantmod::dailyReturn(quantmod::Ad(x), type = "arithmetic")
    colnames(r) <- sym_y
    r
  }, error = function(e) NULL)
}

# ==== Inputs: holdings (tickers) ====
hold <- readxl::read_excel("here("Data/spy_holdings.xlsx")) %>% janitor::clean_names()
sym_col <- if ("symbol" %in% names(hold)) "symbol" else "Symbol"
tickers <- unique(hold[[sym_col]])

# ==== Date span for prices ====
# Start a few years before any testing to ensure betas can form
PRICES_START <- as.Date(sprintf("%d-01-01", FIRST_TEST_YEAR - 3))
PRICES_END   <- Sys.Date()

# ==== Download SPY (benchmark) and constituents ====
spy_xts <- quantmod::getSymbols("SPY", auto.assign = FALSE, from = PRICES_START, to = PRICES_END)
spy_ret <- quantmod::dailyReturn(quantmod::Ad(spy_xts), type = "arithmetic"); colnames(spy_ret) <- "SPY"

message("Downloading ", length(tickers), " tickers from Yahoo...")
rets_list <- lapply(tickers, get_ret_xts, from = PRICES_START, to = PRICES_END)
rets_list <- Filter(Negate(is.null), rets_list)

# Always include SPY
all_series <- c(list(spy_ret), rets_list)
stock_rets_xts <- do.call(merge.xts, c(all_series, list(join = "outer")))
# keep rows where SPY exists
stock_rets_xts <- stock_rets_xts[!is.na(stock_rets_xts$SPY), ]

# To tibble
stock_rets <- tibble(date = as.Date(index(stock_rets_xts))) %>%
  dplyr::bind_cols(as_tibble(stock_rets_xts))

# Long form (exclude SPY from cross-sectional averages)
stock_rets_long <- stock_rets %>%
  tidyr::pivot_longer(-date, names_to = "ticker", values_to = "daily_return") %>%
  dplyr::filter(!is.na(daily_return), ticker != "SPY")

# ==== Rolling betas (custom 180-day window) ====
roll_cov <- function(x, y, width) {
  n <- length(x); out <- rep(NA_real_, n)
  for (i in seq_len(n)) if (i >= width)
    out[i] <- cov(x[(i-width+1):i], y[(i-width+1):i], use = "pairwise.complete.obs")
  out
}
roll_var <- function(x, width) {
  n <- length(x); out <- rep(NA_real_, n)
  for (i in seq_len(n)) if (i >= width)
    out[i] <- var(x[(i-width+1):i], na.rm = TRUE)
  out
}

R <- stock_rets %>% arrange(date)
spy_vec <- R$SPY
betas_mat <- matrix(NA_real_, nrow = nrow(R), ncol = ncol(R) - 1,
                    dimnames = list(as.character(R$date), names(R)[names(R)!="date"]))
for (nm in colnames(R)[colnames(R) != "date"]) {
  if (nm == "SPY") {
    betas_mat[, nm] <- 1
  } else {
    cov_is  <- roll_cov(R[[nm]], spy_vec, width = BETA_LOOKBACK_DAYS)
    var_spy <- roll_var(spy_vec, width = BETA_LOOKBACK_DAYS)
    betas_mat[, nm] <- as.numeric(cov_is / var_spy)
  }
}
betas <- as_tibble(betas_mat) %>% mutate(date = as.Date(rownames(betas_mat))) %>% relocate(date)
for (nm in setdiff(names(betas), "date")) {
  betas[[nm]] <- zoo::na.locf(betas[[nm]], fromLast = TRUE, na.rm = FALSE)
}

# ==== Choose clustering mode ====
CLUSTER_TAG <- switch(
  CLUSTER_MODE,
  hclust         = "hclust_roll",
  kmeans         = "kmeans_static",
  kmeans_rolling = "kmeans_roll",
  static_hclust  = "hclust_static",
  "unknown"
)

cluster_labels <- NULL
cluster_daily  <- NULL

if (CLUSTER_MODE == "hclust") {
  # Rolling hierarchical clustering on beta paths
  rolling_hc <- cluster_betas_rolling(
    betas_df   = betas,
    rebalance  = REB_FREQ,
    lookback   = CL_LOOKBACK_DAYS,
    method     = "average",
    metric     = "correlation",
    max_k      = 10,
    min_non_na = 0.9,
    seed       = SEED
  )
  if (nrow(rolling_hc) == 0L) stop("No valid rolling clusters produced.")
  # Carry weekly labels to daily, then aggregate daily returns by cluster
  cluster_map <- rolling_hc %>%
    dplyr::group_by(ticker) %>% dplyr::arrange(date, .by_group = TRUE) %>%
    tidyr::fill(cluster, .direction = "down") %>% dplyr::ungroup()
  stock_week   <- stock_rets_long %>% dplyr::mutate(week = lubridate::floor_date(date, "week"))
  cluster_week <- cluster_map        %>% dplyr::rename(week = date)
  stock_with_cluster <- stock_week %>% dplyr::left_join(cluster_week, by = c("week","ticker"))
  cluster_daily <- stock_with_cluster %>%
    dplyr::group_by(date, cluster) %>%
    dplyr::summarise(avg_daily_return = mean(daily_return, na.rm = TRUE), .groups = "drop") %>%
    dplyr::arrange(date, cluster)
  cluster_labels <- rolling_hc
  
} else if (CLUSTER_MODE == "kmeans") {
  # Static K-means on average betas over last CL_LOOKBACK_DAYS before FIRST_TEST_YEAR
  end_for_clusters <- as.Date(sprintf("%d-12-31", FIRST_TEST_YEAR - 1))
  km_betas <- cluster_betas_kmeans(
    betas_df   = betas,
    k          = 5,
    lookback   = CL_LOOKBACK_DAYS,
    scale_vars = TRUE,
    end_date   = end_for_clusters
  )
  labels <- km_betas$cluster_df %>% dplyr::select(ticker, cluster)
  cluster_daily <- stock_rets_long %>%
    dplyr::left_join(labels, by = "ticker") %>%
    dplyr::group_by(date, cluster) %>%
    dplyr::summarise(avg_daily_return = mean(daily_return, na.rm = TRUE), .groups = "drop") %>%
    dplyr::arrange(date, cluster)
  cluster_labels <- labels
  
} else if (CLUSTER_MODE == "kmeans_rolling") {
  rolling_km <- cluster_betas_kmeans_rolling(
    betas_df   = betas,
    k          = 5,
    rebalance  = REB_FREQ,
    lookback   = CL_LOOKBACK_DAYS,
    min_non_na = 0.9,
    nstart     = 25,
    seed       = SEED,
    track      = "greedy"
  )
  cluster_map <- rolling_km %>%
    dplyr::group_by(ticker) %>% dplyr::arrange(date, .by_group = TRUE) %>%
    tidyr::fill(cluster, .direction = "down") %>% dplyr::ungroup()
  stock_week   <- stock_rets_long %>% dplyr::mutate(week = lubridate::floor_date(date, "week"))
  cluster_week <- cluster_map        %>% dplyr::rename(week = date)
  stock_with_cluster <- stock_week %>% dplyr::left_join(cluster_week, by = c("week","ticker"))
  cluster_daily <- stock_with_cluster %>%
    dplyr::group_by(date, cluster) %>%
    dplyr::summarise(avg_daily_return = mean(daily_return, na.rm = TRUE), .groups = "drop") %>%
    dplyr::arrange(date, cluster)
  cluster_labels <- rolling_km
  
} else if (CLUSTER_MODE == "static_hclust") {
  static_hc <- cluster_betas_static(
    betas_df = betas,
    end_date = max(betas$date),
    lookback = CL_LOOKBACK_DAYS,
    method   = "average",
    metric   = "correlation",
    max_k    = 10,
    seed     = SEED
  )
  if (is.null(static_hc)) stop("Static hclust returned NULL (not enough data).")
  labels <- static_hc$cluster_df %>% dplyr::select(ticker, cluster)
  cluster_daily <- stock_rets_long %>%
    dplyr::left_join(labels, by = "ticker") %>%
    dplyr::group_by(date, cluster) %>%
    dplyr::summarise(avg_daily_return = mean(daily_return, na.rm = TRUE), .groups = "drop") %>%
    dplyr::arrange(date, cluster)
  cluster_labels <- labels
} else {
  stop("CLUSTER_MODE must be one of: hclust, kmeans, kmeans_rolling, static_hclust")
}

# ==== Quick summary ====
cat("\n# of dates in cluster_daily:", dplyr::n_distinct(cluster_daily$date), "\n")
cat("Example cluster counts by date:\n")
print(
  cluster_daily %>%
    dplyr::count(date, cluster) %>%
    dplyr::count(date, name = "k") %>%
    dplyr::arrange(date) %>%
    head(10)
)

## ==== Export ====
#out_dir <- "outputs"; if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
#mode_tag <- switch(CLUSTER_MODE,
#                   hclust="hclust_roll", kmeans="kmeans_static",
#                   kmeans_rolling="kmeans_roll", static_hclust="hclust_static", "unknown")
#stamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
#
#readr::write_csv(cluster_labels, file.path(out_dir, paste0("cluster_labels_", mode_tag, "_", stamp, ".csv")))
#readr::write_csv(cluster_daily,  file.path(out_dir, paste0("cluster_daily_",  mode_tag, "_", stamp, ".csv")))
#
## Optional: also Excel bundle
#writexl::write_xlsx(
#  list(cluster_labels = cluster_labels, cluster_daily = cluster_daily),
#  path = file.path(out_dir, paste0("clusters_", mode_tag, "_", stamp, ".xlsx"))
#)

#message("Done. Files in ", out_dir)


var_spy <- roll_var(spy_vec, 180)
tail(var_spy, 10)
