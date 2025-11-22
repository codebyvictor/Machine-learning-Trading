# Clear environment, console and plots
rm(list = ls())
cat("\014")
if (!is.null(dev.list())) dev.off()

# Note: This file takes around 5 minutes to run

###########################################################
#----------------------- IMPORTS -------------------------#
###########################################################

library(tidyverse)
library(janitor)
library(glmnet)
library(MASS)
library(lubridate)
library(quantmod)
library(caret)
library(readxl)
library(TTR)
library(dplyr)
library(ggplot2)
library(xts)
library(ranger)
library(randomForest)
library(fastshap)
library(shapviz)
library(here)
source(here("function", "Functions_main.R"))

###########################################################
#----------------- Data pre-processing -------------------#
###########################################################

# Import data
features <- read_excel(here("data", "All_Features.xlsx"))
features$Date <- as.Date(features$Date)
df <- features

# For variables that follow a log-normal distribution, transform to log values
log_cols = c('pc_oi', 'pc_vol', 'otm_pc_oi_st', 
             'otm_pc_vol_st', 'otm_pc_oi_lt', 
             'otm_pc_vol_lt', 'otm_put_oi_ma', 
             'otm_call_oi_ma', 'otm_put_vol_ma', 
             'otm_call_vol_ma')
df[log_cols] <- lapply(df[log_cols], log)

# Add target variables (SPY returns and SPY direction) to dataset
spy <- getSymbols("SPY", auto.assign = FALSE, from = min(df$Date), to = max(df$Date))
spy_df <- data.frame(
  Date = as.Date(index(spy)),
  spy_return = dplyr::lead(as.numeric(dailyReturn(Ad(spy), type = "arithmetic")), 1),
  spy_direction = 0)
spy_df[!is.na(spy_df$spy_return) & spy_df$spy_return > 0, "spy_direction"] <- 1

# Add technical indicators
spy_ind <- data.frame(
  Date       = as.Date(index(spy)),
  ATR        = dplyr::lag(f_ATR(spy)),
  ADX        = dplyr::lag(f_ADX(spy)),
  Aroon      = dplyr::lag(f_Aroon(spy)),
  BB         = dplyr::lag(f_BB(spy)),
  ChaikinVol = dplyr::lag(f_ChaikinVol(spy)),
  CLV        = dplyr::lag(f_CLV(spy)),
  EMV        = dplyr::lag(f_EMV(spy)),
  MACD       = dplyr::lag(f_MACD(spy)),
  MFI        = dplyr::lag(f_MFI(spy)),
  SAR        = dplyr::lag(f_SAR(spy)),
  SMI        = dplyr::lag(f_SMI(spy)),
  Volat      = dplyr::lag(f_Volat(spy))
)

# Merge with main data
spy_features <- spy_df %>%
  left_join(spy_ind, by = "Date") %>%
  filter(complete.cases(.))

df <- spy_features %>%
  left_join(df, by = "Date") %>%
  filter(complete.cases(.))

# Import SPY holdings
spy_holdings <- read_excel(here("Data", "spy_holdings.xlsx"))
spy_tickers <- spy_holdings$Symbol
stock_rets_xts <- xts(order.by = as.Date(index(spy))) 

# calculate returns of SPY constituents
for (ticker in spy_tickers) {
  # Replaces all "." with "-" for Yahoo Finance (BRK.B -> BRK-B)
  ticker_fmt <- gsub("\\.", "-", ticker)
  tryCatch({
    ticker_tmp <- getSymbols(ticker_fmt, auto.assign = FALSE, from = min(df$Date), to = max(df$Date))
    # Make sure stock was traded on first observation date
    if (min(df$Date) == min(as.Date(index(ticker_tmp)))) {
      ticker_rets <- dailyReturn(Ad(ticker_tmp), type = "arithmetic")
      colnames(ticker_rets) <- ticker_fmt
      stock_rets_xts <- merge.xts(stock_rets_xts, ticker_rets)
    }
  })
}

# Stock returns as dataframe
stock_rets <- data.frame(
  Date = index(stock_rets_xts),
  stock_rets_xts
)

rownames(stock_rets) <- stock_rets$Date
stock_rets$Date <- NULL
spy_tickers <- colnames(stock_rets)

# Add SPY returns to Dataframe
stock_rets["SPY"] <- dailyReturn(Ad(spy), type = "arithmetic")
stock_rets <- stock_rets[-1, ]
stock_rets <- na.omit(stock_rets) #remove NA

### Estimate beta for SPY constituents
window <- 180
betas <- as.data.frame(
  matrix(0, nrow = nrow(stock_rets), ncol = ncol(stock_rets),
         dimnames = list(rownames(stock_rets), names(stock_rets)))
)

# Compute beta as B_i = Cov(r_i, r_m) / Var(r_m) where r_m is the returns of the SPY
for (ticker in spy_tickers) {
  ticker_rets <- stock_rets[ticker]
  cov_tciker_spy <- TTR::runCov(ticker_rets, stock_rets["SPY"], n = window)
  var_spy  <- TTR::runVar(stock_rets["SPY"], n = window)
  beta_tmp <- cov_tciker_spy / var_spy
  betas[ticker] <- beta_tmp
}

# Back propagate betas for first observations
betas[, spy_tickers] <- lapply(betas[, spy_tickers], function(x) { na.locf(x, fromLast = TRUE) })
betas <- betas %>% rownames_to_column("Date")
betas <- betas[, -which(names(betas) == "SPY")]

# Beta buckets
n_buckets <- 10
beta_buckets <- betas %>%
  pivot_longer(
    cols = -Date,
    names_to = "ticker",
    values_to = "beta"
  ) %>%
  group_by(Date) %>%
  mutate(bucket = ntile(beta, n_buckets)) %>%
  ungroup()

# -----------------------------------------------------------------------------
###### Try with clustering buckets instead of n-quantiles buckets (not used in final strategy)
# n_buckets <- 31
# # Pivot betas to long format
# beta_long <- betas %>%
#   pivot_longer(
#     cols = -Date,
#     names_to = "ticker",
#     values_to = "beta"
#   )
# 
# # Import custom cluster assignments
# custom_clusters <- read_excel(here("Data", "clusters_kmeans_static_k_30.xlsx")) %>% 
#   janitor::clean_names() %>%
#   rename(ticker = ticker, cluster = cluster) %>%
#   mutate(cluster = as.integer(cluster))
# 
# # Merge with betas and rename for consistency
# beta_buckets <- beta_long %>%
#   left_join(custom_clusters, by = "ticker") %>%
#   rename(bucket = cluster)
# -----------------------------------------------------------------------------

# Returns by buckets
stock_rets <- stock_rets %>% rownames_to_column("Date")
stock_rets <- stock_rets[, -which(names(stock_rets) == "SPY")]

# Pivot stock returns to long format
stock_rets_long <- stock_rets %>%
  pivot_longer(
    cols = -Date,
    names_to = "ticker",
    values_to = "daily_return"
  )

# Calculate average return (equally weighted)
bucket_rets <- beta_buckets %>%
  left_join(stock_rets_long, by = c("Date", "ticker")) %>%
  group_by(Date, bucket) %>%
  summarise(avg_daily_return = mean(daily_return, na.rm = TRUE), .groups = "drop") %>%
  arrange(Date, bucket)

# Align to trade *next day* after the signal (no look-ahead)
bucket_rets <- bucket_rets %>%
  group_by(bucket) %>%
  arrange(Date, .by_group = TRUE) %>%
  mutate(next_ret = dplyr::lead(avg_daily_return, 1)) %>%
  ungroup()
bucket_rets$Date <- as.Date(bucket_rets$Date)


###########################################################
#------------------ SHAP Values Analysis -----------------#
###########################################################

# Train RF model on all available data
predictors <- setdiff(names(df), c("spy_return", "Date", "spy_direction"))
formula_full <- reformulate(predictors, response = "spy_return")

rf_final <- train(
  formula_full,
  data = df,
  method = "ranger",
  trControl = trainControl(method = "none"),
  tuneGrid = expand.grid(
    mtry = floor(sqrt(ncol(df) - 2)),
    splitrule = "variance",
    min.node.size = 3
  ),
  importance = "permutation"
)

# Compute SHAP values
X_shap <- df %>% dplyr::select(all_of(predictors))
shap_values <- fastshap::explain(
  object = rf_final$finalModel,
  feature_names = names(X_shap),
  X = X_shap,
  pred_wrapper = function(object, newdata) predict(rf_final, newdata)
)

shap_summary <- data.frame(
  Feature = names(X_shap),
  MeanAbsSHAP = apply(abs(shap_values), 2, mean)
) %>%
  arrange(desc(MeanAbsSHAP))

# Aggregate features by bucket
buckets <- list(
  Indicators = c("atr", "ADX", "oscillator", "pctB", "Delt.1.arithmetic", 
                 "EMA", "maEMV", "signal", "mfi", "sar", "SMI", "Volat"),
  GARCH = c("is_sigma", "is_var", "is_vol", "rolled_forecast_vol_1d_annualized", 
            "rolled_alpha", "rolled_beta", "rolled_vol_persistence"),
  Sentiment = c("Average_of_Sentiment_score", "Max_of_Sentiment_deviation", 
                "Average_of_Direction"),
  Options = c("atm_iv_slope_st", "atm_iv_slope_lt",
              "iv_otm_skew_st", "iv_otm_skew_lt",
              "pc_oi", "pc_vol",
              "otm_pc_oi_st", "otm_pc_vol_st",
              "otm_pc_oi_lt", "otm_pc_vol_lt",
              "otm_put_oi_pct_chg", "otm_call_oi_pct_chg",
              "otm_put_vol_pct_chg", "otm_call_vol_pct_chg",
              "otm_put_oi_ma", "otm_call_oi_ma",
              "otm_put_vol_ma", "otm_call_vol_ma"),
  Calendar = c("Consumer_and_Retail", "Energy_and_Commodities",
               "Housing_and_Construction", "Inflation_and_Prices",
               "Labor_Market", "Manufacturing_and_Industry",
               "Monetary_Policy", "Other",
               "Trade_and_Current_Account", "Holiday"))

bucket_map <- tibble(
  Feature = unlist(buckets),
  Bucket = rep(names(buckets), lengths(buckets)))

# Aggregate feature-level SHAP into bucket-level
shap_bucket_summary <- shap_summary %>%
  left_join(bucket_map, by = "Feature") %>%
  group_by(Bucket) %>%
  summarise(MeanAbsSHAP = sum(MeanAbsSHAP, na.rm = TRUE)) %>%
  arrange(desc(MeanAbsSHAP))

# Display results
sv <- shapviz::shapviz(shap_values, X_shap)

# Top features
sv_importance(sv, kind = "bar", max_display = 20)

# Top buckets
Shap_values_top_buckets <- ggplot(shap_bucket_summary, aes(x = reorder(Bucket, MeanAbsSHAP), y = MeanAbsSHAP)) +
  geom_bar(stat = "identity", fill = "#2C7BB6") +
  coord_flip() +
  labs(title = "Bucket-level SHAP Importance (Random Forest)",
       x = "Feature Bucket", y = "Sum(|SHAP|)") +
  theme_minimal(base_size = 13)
print(Shap_values_top_buckets)

# Save to output folder
ggsave(
  filename = here("output", "Shap_values_top_buckets.png"),
  plot = Shap_values_top_buckets,
  width = 10, height = 6, dpi = 300 
)


###########################################################
#-------------------- Model Testing ----------------------#
###########################################################

names(df) <- make.names(names(df), unique = TRUE)
set.seed(1234)

# data for the regression models
x_all <- model.matrix(spy_return ~ . - Date - spy_direction, data = df)[, -1]
y_all <- df$spy_return

# data for the classification models
x_all_class <- model.matrix(spy_direction ~ . - Date - spy_return, data = df)[, -1]
y_all_class <- df$spy_direction

test_years <- sort(unique(year(df$Date)))
test_years <- test_years[test_years >= 2020]
min_train_rows <- 30
models <- c("OLS", "stepwise", "lasso", "relaxed_lasso", "ridge", "bagging", "random_forest", "logit", "rf_class")
pred_store <- list()

for (yr in test_years) {
    train_end <- as.Date(sprintf("%d-12-31", yr - 1))
    test_range <- df$Date >= as.Date(sprintf("%d-01-01", yr)) & df$Date <= as.Date(sprintf("%d-12-31", yr))
    train_idx <- which(df$Date <= train_end)
    test_idx <- which(test_range)
    if (length(train_idx) < min_train_rows || length(test_idx) == 0) next
    
    x_train <- x_all[train_idx, ]
    y_train <- y_all[train_idx]
    x_test <- x_all[test_idx, ]
    y_test <- y_all[test_idx]
    
    x_train_class <- x_all_class[train_idx, ]
    y_train_class <- y_all_class[train_idx]
    x_test_class <- x_all_class[test_idx, ]
    y_test_class <- y_all_class[test_idx]
    
    # Columns to remove from the set
    reg_cols <- c(which(names(df) == "Date"), which(names(df) == "spy_direction"))
    class_cols <- c(which(names(df) == "Date"), which(names(df) == "spy_return"))
    
    df_train <- df[train_idx, -reg_cols]
    df_test <- df[test_idx, -reg_cols]
    
    df_train_class <- df[train_idx, -class_cols]
    df_test_class <- df[test_idx, -class_cols]
    
    # Set SPY direction as up/down factors
    df_train_class$spy_direction <- factor(df_train_class$spy_direction,
                                           levels = c(0, 1),
                                           labels = c("down", "up"))
    df_test_class$spy_direction  <- factor(df_test_class$spy_direction,
                                           levels = c(0, 1),
                                           labels = c("down", "up"))
    
    df_train_class$spy_direction <- relevel(df_train_class$spy_direction, ref = "down")
    df_test_class$spy_direction  <- relevel(df_test_class$spy_direction,  ref = "down")
    
    # formula
    predictors <- setdiff(names(df), c("spy_return", "spy_direction", "Date"))
    formula_full <- reformulate(predictors, response = "spy_return")
    formula_full_class <- reformulate(predictors, response = "spy_direction")
    
    #         OLS          #
    #----------------------#
    p_ols <- predict(lm(formula_full, data = df_train), df_test)
  
    #       Stepwise       #
    #----------------------#
    step_fit <- suppressWarnings(
      stepAIC(lm(spy_return ~ 1, data = df_train),
              scope = list(lower = ~1, upper = formula_full),
              direction = "both", trace = FALSE)
    )
    p_step <- predict(step_fit, df_test)
    
    #        LASSO         #
    #----------------------#
    lasso_fit <- cv.glmnet(x_train, y_train, alpha = 1, family = "gaussian")
    p_lasso <- predict(lasso_fit, newx = x_test, s = "lambda.min")
  
    #    Relaxed LASSO     #
    #----------------------#
    relax_fit <- cv.glmnet(x_train, y_train, alpha = 1, family = "gaussian", relax = TRUE)
    p_relax <- predict(relax_fit, newx = x_test, s = "lambda.min", gamma = "gamma.min")
    
    #        Ridge         #
    #----------------------#
    ridge_fit <- cv.glmnet(x_train, y_train, alpha = 0, family = "gaussian")
    p_ridge <- predict(ridge_fit, newx = x_test, s = "lambda.min")
    
    #       Bagging        #
    #----------------------#
    # Bagging = Random Forest with mtry = number of predictors
    bag_grid <- expand.grid(
      mtry = ncol(x_train),          # all features at each split
      splitrule = "variance", 
      min.node.size = 3
    )
    
    bag_fit <- train(
      formula_full,
      data = df_train,
      method = "ranger",
      trControl = trainControl(method = "none"),
      tuneGrid = bag_grid
    )
    p_bag <- predict(bag_fit, df_test)
    
    #    Random Forest     #
    #----------------------#
    # Typical choice: mtry = sqrt(p)
    rf_grid <- expand.grid(
      mtry = floor(sqrt(ncol(x_train))), 
      splitrule = "variance", 
      min.node.size = 3
    )
    
    rf_fit <- train(
      formula_full,
      data = df_train,
      method = "ranger",
      trControl = trainControl(method = "none"),
      tuneGrid = rf_grid,
      importance = "permutation"
    )
    p_rf <- predict(rf_fit, df_test)
    
    #        Logit         #
    #----------------------#
    logit_fit <- glm(
      formula = formula_full_class,
      data = df_train_class,
      family = binomial(link = "logit"))
    p_logit <- predict(logit_fit, newdata = df_test_class, type = "response")
    
    #    Random Forest     #
    #----------------------#
    rf_grid_class <- expand.grid(
      mtry = floor(sqrt(ncol(x_train_class))),
      splitrule = "gini",                            
      min.node.size = 1                            
    )
    
    rf_fit_class <- train(
      formula_full_class,
      data = df_train_class,
      method = "ranger",
      trControl  = trainControl(method = "none", classProbs = TRUE),
      tuneGrid   = rf_grid_class,
      importance = "permutation",
    )
    p_rf_class <- predict(rf_fit_class, df_test_class, type="prob")[, "up"]
    
    # Save results
    pred_store[[as.character(yr)]] <- tibble(
      date = df$Date[test_idx],
      y = y_test,
      y_class = y_test_class,
      OLS = p_ols,
      stepwise = p_step,
      lasso = as.numeric(p_lasso),
      relaxed_lasso = as.numeric(p_relax),
      ridge = as.numeric(p_ridge),
      bagging = as.numeric(p_bag),
      random_forest = as.numeric(p_rf),
      logit = as.numeric(p_logit),
      rf_class = as.numeric(p_rf_class)
  )
}

# Combine predictions
preds <- bind_rows(pred_store) %>% arrange(date)
preds$date <- as.Date(preds$date)

# Compute RMSE and R^2 for all models
quality_fit <- data.frame(
  RMSE = sapply(head(models, -2), function(m) sqrt(mean((preds[[m]] - preds$y)^2, na.rm = TRUE))),
  R2   = sapply(head(models, -2), function(m) cor(preds[[m]], preds$y, use = "complete.obs")^2)
)

# Convert prob of up movement to 1/0 values
threshold <- 0.5
preds$logit <- as.integer(preds$logit >= threshold)
preds$rf_class <- as.integer(preds$rf_class >= threshold)

confusion_logit <- confusionMatrix(
  data = as.factor(preds$logit),
  reference = as.factor(preds$y_class))

confusion_rf <- confusionMatrix(
  data = as.factor(preds$rf_class),
  reference = as.factor(preds$y_class))

# Beta strat
preds_beta <- preds %>%
  pivot_longer(cols = all_of(models), names_to = "model", values_to = "pred") %>%
  mutate(chosen_bucket = dplyr::case_when(
    pred > 0 ~ n_buckets,  # long high beta
    TRUE ~ 1L              # otherwise long low beta
  ))

# Join the next-day bucket return to the signal date
strat_beta <- preds_beta %>%
  left_join(bucket_rets,
            by = c("date" = "Date", "chosen_bucket" = "bucket")) %>%
  transmute(date = date,
            model,
            chosen_bucket,
            strat_ret = next_ret)

# Replace missing with 0 (no position / missing data)
strat_beta <- strat_beta %>%
  mutate(strat_ret = ifelse(is.na(strat_ret), 0, strat_ret))

# Daily equity curves
long_only_daily <- strat_beta %>%
  group_by(model) %>%
  arrange(date, .by_group = TRUE) %>%
  mutate(cum_growth = cumprod(1 + strat_ret)) %>%
  ungroup()

# Weekly aggregation
weekly_curves <- strat_beta %>%
  mutate(week = lubridate::floor_date(date, "week")) %>%
  group_by(model, week) %>%
  summarise(weekly_ret = prod(1 + strat_ret) - 1, .groups = "drop") %>%
  group_by(model) %>%
  arrange(week, .by_group = TRUE) %>%
  mutate(cum_growth = cumprod(1 + weekly_ret)) %>%
  ungroup()

# Buy-and-hold strategy (baseline)
buy_hold <- preds %>%
  mutate(week = floor_date(date, "week")) %>%
  group_by(week) %>%
  summarise(weekly_ret = prod(1 + y) - 1, .groups = "drop") %>%
  arrange(week) %>%
  mutate(cum_growth = cumprod(1 + weekly_ret),
         model = "buy_hold")

# Combine all curves
curves <- dplyr::bind_rows(
  weekly_curves %>% dplyr::select(week, model, cum_growth) %>% rename(date = week),
  buy_hold %>% dplyr::select(week, model, cum_growth) %>% rename(date = week)
)


###########################################################
#---------- Weighted Portfolio of Strategies -------------#
###########################################################

# compute all measures (sharpe, volatility, returns)
metrics <- weekly_curves %>%
  group_by(model) %>%
  summarise(
    sharpe = sharpe_ratio(weekly_ret),
    vol = sd(weekly_ret, na.rm = TRUE),
    avg_return = mean(weekly_ret, na.rm = TRUE),
    .groups = "drop"
  )

# Add buy-and-hold measures
buy_hold_metrics <- tibble(
  model = "buy_hold",
  sharpe = sharpe_ratio(buy_hold$weekly_ret),
  vol = sd(buy_hold$weekly_ret, na.rm = TRUE),
  avg_return = mean(buy_hold$weekly_ret, na.rm = TRUE)
)
metrics <- bind_rows(metrics, buy_hold_metrics)

# Compute "eligible" models and weights for each strategy (this gives the metrics values and the assigned weights)
eligible_sharpe <- compute_weights(metrics, buy_hold_metrics$sharpe, "sharpe", `>`)
eligible_vol    <- compute_weights(metrics, buy_hold_metrics$vol, "vol", `<`)
eligible_ret    <- compute_weights(metrics, buy_hold_metrics$avg_return, "avg_return", `>`)


###########################################################
#----- Build strategy returns and plot equity curves -----#
###########################################################

# calculate returns for the 3 different weighted strategies portfolios
max_sharpe <- make_weighted_curve(eligible_sharpe, "Max Sharpe")
min_vol    <- make_weighted_curve(eligible_vol, "Min Vol")
weighted_alpha    <- make_weighted_curve(eligible_ret, "Weighted Alpha")
curves_final <- bind_rows(curves, max_sharpe, min_vol, weighted_alpha)

# Plot All models cumulative growth over the testing period
all_equitycurves <- ggplot(curves_final, aes(x = date, y = cum_growth, color = model)) +
  geom_line(linewidth = 0.9) +
  scale_x_date(
    date_labels = "%b-%Y",   # Show month-year (e.g., Jan-2021)
    date_breaks = "1 months" # Only one tick every 3 months
  ) +
  labs(
    title = "All Equity Curves",
    y = "Cumulative Growth", x = NULL, color = "Model"
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))
print(all_equitycurves)

# Plot the final strategies equity curves
selected_models <- c("Weighted Alpha", "Min Vol", "Max Sharpe", "buy_hold")
curves_subset <- curves_final %>%
  filter(model %in% selected_models)
final_equitycurves <- ggplot(curves_subset, aes(x = date, y = cum_growth, color = model)) +
  geom_line(linewidth = 1.1) +
  scale_x_date(
    date_labels = "%b-%Y",   # Show month-year (e.g., Jan-2021)
    date_breaks = "1 months" # Only one tick every 3 months
  ) +
  labs(
    title = "Equity curves: Weighted Strategies vs Buy and Hold",
    y = "Cumulative Growth",
    x = NULL,
    color = "Strategy"
  ) +
  theme_minimal(base_size = 13) +
  theme(
    legend.position = "bottom",
    legend.title = element_text(face = "bold"),
    plot.title = element_text(face = "bold", size = 15),
    axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)
  )
print(final_equitycurves)

# save plots in output folder
ggsave(
  filename = here("output", "all_equity_curves.png"),
  plot = all_equitycurves,
  width = 10, height = 6, dpi = 300
)

ggsave(
  filename = here("output", "Final_equity_curves.png"),
  plot = final_equitycurves,
  width = 10, height = 6, dpi = 300
)

