### Imports
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

### Technical Indicators
f_ATR        <- function(x) {ATR(HLC(x))[, "atr"]}
f_ADX        <- function(x) {ADX(HLC(x))[, "ADX"]}
f_Aroon      <- function(x) {aroon(cbind(Hi(x), Lo(x)), n = 2)$oscillator}
f_BB         <- function(x) {BBands(HLC(x))[, "pctB"]}
f_ChaikinVol <- function(x) {Delt(chaikinVolatility(cbind(Hi(x), Lo(x))))[, 1]}
f_CLV        <- function(x) {EMA(CLV(HLC(x)))[, 1]}
f_EMV        <- function(x) {EMV(cbind(Hi(x), Lo(x)), Vo(x))[, 2]}
f_MACD       <- function(x) {MACD(Cl(x))[, 2]}
f_MFI        <- function(x) {MFI(HLC(x), Vo(x))}
f_SAR        <- function(x) {SAR(cbind(Hi(x), Cl(x))) [, 1]}
f_SMI        <- function(x) {SMI(HLC(x))[, "SMI"]}
f_Volat      <- function(x) {volatility(OHLC(x), calc = "garman")[, 1]}

### Compute Sharpe Ratio
sharpe_ratio <- function(returns, freq = 52) {
  mean(returns, na.rm = TRUE) / sd(returns, na.rm = TRUE) * sqrt(freq)
}

### Calculate and normalize the weights for the finalized strategies
compute_weights <- function(metric_df, buy_hold_metric, metric_col, comparison, normalize = TRUE) {
  eligible <- metric_df %>%
    filter(model != "buy_hold") %>%
    filter({{comparison}}(.data[[metric_col]], buy_hold_metric))
  
  if (normalize && nrow(eligible) > 0) {
    eligible <- eligible %>%
      mutate(weight = .data[[metric_col]] / sum(.data[[metric_col]], na.rm = TRUE))
  }
  eligible
}

### Calculate the cumulative growth for the final strategies
make_weighted_curve <- function(eligible_tbl, label) {
  if (nrow(eligible_tbl) == 0) return(NULL)
  wide <- weekly_curves %>%
    dplyr::select(model, week, weekly_ret) %>%
    pivot_wider(names_from = model, values_from = weekly_ret)
  wide$weighted_ret <- rowSums(
    sapply(eligible_tbl$model, function(m)
      wide[[m]] * eligible_tbl$weight[eligible_tbl$model == m]),
    na.rm = TRUE
  )
  tibble(
    date = wide$week,
    model = label,
    cum_growth = cumprod(1 + wide$weighted_ret)
  )
}
