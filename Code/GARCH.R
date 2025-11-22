## ================================================== ##
## Rolled Fit GARCH(1,1) for SPY (2015–2023)


libs <- c("quantmod", "rugarch", "xts", "tibble")
invisible(lapply(libs, function(x) if (!requireNamespace(x, quietly = TRUE)) install.packages(x)))
invisible(lapply(libs, library, character.only = TRUE, quietly = TRUE))

## Data: SPY close price, daily log rets (2015-01-01 - 2023-12-31)
from <- "2015-01-01"; to <- "2023-12-31"
spy       <- suppressWarnings(getSymbols("SPY", src = "yahoo", from = from, to = to, auto.assign = FALSE))
spy_close <- Cl(spy)
ret       <- na.omit(diff(log(spy_close)))  # daily log-returns

## Fit rolled GARCH (1,1)
#rugarch (enforces GARCH constraints (positivity, stationarity/mean rev)) 

## GARCH(1,1) spec
spec <- ugarchspec(
  variance.model     = list(model = "sGARCH", garchOrder = c(1, 1)), #sGARCH alpha + Beta < 1 (use iGARCH to allow unit root vol)
  mean.model         = list(armaOrder = c(0, 0), include.mean = FALSE),
  distribution.model = "norm"
)

##  in-sample vol features
is_fit   <- ugarchfit(spec = spec, data = ret)
is_sigma <- sigma(is_fit)              # daily conditional sd
is_var   <- is_sigma^2                 # daily conditional variance
is_vol_annualized  <- is_sigma * sqrt(252)       # annualized conditional vol


## Rolling 1-step-ahead forecast (annualized) with refits
initial_window <- 500   # approx 2 years 
refit.every    <- 22    # refit / month environ

roll <- ugarchroll(
  spec, data = ret, n.ahead = 1,
  forecast.length = NROW(ret) - initial_window,
  refit.every     = refit.every,
  refit.window    = "moving",
  solver          = "hybrid",
  calculate.VaR   = FALSE
)

roll_df <- as.data.frame(roll)  
fc_sigma_1d <- xts(roll_df$Sigma, order.by = as.Date(rownames(roll_df)))
rolled_forecast_vol_1d_annualized <- fc_sigma_1d * sqrt(252)

# Build blocks (rolled) params for forecasts
library(zoo)  # for na.locf

ret_dates <- as.Date(index(ret))
fc_dates  <- as.Date(index(rolled_forecast_vol_1d_annualized))

rolled_alpha_xts <- xts::xts(rep(NA_real_, length(fc_dates)), order.by = fc_dates)
rolled_beta_xts  <- xts::xts(rep(NA_real_, length(fc_dates)), order.by = fc_dates)

nR <- NROW(ret)
refit_points <- seq(from = initial_window, to = nR - 1, by = refit.every)

assign_count <- 0L

for (rp in refit_points) {
  est_start <- rp - initial_window + 1
  est_end   <- rp
  est_win   <- ret[est_start:est_end]
  
  # refit 
  fit_roll <- try(
    ugarchfit(
      spec = spec, data = est_win,
      solver = "hybrid",
      fit.control = list(scale = TRUE),
      solver.control = list(trace = 0)
    ),
    silent = TRUE
  )
  if (inherits(fit_roll, "try-error") || rugarch::convergence(fit_roll) != 0) {
    a_last <- suppressWarnings(as.numeric(tail(na.omit(rolled_alpha_xts), 1)))
    b_last <- suppressWarnings(as.numeric(tail(na.omit(rolled_beta_xts), 1)))
    a <- if (length(a_last)) a_last else 0.05
    b <- if (length(b_last)) b_last else 0.90
  } else {
    ca <- coef(fit_roll)
    a  <- unname(ca["alpha1"])
    b  <- unname(ca["beta1"])
  }
  
  # dates covered by this refit
  fc_start_idx <- rp + 1
  fc_end_idx   <- min(rp + refit.every, nR - 1)
  if (fc_start_idx > fc_end_idx) next
  
  block_ret_dates <- ret_dates[fc_start_idx:fc_end_idx]
  pos <- which(index(rolled_alpha_xts) %in% block_ret_dates)
  
  if (length(pos)) {
    rolled_alpha_xts[pos] <- a
    rolled_beta_xts[pos]  <- b
    assign_count <- assign_count + length(pos)
  }
}
cat("Assigned rolled params to", assign_count, "forecast dates\n")

# carry forward between refits (remove NAs)
rolled_alpha_xts <- na.locf(rolled_alpha_xts, na.rm = FALSE)
rolled_beta_xts  <- na.locf(rolled_beta_xts,  na.rm = FALSE)

# Vol persistence
rolled_vol_persistence_xts <- rolled_alpha_xts + rolled_beta_xts




#### Construct Feature table ####
features_xts <- Reduce(
  function(x, y) merge(x, y, join = "inner"),
  list(
    is_sigma,
    is_var,
    is_vol_annualized,
    rolled_forecast_vol_1d_annualized,
    rolled_alpha_xts,
    rolled_beta_xts,
    rolled_vol_persistence_xts
  )
)

colnames(features_xts) <- c(
  "is_sigma",
  "is_var",
  "is_vol",
  "rolled_forecast_vol_1d_annualized",
  "rolled_alpha",
  "rolled_beta",
  "rolled_vol_persistence"
)

features <- tibble::tibble(
  date                              = as.Date(index(features_xts)),
  is_sigma                          = as.numeric(features_xts$is_sigma),
  is_var                            = as.numeric(features_xts$is_var),
  is_vol                            = as.numeric(features_xts$is_vol),
  rolled_forecast_vol_1d_annualized = as.numeric(features_xts$rolled_forecast_vol_1d_annualized),
  rolled_alpha                      = as.numeric(features_xts$rolled_alpha),
  rolled_beta                       = as.numeric(features_xts$rolled_beta),
  rolled_vol_persistence            = as.numeric(features_xts$rolled_vol_persistence)
)

print(head(features, 8))


