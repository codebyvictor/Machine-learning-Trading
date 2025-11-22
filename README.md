# Project 1: Long Beta Trading Strategy 

In a fast-evolving market environment, our strategy leverages advanced quantitative tools to identify return
opportunities while controlling risk. We combine machine learning with economically motivated signals—U.S.
macroeconomic calendar surprises, option-implied information (level, term, smile, volume, open interest), Financial Times trade/tariff news sentiment, and technical indicators on SPY and VIX—augmented by volatility-regime features (including rolled GARCH(1,1) parameters and persistence).

The objective is to improve risk-adjusted performance relative to a buy-and-hold exposure to the S&P 500
(SPY). Rather than investing in the SPY directly, we opted to expand our framework to include its constituents. 
We see this as a way to integrate higher levels of volatility into our strategy, and therefore capture higher upside potential.

We implement a walk-forward design with a 5-year rolling training window and a 1-year test window
(date debut - date fin coverage). Daily predictions are translated into long/short SPY signals and reported
at least at a monthly frequency. Transparency is ensured through clear documentation of feature enginee-
ring (including revision-aware handling of macro releases and fixed-maturity interpolation for 3-month vola-
tility), model selection (OLS, stepwise AIC, LASSO/Relaxed LASSO, Ridge, bagging, Random Forest ; plus
classification variants), and evaluation (out-of-sample R2, RMSE, annualized return/volatility, Sharpe, max
drawdown).

## Prerequisites

For this project to run successfully on your machine you need to have R installed as well as the following packages in R:

- `tidyverse`
- `janitor`
- `glmnet`
- `MASS`
- `lubridate`
- `quantmod`
- `caret`
- `readxl`
- `TTR`
- `dplyr`
- `ggplot2`
- `xts`
- `ranger`
- `randomForest`
- `fastshap`
- `shapviz`

## R Files Description
The only needed file for the project to run is main.r
- `run_strategy.R`: This is the main script that undertakes the overall analysis. It consists of loading the features file, computing the beta buckets, implementing machine learning models, and implementing a portfolio of weighted strategies. 

- `ClusterBetas.R`: This script groups the necessary Clustering functions. 
  - gap_statistic function: Tibshirani et al. Gap Statistic for choosing k. 
  - compute_distance_matrix: correlation-distance on beta paths.
  - make_beta_feature_matrix: builds a matrix of beta paths over a window; row-wise z-scoring; 
  - cluster_betas_static: one-shot hierarchical clustering on a recent lookback window
  - cluster_betas_rolling :rolling (weekly/monthly) hierarchical clustering on beta paths
  - cluster_betas_kmeans: static K-means on average beta level over a lookback 
  - cluster_betas_kmeans_rolling: rolling K-means on beta paths 
- inputs: betas_df 
For hierarchical modes, clustering uses correlation distance on the entire beta path (shape/co-movement).
For K-means “level” mode, clustering uses the average beta over the lookback.

- `ClusteringBacktestMain.R`: This script's main role is to run the clustering using the clustering functions with the option of selecting the clustering mode. 

1. Loads SPY + constituents, builds rolling 180-day betas
2. Selects clustering mode via CLUSTER_MODE:
  "hclust" - rolling hierarchical on beta paths (weekly/monthly rebalance).
  "kmeans" -  static K-means on average beta over the last CL_LOOKBACK_DAYS before FIRST_TEST_YEAR.
  "kmeans_rolling" -  rolling K-means on beta paths.
  "static_hclust" - one-shot hierarchical on the latest window.
3. Runs the clustering and outputs the results. 

Key parameters:
BETA_LOOKBACK_DAYS = 180 (for betas), CL_LOOKBACK_DAYS = 180 (for clustering window),
REB_FREQ = "weeks" or "months", FIRST_TEST_YEAR (anchors static K-means window end), SEED.

Outputs (written to outputs/):
 — cluster associations
 — daily avg cluster returns 

- `GARCH.R`: This script computes sGARCH(1,1) features for SPY (via rugarch).
Method: daily SPY log returns; rolling 500-day estimation; monthly refits (ugarchroll); Gaussian   innovations.
Features produced:
    -In-sample: is_sigma (sd), is_var (variance), is_vol (annualized).
    -Forecast: rolled_forecast_vol_1d_annualized (one-step-ahead annualized vol).
    -Parameters: rolled_alpha, rolled_beta, rolled_vol_persistence = alpha + beta.
Usage: outputs are merged master features table and used Main.R as predictors.

- `calendar_events_df.ipynb`: This python script builds the calendar events dataframe. We first obtained U.S. economic calendar event data from Investing.comwhich we extracted using a data-scraping Chrome add-on "Easy Scraper", allowing us to download all calendar events over our sample period. The list of events were then grouped into categories, reflecting different areas of the macroeconomic environment that may influence equity markets in distinct ways. We define surprises as the relative difference between the actual and forecasted values for each event. Using a relative measure standardizes the surprises across indicators, ensuring that the resulting values are comparable across event types and not distorted by differences in scale or magnitude.

- `run_data.py`: This python script reads the spyOptionsData.csv file and aggregates the data by days-to-expiration and moneyness bucket to compute daily options features for the SPY. 

Description: This python script retrieve options features from the US equity options market. We obtained daily SPY options data from OptionMetrics IvyDB. The sample was filtered to ensure liquidity and we kept contracts within well-defined days-to-expiration buckets (DTE) with a focus on ATM and OTM strikes. Our set of options features capture three dimensions. First, the volatility term structure and smile. The slope of the implied volatility curve is measured both in the short term (41-60 DTE vs 30-40 DTE) and long term (61-90 DTE vs 30-40 DTE). Similarly, skew is defined as the spread between OTM puts and OTM calls for both short- and long-term maturities. These measures reflect how investors price downside risk. Second, we estimate positioning and sentiment with put-call ratios based on open interest and volume. An elevated put activity relative to calls often reflects a higher demand for protection against downside risk, while a spike in call activity may indicate speculative risk taking from investors. Finally, to account for changes in market positioning, we also include percentage changes in OI and volume for OTM puts and calls. We also compute the ratio of OI and volume for OTM puts and calls to their 10-day simple moving average to capture relative deviations from recent activity, allowing us to highlight unusual shifts in option demand.

1. Load spyOptionsData.csv and import SPY prices from the web (Stooq).
2. Aggregate daily options data by moneyness ['deep_itm', 'itm', 'atm', 'otm', 'deep_otm'], days-to-expiration ['30_40', '41_60', '61_90'], and Call/Put flag, resulting in 30 buckets.
3. Compute features and output the results

- Options Features produced:
  - atm_iv_slope_st:	[41 to 60 DTE ATM Call Options IV] - [30 to 40 DTE ATM Call Options IV]
  - atm_iv_slope_lt:	[61 to 90 DTE ATM Call Options IV] - [30 to 40 DTE ATM Call Options IV]
  - iv_otm_skew_st:	[30 to 40 DTE OTM Put Options IV] - [30 to 40 DTE OTM Call Options IV]
  - iv_otm_skew_lt:	[61 to 90 DTE OTM Put Options IV] - [61 to 90 DTE OTM Call Options IV]
  - pc_oi:	[Put Options Total OI] / [Call Options Total OI]
  - pc_vol:	[Put Options Total Volume] / [Call Options Total Volume]
  - otm_pc_oi_st:	[30 to 40 DTE OTM Put Options OI] / [30 to 40 DTE OTM Call Options OI]
  - otm_pc_vol_st:	[30 to 40 DTE OTM Put Options Volume] / [30 to 40 DTE OTM Call Options Volume]
  - otm_pc_oi_lt:	[61 to 90 DTE OTM Put Options OI] / [61 to 90 DTE OTM Call Options OI]
  - otm_pc_vol_lt:	[61 to 90 DTE OTM Put Options Volume] / [61 to 90 DTE OTM Call Options Volume]
  - otm_put_oi_pct_chg:	% Change for [OTM Put Options OI]
  - otm_call_oi_pct_chg:	% Change for [OTM Call Options OI]
  - otm_put_vol_pct_chg: % Change for [OTM Put Options Volume]
  - otm_call_vol_pct_chg:	% Change for [OTM Call Options Volume]
  - otm_put_oi_ma:	[OTM Put Options OI] / 10d SMA of [OTM Put Options OI]
  - otm_call_oi_ma:	[OTM Call Options OI] / 10d SMA of [OTM Call Options OI]
  - otm_put_vol_ma:	[OTM Put Options Volume] / 10d SMA of [OTM Put Options Volume]
  - otm_call_vol_ma:	[OTM Call Options Volume] / 10d SMA of [OTM Call Options Volume]

  - Outputs: spy_options_features.csv

- `run_install_packages.py`: File to install all packages required to run the r scripts.

## Data Files Description
- `All_Features.xlsx`: File containing all features (options, macroeconomic indicators, GARCH parameters, tariffs news sentiment)

- `Clusters_kmeans_static_k_10.xlsx`: Output of the CltereringBacktestMain.R file that classifies all components of the spy into 10 clusters using k-means method with k=10.

- `spy_holdings.xlsx`: File containing all weights and prices at the end of the period for all components of the SPY. 

- `Tariff_news_Data.xlsx`: File containing the raw data for the Financial Times tariffs article, with title, description, link, sentiment score, direction. We scrape around 10,000 articles tagged to trade/tariffs from 2015 to 2025 and score sentiment with an LLM prompt constrained to output a 1--10 score (temperature~0 for determinism). The ChatGPT-5 prompt is defined as follows: "You are a financial sentiment analyzer. Your task is to read the following news article and evaluate its likely impact on U.S. large-cap public equities. In your answer, only provide a sentiment score from 1 to 10, where: 1 = extremely negative impact; 10 = extremely positive impact; 5 = neutral / no significant impact". All articles with value of 5 are omitted from our database to reduce the noise. We then aggregate to daily by averaging the scores of all articles in one day; missing days are forward-filled conservatively. We also define a direction measure that averages the direction of all articles in a day, bounded between 0 and 1. Finally, "maximum sentiment deviation" represents the article whose sentiment score diverges most significantly from the neutral value of 5, thereby giving an indicator of the day's most impactful content. 

- `raw_calendar_data`: Folder containing all the raw economic calendar data (by year), with day, time, currency, web link, title, actual value, previous value and forecasted value

- `spyOptionsData.csv`: *This file is 300Mb and is too large to upload* File containing the raw SPY options data from OptionMetrics IvyDB. The query is on the date range 2014-08-31 to 2023-08-31 and is filtered on ticker = SPY and Open Interest > 0, with day-to-expiration between 30 and 90 days. 

- `spy_options_features.csv`: Output of the run_data.py file that compute daily features on option data for the SPY etf.




