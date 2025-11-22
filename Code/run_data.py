# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Import data

# SPY options data from 2014-08-31 to 2023-08-31 (OptionMetrics)
path_to_file = os.path.join("..", "Data", "spyOptionsData.csv") #spyOptionsData.csv is not in the repository due to its weight
rawOptions = pd.read_csv(path_to_file)

# Daily SPY from Stooq (ticker needs .us suffix)
url = "https://stooq.com/q/d/l/?s=spy.us&i=d"
spy = pd.read_csv(url, parse_dates=["Date"]).rename(columns=str.lower).dropna()

# Prices dataframe
dfPrices = spy[["date", "close"]].sort_values("date").reset_index(drop=True)
dfPrices["logRets"] = np.log(dfPrices["close"] / dfPrices["close"].shift(1))
dfPrices["date"] = pd.to_datetime(dfPrices["date"])
dfPrices = dfPrices.dropna().reset_index(drop=True)

# Options dataframe
dfOptions = rawOptions[['date', 'exdate', 'cp_flag', 'strike_price', 'best_bid', 'best_offer', 'volume', 'open_interest', 'impl_volatility']].copy()
dfOptions['date'] = pd.to_datetime(dfOptions['date'], format="%Y-%m-%d")
dfOptions['exdate'] = pd.to_datetime(dfOptions['exdate'], format="%Y-%m-%d")
dfOptions['strike_price'] = dfOptions['strike_price'] / 1000
dfOptions = dfOptions.sort_values(by="date").reset_index(drop=True)

# Calculate days-to-expiration (dte) and moneyness
# spot price
dfOptions = dfOptions.merge(dfPrices, on='date', how='left')
dfOptions = dfOptions.rename(columns={'close': 'spot_price'})
dfOptions['spot_price'] = dfOptions['spot_price'].fillna(method='ffill')

# days to expiration
dfOptions['dte'] = (dfOptions['exdate'] - dfOptions['date']).dt.days

# moneyness
dfOptions['moneyness'] = dfOptions['strike_price'] / dfOptions['spot_price']

# Create buckets by moneyness and dte
# call options moneyness buckets
bins = [0.00, 0.80, 0.95, 1.05, 1.20, np.inf]
labels = ['deep_itm', 'itm', 'atm', 'otm', 'deep_otm']
dfOptions['moneyness_bucket_call'] = pd.cut(dfOptions.loc[dfOptions['cp_flag'] == 'C', 'moneyness'], bins=bins, labels=labels, right=False)

# put options moneyness buckets
bins = [0.00, 0.80, 0.95, 1.05, 1.20, np.inf]
labels = ['deep_otm', 'otm', 'atm', 'itm', 'deep_itm']
dfOptions['moneyness_bucket_put'] = pd.cut(dfOptions.loc[dfOptions['cp_flag'] == 'P', 'moneyness'], bins=bins, labels=labels, right=False)

# merge columns into a single moneyness bucket column
dfOptions['moneyness_bucket'] = np.where(dfOptions['cp_flag'] == 'C', dfOptions['moneyness_bucket_call'], dfOptions['moneyness_bucket_put'])
dfOptions = dfOptions.drop(columns=['moneyness_bucket_call', 'moneyness_bucket_put'])

# tenor buckets
bins = [30, 41, 61, 91]
labels = ['30_40', '41_60', '61_90']
dfOptions['tenor_bucket'] = pd.cut(dfOptions['dte'], bins=bins, labels=labels, right=False)

# Format options dataframe by bucket
sortCols = ['date', 'cp_flag', 'dte', 'moneyness', 'strike_price']
orderCols = ['date', 'exdate', 'cp_flag', 'tenor_bucket', 'moneyness_bucket', 'strike_price', 'spot_price', 'dte', 'moneyness', 'volume', 'open_interest', 'impl_volatility']
dfOptions = dfOptions.sort_values(by=sortCols).reset_index(drop=True)
dfOptions = dfOptions[orderCols]

# Daily open interest and volume on options
# calls
callsDailyTotals = (
    dfOptions[dfOptions['cp_flag'] == 'C']
      .groupby("date", observed=True)
      .agg(total_vol_C=("volume", "sum"),
           total_oi_C=("open_interest", "sum"))
)

# puts
putsDailyTotals = (
    dfOptions[dfOptions['cp_flag'] == 'P']
      .groupby("date", observed=True)
      .agg(total_vol_P=("volume", "sum"),
           total_oi_P=("open_interest", "sum"))
)

# merge calls and puts
dailyTotals = callsDailyTotals.merge(putsDailyTotals, on='date', how='outer').fillna(0.0).reset_index()
dailyTotals['total_vol'] = dailyTotals['total_vol_C'] + dailyTotals['total_vol_P']
dailyTotals['total_oi'] = dailyTotals['total_oi_C'] + dailyTotals['total_oi_P']

# set date as index
dailyTotals.set_index('date', inplace=True)

# Aggregate option data by bucket
def f_safeDiv(num, den):
    return np.where(den==0 | np.isnan(den), np.nan, num/den)

def f_aggBlock(x: pd.DataFrame):
    vol = x["volume"].sum()
    oi = x["open_interest"].sum()
    # OI and volume weighted IV
    iv_oi_w = (x["impl_volatility"] * x["open_interest"]).sum()
    iv_vol_w = (x["impl_volatility"] * x["volume"]).sum()
    # output
    out = {
        "sum_volume": vol,
        "sum_oi": oi,
        "avg_iv_oi": f_safeDiv(iv_oi_w, oi),
        "avg_iv_vol": f_safeDiv(iv_vol_w, vol)
    }
    return pd.Series(out)

bucketCols = ["date", "cp_flag", "tenor_bucket", "moneyness_bucket"]

dfOptionsAgg = (
    dfOptions.groupby(bucketCols, observed=True)
      .apply(f_aggBlock)
      .reset_index()
)

dfOptionsAgg = dfOptionsAgg.merge(dailyTotals, on="date", how="left")
dfOptionsAgg = dfOptionsAgg.sort_values(bucketCols).reset_index(drop=True)

# Compute features
def f_ensureList(x):
    if x is None:
        return []                 # or [None], if you prefer
    if isinstance(x, list):
        return x                  # already a list → no change
    if isinstance(x, str):
        return [x]                # keep the whole string as one item
    try:
        return list(x)            # e.g., tuple/set/np.array/Series → list(...)
    except TypeError:
        return [x]                # non-iterables (int, float, custom) → wrap
    
def f_pick(df, moneyness, tenor, cp_flag, col):
    if (isinstance(moneyness, list) or isinstance(tenor, list) or isinstance(cp_flag, list)):
        moneyness = f_ensureList(moneyness)
        tenor = f_ensureList(tenor)
        cp_flag = f_ensureList(cp_flag)
        mask = (
            (df["cp_flag"].isin(cp_flag)) &
            (df["tenor_bucket"].isin(tenor)) &
            (df["moneyness_bucket"].isin(moneyness))
        )
    else:
        mask = (
            (df["cp_flag"]==cp_flag) &
            (df["tenor_bucket"]==tenor) &
            (df["moneyness_bucket"]==moneyness)
        )
    return df.loc[mask].groupby("date", observed=True)[col].max()

# atm iv by tenor for calls
atm_iv_30_40 = f_pick(dfOptionsAgg, "atm", "30_40", "C", col="avg_iv_oi").to_frame("atm_iv_30_40")
atm_iv_41_60 = f_pick(dfOptionsAgg, "atm", "41_60", "C", col="avg_iv_oi").to_frame("atm_iv_41_60")
atm_iv_61_90 = f_pick(dfOptionsAgg, "atm", "61_90", "C", col="avg_iv_oi").to_frame("atm_iv_61_90")

# short term iv slope ### feature #1
atm_iv_slope_st = atm_iv_41_60["atm_iv_41_60"] - atm_iv_30_40["atm_iv_30_40"]

# long term iv slope ### feature #2
atm_iv_slope_lt = atm_iv_61_90["atm_iv_61_90"] - atm_iv_30_40["atm_iv_30_40"]

# put vs call otm iv skew, for 30-40 and 61-90 day tenors
iv_otm_put_30  = f_pick(dfOptionsAgg, "otm",  "30_40", "P", col="avg_iv_oi").to_frame("iv_otm_put_30")
iv_otm_call_30 = f_pick(dfOptionsAgg, "otm", "30_40", "C", col="avg_iv_oi").to_frame("iv_otm_call_30")
iv_otm_put_90  = f_pick(dfOptionsAgg, "otm",  "61_90", "P", col="avg_iv_oi").to_frame("iv_otm_put_90")
iv_otm_call_90 = f_pick(dfOptionsAgg, "otm", "61_90", "C", col="avg_iv_oi").to_frame("iv_otm_call_90")

# short term otm iv skew ### feature #3
iv_otm_skew_st = iv_otm_put_30["iv_otm_put_30"] - iv_otm_call_30["iv_otm_call_30"]

# long term otm iv skew ### feature #4
iv_otm_skew_lt = iv_otm_put_90["iv_otm_put_90"] - iv_otm_call_90["iv_otm_call_90"]

# put/call ratios for oi and volume, all buckets ### features #5 and #6
pc_oi = dailyTotals["total_oi_P"].div(dailyTotals["total_oi_C"]).mask((dailyTotals["total_oi_P"] == 0) | (dailyTotals["total_oi_C"] == 0), np.nan)
pc_vol = dailyTotals["total_vol_P"].div(dailyTotals["total_vol_C"]).mask((dailyTotals["total_vol_P"] == 0) | (dailyTotals["total_vol_C"] == 0), np.nan)

# otm put/call ratios for oi and volume, for 30-40 and 61-90 day tenors
otm_call_oi_30 = f_pick(dfOptionsAgg, "otm", "30_40", "C", col="sum_oi").to_frame("otm_call_oi_30")
otm_call_oi_90 = f_pick(dfOptionsAgg, "otm", "61_90", "C", col="sum_oi").to_frame("otm_call_oi_90")
otm_put_oi_30 = f_pick(dfOptionsAgg, "otm", "30_40", "P", col="sum_oi").to_frame("otm_put_oi_30")
otm_put_oi_90 = f_pick(dfOptionsAgg, "otm", "61_90", "P", col="sum_oi").to_frame("otm_put_oi_90")

otm_call_vol_30 = f_pick(dfOptionsAgg, "otm", "30_40", "C", col="sum_volume").to_frame("otm_call_vol_30")
otm_call_vol_90 = f_pick(dfOptionsAgg, "otm", "61_90", "C", col="sum_volume").to_frame("otm_call_vol_90")
otm_put_vol_30 = f_pick(dfOptionsAgg, "otm", "30_40", "P", col="sum_volume").to_frame("otm_put_vol_30")
otm_put_vol_90 = f_pick(dfOptionsAgg, "otm", "61_90", "P", col="sum_volume").to_frame("otm_put_vol_90")

# short term otm put/call oi and volume ratio ### features #7 and #8
otm_oi_30 = pd.merge(otm_put_oi_30, otm_call_oi_30, left_index=True, right_index=True).dropna()
otm_pc_oi_st = otm_oi_30["otm_put_oi_30"].div(otm_oi_30["otm_call_oi_30"]).mask((otm_oi_30["otm_put_oi_30"] == 0) | (otm_oi_30["otm_call_oi_30"] == 0), np.nan)

otm_vol_30 = pd.merge(otm_put_vol_30, otm_call_vol_30, left_index=True, right_index=True).dropna()
otm_pc_vol_st = otm_vol_30["otm_put_vol_30"].div(otm_vol_30["otm_call_vol_30"]).mask((otm_vol_30["otm_put_vol_30"] == 0) | (otm_vol_30["otm_call_vol_30"] == 0), np.nan)

# long term otm put/call oi and volume ratio ### features #9 and #10
otm_oi_90 = pd.merge(otm_put_oi_90, otm_call_oi_90, left_index=True, right_index=True).dropna()
otm_pc_oi_lt = otm_oi_90["otm_put_oi_90"].div(otm_oi_90["otm_call_oi_90"]).mask((otm_oi_90["otm_put_oi_90"] == 0) | (otm_oi_90["otm_call_oi_90"] == 0), np.nan)

otm_vol_90 = pd.merge(otm_put_vol_90, otm_call_vol_90, left_index=True, right_index=True).dropna()
otm_pc_vol_lt = otm_vol_90["otm_put_vol_90"].div(otm_vol_90["otm_call_vol_90"]).mask((otm_vol_90["otm_put_vol_90"] == 0) | (otm_vol_90["otm_call_vol_90"] == 0), np.nan)

# otm puts and calls % change in oi and volume, for all tenors
otm_put_oi = f_pick(dfOptionsAgg, "otm", ["30_40", "41_60", "61_90"], "P", col="sum_oi").to_frame("otm_put_oi")
otm_call_oi = f_pick(dfOptionsAgg, "otm", ["30_40", "41_60", "61_90"], "C", col="sum_oi").to_frame("otm_call_oi")
otm_put_vol = f_pick(dfOptionsAgg, "otm", ["30_40", "41_60", "61_90"], "P", col="sum_volume").to_frame("otm_put_vol")
otm_call_vol = f_pick(dfOptionsAgg, "otm", ["30_40", "41_60", "61_90"], "C", col="sum_volume").to_frame("otm_call_vol")

# ### features #11 to #14
otm_put_oi_pct_chg = otm_put_oi.pct_change().rename(columns={"otm_put_oi": "otm_put_oi_pct_chg"})['otm_put_oi_pct_chg']
otm_call_oi_pct_chg = otm_call_oi.pct_change().rename(columns={"otm_call_oi": "otm_call_oi_pct_chg"})['otm_call_oi_pct_chg']
otm_put_vol_pct_chg = otm_put_vol.pct_change().rename(columns={"otm_put_vol": "otm_put_vol_pct_chg"})['otm_put_vol_pct_chg']
otm_call_vol_pct_chg = otm_call_vol.pct_change().rename(columns={"otm_call_vol": "otm_call_vol_pct_chg"})['otm_call_vol_pct_chg']

# otm puts and calls oi and volume compared to SMA, for all tenors ### features #15 to #18
window = 10
otm_put_oi_ma = otm_put_oi.divide(otm_put_oi.rolling(window=window).mean()).rename(columns={"otm_put_oi": "otm_put_oi_ma"})['otm_put_oi_ma']
otm_call_oi_ma = otm_call_oi.divide(otm_call_oi.rolling(window=window).mean()).rename(columns={"otm_call_oi": "otm_call_oi_ma"})['otm_call_oi_ma']
otm_put_vol_ma = otm_put_vol.divide(otm_put_vol.rolling(window=window).mean()).rename(columns={"otm_put_vol": "otm_put_vol_ma"})['otm_put_vol_ma']
otm_call_vol_ma = otm_call_vol.divide(otm_call_vol.rolling(window=window).mean()).rename(columns={"otm_call_vol": "otm_call_vol_ma"})['otm_call_vol_ma']

# final features dataframe
# index to align on
idx = dailyTotals.index.unique().sort_values()
features = pd.DataFrame(index=idx)
features.index.name = "date"

# ensure every Series has a column name
series_map = {
    "atm_iv_slope_st": atm_iv_slope_st,
    "atm_iv_slope_lt": atm_iv_slope_lt,
    "iv_otm_skew_st": iv_otm_skew_st,
    "iv_otm_skew_lt": iv_otm_skew_lt,
    "pc_oi": pc_oi,
    "pc_vol": pc_vol,
    "otm_pc_oi_st": otm_pc_oi_st,
    "otm_pc_vol_st": otm_pc_vol_st,
    "otm_pc_oi_lt": otm_pc_oi_lt,
    "otm_pc_vol_lt": otm_pc_vol_lt,
    "otm_put_oi_pct_chg": otm_put_oi_pct_chg,
    "otm_call_oi_pct_chg": otm_call_oi_pct_chg,
    "otm_put_vol_pct_chg": otm_put_vol_pct_chg,
    "otm_call_vol_pct_chg": otm_call_vol_pct_chg,
    "otm_put_oi_ma": otm_put_oi_ma,
    "otm_call_oi_ma": otm_call_oi_ma,
    "otm_put_vol_ma": otm_put_vol_ma,
    "otm_call_vol_ma": otm_call_vol_ma,
}

series_list = [s.rename(name) for name, s in series_map.items()]

# join all at once (aligns on the date index)
features = features.join(series_list, how="left").iloc[(window-1):, :].reset_index()
features.dropna(inplace=True)

# export to excel
out_path = os.path.join("..", "Data", "spy_options_features.csv")
features.to_csv(out_path, sep=';', na_rep='0', index=False)

