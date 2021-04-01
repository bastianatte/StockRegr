from config import main_conf as mc
import logging
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ta.momentum import ROCIndicator, WilliamsRIndicator, RSIIndicator, stochrsi_k
from ta.trend import EMAIndicator, MACD
from ta.volume import AccDistIndexIndicator
from utils.misc import get_logger


seldf_log = get_logger(__name__)
seldf_log.setLevel(logging.INFO)


def preprocess_df(df, csv_string):
    """
    It creates an unique list of dfs by reading
    csv files. It select list of features by
    looking at config file.
    :param df: pandas df
    :param csv_string: stock name string
    """
    df = df[mc["columns"]]
    df = create_ticker(df, csv_string)
    # lagged features
    df = lagged_daily_price_return(df)
    df = create_next_day_return(df)
    df = lagged_daily_indicators(df)
    # Indicators
    df = create_tech_indicators(df)
    scaled_df = scale_df(df)
    return df, scaled_df


def scale_df(df):
    scaled_df = df.copy()
    col_names = mc["features"]
    features = scaled_df[col_names]
    trans_feat = MinMaxScaler().fit_transform(features.values)
    scaled_df[col_names] = trans_feat
    return scaled_df


def create_tech_indicators(df):
    """
    Create Technical Indicators TI
    :param df: dataframe
    :return: dataframe
    """
    df = ta_ema(df)
    df = ta_stochastic(df)
    df = ta_price_rate_of_change(df)
    df = ta_rsi(df)
    df = ta_acc_distr_index(df)
    df = ta_macd(df)
    df = ta_williams_r_indicator(df)
    df = disparity(df)
    return df


def create_ticker(df, stock_ticker):
    """
    Create stock ticker for each raw.
    :param df: Unmodified pandas dataframe
    :param stock_ticker: stock ticker
    :return: modified dataframe
    """
    df[mc["ticker"]] = stock_ticker
    return df


def lagged_daily_price_return(df):
    """
    Create actual day return variable.
    :param df: Unmodified pandas dataframe
    :return: modified pandas dataframe
    """
    df[mc["lagged_daily_rt"]] = (df['Close'].subtract(df['Open'])).div(df['Open'])
    return df


def create_next_day_return(df):
    """"
    Create actual next day return by
    shifting actual day return variable
    by -1.
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    df[mc["label"]] = df[mc["lagged_daily_rt"]].shift(periods=-1, fill_value=0)
    return df


def lagged_daily_indicators(df):
    """
    It maked the lagged daily return for a given trading
    day d, in the lag [d−10, d−1].
    :param df:
    :return: pandas dataframe
    """
    temp_df = df.copy()
    for i in range(1, 11):
        string = "label_" + str(i)
        temp_df[mc[string]] = temp_df[mc["lagged_daily_rt"]].shift(periods=i, fill_value=0)
    return temp_df


def ta_ema(df):
    """
    It create the exponential moving average
    variable.
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    temp_df = df.copy()
    # test = EMAIndicator(temp_df["Close"], window=mc["ema_period"], fillna=True)
    test = EMAIndicator(temp_df["Close"], fillna=False)
    temp_df["ta_ema"] = test.ema_indicator()
    return temp_df


def ta_stochastic(df):
    """
    Stochastic %K calculation
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    temp_df = df.copy()
    temp_df["stoch"] = stochrsi_k(temp_df["Close"])
    # temp_df["stoch"] = stochrsi_k(temp_df["Close"], window=mc["stochastic_high"],
    #                               smooth1=mc["stochastic_low"], smooth2=mc["stochastic_low"])
    return temp_df


def ta_price_rate_of_change(df):
    """
    Price rate of change (ROC) calculation
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    temp_df = df.copy()
    test = ROCIndicator(close=temp_df["Close"])
    # test = ROCIndicator(close=temp_df["Close"], window=mc["proc_period"])
    temp_df["ta_proc"] = test.roc()
    return temp_df


def ta_rsi(df):
    """
    Relative Strenght Index
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    temp_df = df.copy()
    # temp = RSIIndicator(close=df["Close"], window=mc["rsi_wind"])
    temp = RSIIndicator(close=df["Close"])
    temp_df["ta_rsi"] = temp.rsi()
    return temp_df


def ta_acc_distr_index(df):
    """
    Accumulation Distribution Index
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    temp_df = df.copy()
    temp = AccDistIndexIndicator(temp_df["High"], temp_df["Low"], temp_df["Close"],
                                 temp_df["Volume"], fillna=False)
    temp_df["adi"] = temp.acc_dist_index()
    return temp_df


def ta_macd(df):
    """
    Moving Average Convergence Divergence (MACD) calculation.
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    temp_df = df.copy()
    temp = MACD(close=temp_df["Close"], fillna=False)
    # temp = MACD(close=temp_df["Close"], window_slow=26, window_fast=12, window_sign=9, fillna=True)
    temp_df["macd"] = temp.macd()
    temp_df["macd_diff"] = temp.macd_diff()
    temp_df["macd_signal"] = temp.macd_signal()
    return temp_df


def ta_williams_r_indicator(df):
    """
    Williams R Indicator calculation.
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    temp_df = df.copy()
    temp = WilliamsRIndicator(high=temp_df["High"], low=temp_df["Low"], close=temp_df["Close"], fillna=False)
    temp_df["will_r_ind"] = temp.williams_r()
    return temp_df


def disparity(df):
    """
    Disparity Calculation
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    temp_df = df.copy()
    sma5 = temp_df["Close"].rolling(window=5).mean()
    sma10 = temp_df["Close"].rolling(window=10).mean()
    temp_df["disp_5"] = (temp_df["Close"]/sma5)*100
    temp_df["disp_10"] = (temp_df["Close"]/sma10)*100
    return temp_df


# def relative_strenght_index(df):
#     """
#     Relative Strenght Index
#     :param df: pandas dataframe
#     :return: pandas dataframe
#     """
#     # Window length for moving average
#     window_length = 14
#     temp_df = df.copy()
#     close = temp_df['Close']
#     # Get the difference in price from previous step
#     delta = close.diff()
#     # Make the positive gains (up) and negative gains (down) Series
#     up, down = delta.copy(), delta.copy()
#     up[up < 0] = 0
#     down[down > 0] = 0
#
#     # Calculate the EWMA
#     roll_up1 = up.ewm(span=window_length).mean()
#     roll_down1 = down.abs().ewm(span=window_length).mean()
#
#     # Calculate the RSI based on EWMA
#     rs1 = roll_up1 / roll_down1
#     rsi1 = 100.0 - (100.0 / (1.0 + rs1))
#
#     # Calculate the SMA
#     roll_up2 = up.rolling(window_length).mean()
#     roll_down2 = down.abs().rolling(window_length).mean()
#
#     # Calculate the RSI based on SMA
#     rs2 = roll_up2 / roll_down2
#     rsi2 = 100.0 - (100.0 / (1.0 + rs2))
#     temp_df["rsi_ewma"] = rsi1
#     temp_df["rsi_sma"] = rsi2
#     return temp_df


# def acc_distr_index(df):
#     temp_df = df.copy()
#     temp_df["adi_1"] = ((temp_df["Close"] - temp_df["Low"])-(temp_df["High"] - temp_df["Close"]) /
#                         (temp_df["High"] - temp_df["Low"]))*temp_df["Volume"]
#     return temp_df
#
#
# def acc_distr_index_test(df, k):
#     """
#     Accumulation Distribution Index
#     :param df: pandas dataframe
#     :param k: window parameter
#     :return: pandas dataframe
#     """
#     temp_df = df.copy()
#     low_min = temp_df["Low"].rolling(window=k).min()
#     high_max = temp_df["High"].rolling(window=k).max()
#     # Fast Stochastic
#     temp_df["adi_2"] = (((temp_df["Close"]-low_min)-(high_max-temp_df["Close"]))/(high_max-low_min))*temp_df["Volume"]
#     return temp_df


# def stochastics(df, k, d):
#     """
#     Fast stochastic calculation
#     %K = (Current Close - Lowest Low)/
#     (Highest High - Lowest Low) * 100
#     %D = 3-day SMA of %K
#
#     Slow stochastic calculation
#     %K = %D of fast stochastic
#     %D = 3-day SMA of %K
#
#     When %K crosses above %D, buy signal
#     When the %K crosses below %D, sell signal
#     """
#     temp_df = df.copy()
#     # Set minimum low and maximum high of the k stoch
#     low_min = temp_df["Low"].rolling(window=k).min()
#     high_max = temp_df["High"].rolling(window=k).max()
#     # Fast Stochastic
#     temp_df['k_fast'] = 100 * (temp_df["Close"] - low_min)/(high_max - low_min)
#     temp_df['d_fast'] = temp_df['k_fast'].rolling(window=d).mean()
#     # Slow Stochastic
#     temp_df['k_slow'] = temp_df["d_fast"]
#     temp_df['d_slow'] = temp_df['k_slow'].rolling(window=d).mean()
#     return temp_df

# def daily_return(df):
#     """
#     Calculate daily return
#     :param df: pandas dataframe
#     :return: pandas dataframe
#     """
#     temp_df = df.copy()
#     temp_df[mc["daily_return"]] = temp_df["Close"]/temp_df["Close"].shift(1)-1
#     return temp_df


# def pct_daily_return(df):
#     temp_df = df.copy()
#     temp_df["pct_return"] = temp_df["Close"].pct_change()
#     temp_df["pct_return_mean"] = temp_df["pct_return"].mean()
#     temp_df["port_return"] = np.sum(temp_df["pct_return_mean"])
#     return temp_df


# def log_return(df):
#     temp_df = df.copy()
#     # daily log return
#     temp_df[mc["log_return"]] = np.log(temp_df['Close'] / temp_df['Close'].shift(1))
#     # daily std dev
#     # print(temp_df[mc["log_return"]].std())
#     temp_df["daily_rt_std"] = temp_df[mc["log_return"]].std()
#     # annualized daily std dev
#     temp_df["ann_daily_std"] = temp_df["daily_rt_std"] * 252 ** 0.5
#     # # volatility
#     temp_df[mc["volatility"]] = temp_df[mc["log_return"]].std() * np.sqrt(252)
#     temp_df["vol"] = temp_df[mc["log_return"]].rolling(window=252).std() * np.sqrt(252)
#     # # sharpe ratio
#     # temp_df[mc["sharpe_ratio"]] = (temp_df[mc["log_return"]].mean() - mc["rfr"]) / temp_df[mc["volatility"]]
#     return temp_df


# def ema(df):
#     """
#     It create the exponential moving average
#     variable.
#     :param df: pandas dataframe
#     :return: pandas dataframe
#     """
#     mod_price = df['Close'].copy()
#     ema10alt = mod_price.ewm(span=mc["ema_period"], adjust=False).mean()
#     mod_price.iloc[0:9] = np.nan
#     df['ema'] = np.round(ema10alt, decimals=6)
#     return df

# def price_rate_of_change(df):
#     """
#     Price rate of change (ROC) calculation
#     :param df: pandas dataframe
#     :return: pandas dataframe
#     """
#     temp_df = df.copy()
#     temp_df["proc"] = temp_df["Close"].pct_change(periods=mc["proc_period"])*100
#     return temp_df
#     return temp_df



