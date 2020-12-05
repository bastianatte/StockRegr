from config.config import main_conf as mc
import logging
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
    seldf_log.info("{} with shape: {}".format(csv_string, df.shape))
    df = create_actual_day_rt(df)
    df = create_next_day_return(df)
    df = create_ticker(df, csv_string)
    return df


def create_actual_day_rt(df):
    """
    Create actual day return variable.
    :param df: Unmodified pandas dataframe
    :return: modified pandas dataframe
    """
    df[mc["actual_day_rt_clm"]] = (df['Close'].subtract(df['Open'])).div(df['Open'])
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


def create_next_day_return(df):
    """"
    Create actual next day return by
    shifting actual day return variable
    by -1.
    :param df: pandas dataframe
    :return: pandas dataframe
    """
    df[mc["label"]] = df[mc["actual_day_rt_clm"]].shift(periods=-1, fill_value=0)
    return df
