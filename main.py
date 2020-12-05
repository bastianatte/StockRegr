import argparse
import fnmatch
# import json
import logging
import os
import pandas as pd
from config import main_conf as mc

from classes.df_selection import preprocess_df
from utils.misc import get_logger, create_folder, create_each_stock_folder, store_csv
from utils.ranking import rank_exe
from classes.models import Models

parser = argparse.ArgumentParser(description="Load flag to run StockRegression")
parser.add_argument('-i', '--input', type=str, metavar='', required=True,
                    help='Specify the path of the root file'
                    )
parser.add_argument('-o', '--output', type=str, metavar='', required=True,
                    help='Specify the output path'
                    )
# declare logger
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)
logger.info("Reading config file")
# args parser
args = parser.parse_args()


def create_df(ipt):
    """
    Create a general full_df by looping over
    csv files located by input flag.
    :param ipt: args.input
    :return: pandas dataframe
    """
    csv_cnt = 0
    dfs = []
    for csv_file in os.listdir(ipt):
        if fnmatch.fnmatch(csv_file, '*.csv'):
            if csv_cnt == -1:
                break
            else:
                csv_cnt += 1
                csv, csv_string = create_each_stock_folder(str(ipt), csv_file)
                df_temp = pd.read_csv(csv, parse_dates=[mc["date_clm"]])
                df_temp = preprocess_df(df_temp, csv_string)
                dfs.append(df_temp)
    df = pd.concat(dfs)
    return df


def filter_df(df, stock_name, start, end):
    """
    It filter the df looking at the start
    and end date (datetime object).
    :param df: pandas dataframe
    :param stock_name: string stock name
    :param start: start date
    :param end: end date
    :return: pandas dataframe
    """
    df = df[(df[mc["ticker"]] == stock_name) & (df[mc["date_clm"]] > start) & (df[mc["date_clm"]] < end)]
    return df


def split_df(filtered_df, stock_name, start, end):
    """
    It split the df in x and y subset by looking
    at the dates (datetime).
    :param filtered_df: pandas dataframe to split
    :param stock_name: string stock name
    :param start: start date
    :param end: end date
    :return: 
    """
    df_splitted = filter_df(filtered_df, stock_name, start, end)
    x = df_splitted[mc["features"]]
    y = df_splitted[mc["label"]]
    return df_splitted, x, y


if __name__ == '__main__':
    logger.info("in main")
    df_pred_list = []
    dataframe = create_df(args.input)
    for stock in set(dataframe[mc["ticker"]].values):
        df_stock_train, x_train, y_train = split_df(dataframe,
                                                    stock,
                                                    mc["train_start"],
                                                    mc["train_end"])
        df_stock_test, x_test, y_test = split_df(dataframe,
                                                 stock,
                                                 mc["test_start"],
                                                 mc["test_end"])
        if len(df_stock_train) + len(df_stock_test) < 3000:
            logger.info("{} skip due to not enough data.".format(stock))
            continue
        logger.info("{} with shape: {}".format(stock, df_stock_train.shape[0]))
        mod = Models(x_train, y_train, x_test, y_test)
        pred_rf = mod.random_forest()
        pred_lr = mod.linear_model()
        df_pred = df_stock_test.copy()
        df_pred[mc["rf_clm_name"]] = pred_rf
        df_pred[mc["lr_clm_name"]] = pred_lr
        df_pred_list.append(df_pred)
    dataframe_pred = pd.concat(df_pred_list)
    df_pred_rf_long, df_pred_rf_short, profit_rf_df = rank_exe(dataframe_pred, mc["rf_clm_name"])
    df_pred_lr_long, df_pred_lr_short, profit_lr_df = rank_exe(dataframe_pred, mc["lr_clm_name"])
    # storing csv
    path_csv = create_folder(args.output, "csv")
    store_csv(df_pred_rf_long, path_csv, "rf_long")
    store_csv(df_pred_rf_short, path_csv, "rf_short")
    store_csv(df_pred_lr_long, path_csv, "lr_long")
    store_csv(df_pred_lr_short, path_csv, "lr_short")
    store_csv(dataframe_pred, path_csv, "pred")
    store_csv(dataframe, path_csv, "general")
    store_csv(profit_rf_df, path_csv, "profit_rf")
    store_csv(profit_lr_df, path_csv, "profit_lr")

