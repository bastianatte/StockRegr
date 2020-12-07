import argparse
import fnmatch
import logging
import os
import pandas as pd
from config import main_conf as mc

from classes.df_selection import preprocess_df
from utils.misc import get_logger, create_folder, csv_maker, store_csv
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
                csv, csv_string = csv_maker(str(ipt), csv_file)
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
    bad_df_list = []
    dataframe = create_df(args.input)
    for window in mc["train_window"]:
        year_from = window[0].year
        year_to = window[1].year
        logger.info("### Time windows: from {} to {}".format(year_from, year_to))
        for stock in set(dataframe[mc["ticker"]].values):
            df_stock_train, x_train, y_train = split_df(
                dataframe,
                stock,
                start=window[0],
                end=window[1])
            df_stock_test, x_test, y_test = split_df(
                dataframe,
                stock,
                mc["test_start"],
                mc["test_end"])
            if len(df_stock_train) + len(df_stock_test) < 3000:
                bad_df_list.append(stock)
                # logger.info("{} skip due to not enough data.".format(stock))
                continue
            # logger.info("{} with shape: {}".format(stock, df_stock_train.shape[0]))
            # Apply models
            mod = Models(x_train, y_train, x_test, y_test)
            pred_rf = mod.random_forest()
            pred_lr = mod.linear_model()
            df_pred = df_stock_test.copy()
            df_pred[mc["rf_clm_name"]] = pred_rf
            df_pred[mc["lr_clm_name"]] = pred_lr
            df_pred_list.append(df_pred)
        logger.info("{} bad stocks over {}  ".format(len(bad_df_list), len(dataframe[mc["ticker"]].values)))
        # concatenation to a single pred df
        dataframe_pred = pd.concat(df_pred_list)
        # a ranking for a model
        df_pred_rf_long, df_pred_rf_short, profit_rf_df = rank_exe(dataframe_pred, mc["rf_clm_name"])
        df_pred_lr_long, df_pred_lr_short, profit_lr_df = rank_exe(dataframe_pred, mc["lr_clm_name"])

        # storing csv
        path_csv = create_folder(args.output, "csv")
        store_csv(df_pred_rf_long, path_csv, f"rf_long-{year_from}-{year_to}")
        store_csv(df_pred_rf_short, path_csv, f"rf_short-{year_from}-{year_to}")
        store_csv(df_pred_lr_long, path_csv, f"lr_long-{year_from}-{year_to}")
        store_csv(df_pred_lr_short, path_csv, f"lr_short-{year_from}-{year_to}")
        store_csv(dataframe_pred, path_csv, f"pred-{year_from}-{year_to}")
        store_csv(dataframe, path_csv, f"general-{year_from}-{year_to}")
        store_csv(profit_rf_df, path_csv, f"profit_rf-{year_from}-{year_to}")
        store_csv(profit_lr_df, path_csv, f"profit_lr-{year_from}-{year_to}")

