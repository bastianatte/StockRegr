import argparse
import fnmatch
import logging
import os
import pandas as pd
from config import main_conf as mc
import time
from classes.df_selection import preprocess_df
from datetime import timedelta
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
args = parser.parse_args()
logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)


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


def filter_df(df, stock_name, start_date, end_date):
    """
    It filter the df looking at the start
    and end date (datetime object).
    :param df: pandas dataframe
    :param stock_name: string stock name
    :param start_date: start date
    :param end_date: end date
    :return: pandas dataframe
    """
    df = df[(df[mc["ticker"]] == stock_name) &
            (df[mc["date_clm"]] > start_date) &
            (df[mc["date_clm"]] < end_date)]
    return df


def split_df(filtered_df, stock_name, date_start, date_end):
    """
    It split the df in x and y subset by looking
    at the dates (datetime).
    :param filtered_df: pandas dataframe to split
    :param stock_name: string stock name
    :param date_start: start date
    :param date_end: end date
    :return: 
    """
    df_splitted = filter_df(filtered_df, stock_name, date_start, date_end)
    x = df_splitted[mc["features"]]
    y = df_splitted[mc["label"]]
    return df_splitted, x, y


def make_threshold_lenght(year_start, year_end, test_wind):
    """
    Return the minimun lenght to apply regression models.
    :param year_start: train year start
    :param year_end: train year end
    :param test_wind: test windows
    :return: threshold lenght
    """
    days_in_year = 250
    return ((year_end - year_start) * 250) + (test_wind * days_in_year)


def create_test_df(train_end):
    """
    Calcuate test windows for the walkforward
    procedure.
    :param train_end: Training end date
    :return: Test start date, test end date
    """
    test_start = train_end + timedelta(days=1)
    test_end = test_start + timedelta(days=365)
    return test_start, test_end


if __name__ == '__main__':
    logger.info("in main")
    df_pred_list = []
    wnd_cnt = 1
    dataframe = create_df(args.input)
    start = time.time()
    logger.info("{} train windows.".format(len(mc["train_window"])))

    for window in mc["train_window"]:
        df_pred_list.clear()
        window_start = time.time()
        bad_df_list = []
        year_from = window[0].year
        year_to = window[1].year
        test_start, test_end = create_test_df(window[1])
        test_window = test_end.year - test_start.year
        thresh_raw = make_threshold_lenght(year_from, year_to, test_window)
        logger.info("#### Time windows #{}: Train from {} to {}"
                    "#### Test from : {} to {}".format(wnd_cnt, year_from, year_to, test_start, test_end))
        wnd_cnt += 1
        for stock in set(dataframe[mc["ticker"]].values):
            df_stock_train, x_train, y_train = split_df(
                dataframe,
                stock,
                date_start=window[0],
                date_end=window[1])
            df_stock_test, x_test, y_test = split_df(
                dataframe,
                stock,
                test_start,
                test_end)
            if len(df_stock_train) + len(df_stock_test) < thresh_raw:
                bad_df_list.append(stock)
                continue
            # Apply models
            # print("train ", stock,  df_stock_test.iloc[0, 0], df_stock_test.iloc[0, 0], df_stock_train.shape)
            mod = Models(x_train, y_train, x_test)
            pred_rf = mod.random_forest()
            pred_lr = mod.linear_model()
            df_pred = df_stock_test.copy()
            df_pred[mc["lr_clm_name"]] = pred_lr
            df_pred[mc["rf_clm_name"]] = pred_rf

            df_pred_list.append(df_pred)
        stop = time.time()
        logger.info("{} bad stocks over {}".format(
            len(bad_df_list),
            len(set(dataframe[mc["ticker"]].values))))
        logger.info("Total run time: {}, model run time: {}".format(
            (stop-start)/60,
            (stop-window_start)/60))
        # concatenation to a single pred df
        dataframe_pred = pd.concat(df_pred_list)
        # a ranking for a model
        df_pred_rf_long, df_pred_rf_short, profit_rf_df = rank_exe(dataframe_pred, mc["rf_clm_name"])
        df_pred_lr_long, df_pred_lr_short, profit_lr_df = rank_exe(dataframe_pred, mc["lr_clm_name"])

        # # storing csv
        path_csv = create_folder(args.output, "csv")
        store_csv(dataframe, path_csv, f"general-{year_from}-{year_to}")
        store_csv(df_pred_rf_long, path_csv, f"rf_long-{year_from}-{year_to}")
        store_csv(df_pred_rf_short, path_csv, f"rf_short-{year_from}-{year_to}")
        store_csv(df_pred_lr_long, path_csv, f"lr_long-{year_from}-{year_to}")
        store_csv(df_pred_lr_short, path_csv, f"lr_short-{year_from}-{year_to}")
        store_csv(dataframe_pred, path_csv, f"pred-{year_from}-{year_to}")
        store_csv(profit_rf_df, path_csv, f"profit_rf-{year_from}-{year_to}")
        store_csv(profit_lr_df, path_csv, f"profit_lr-{year_from}-{year_to}")
        del dataframe_pred
        del profit_rf_df

