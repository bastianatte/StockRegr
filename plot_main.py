import pandas as pd
import os
import fnmatch
import logging
import argparse
from utils.misc import create_folder
from utils.plot import make_multiple_profit_plot, make_multiple_profit_plot_ranking
from config import main_conf as mc
from config import plot_conf as pc
from classes.Plotter import Plotter

from utils.misc import get_logger, csv_maker

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description="Load flag to run StockRegression")
parser.add_argument('-i', '--input', type=str, metavar='', required=True,
                    help='Specify the path of the root file'
                    )
parser.add_argument('-o', '--output', type=str, metavar='', required=True,
                    help='Specify the output path'
                    )
args = parser.parse_args()


def plot_exe(inpt, output):
    """
    Create a general full_df by looping over
    csv files located by input flag.
    :param inpt: args.input
    :param output: args.output
    :return: pandas dataframe
    """
    dfs_list = []
    out_path = create_folder(output, pc["plot_dir"])
    print(output, out_path)
    for csv_file in os.listdir(inpt):
        if fnmatch.fnmatch(csv_file, '*.csv'):
            csv, csv_string = csv_maker(str(inpt), csv_file)
            df_temp = pd.read_csv(csv, parse_dates=[mc["date_clm"]], index_col=0)
            df_temp = setting_date_as_index(df_temp)
            single_model_profit(df_temp, csv_string, out_path)
            dfs_list.append((df_temp, csv_string))
    logger.info("Singular plots done!!")
    multiple_profit_plot(dfs_list, out_path)
    return dfs_list


def single_model_profit(df, csv_str, output):
    """
    Plot daily profit and ls_profit by checking
    the columns
    :param df: pandas df
    :param csv_str: model name string
    :param output: output path
    :return: None
    """
    if pc["profit_csv"] in csv_str:
        if pc["rf_csv"] in csv_str:
            output_loc = create_folder(output, csv_str)
            pl = Plotter(df, csv_str, output_loc)
            pl.plot_profit_template()
        else:
            output_loc = create_folder(output, csv_str)
            pl = Plotter(df, csv_str, output_loc)
            pl.plot_profit_template()


def setting_date_as_index(df):
    """
    Setting "Date columns as index"
    :param df: pandas dataframe
    :return: pandas df indexed
    """
    df.set_index("Date", inplace=True)
    return df


def multiple_profit_plot(df_list, output):
    """
    Make global plot long short profit.
    :param df_list: list of pd dfs
    :param output: output path
    :return: None
    """
    logger.info("in multiple profit plots")
    dfs_profit_list = []
    for item in df_list:
        df = item[0]
        df_name = item[1]
        if pc["daily_profit"] in df.columns.to_list():
            dfs_profit_list.append((df, df_name))
    rf_list, lr_list = create_list_of_same_models(dfs_profit_list)
    make_multiple_profit_plot(dfs_profit_list, "total_daily_profit", output)
    make_multiple_profit_plot_ranking(dfs_profit_list, output)
    make_multiple_profit_plot(rf_list, "rf_daily_profit", output)
    make_multiple_profit_plot(lr_list, "lr_daily_profit", output)


def create_list_of_same_models(df_list):
    """
    It split the df list in as many list
    as used models.
    :param df_list:
    :return: list
    """
    rf = []
    lr = []
    for item in df_list:
        df = item[0]
        df_name = item[1]
        if "lr" in df_name:
            lr.append((df, df_name))
        else:
            rf.append((df, df_name))
    return rf, lr


if __name__ == '__main__':
    logger.info("~~~### WELL DONE, plot section is now ACTIVE ###~~~")
    logger.info("in plot main")
    dfs = plot_exe(args.input, args.output)
