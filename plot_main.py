import pandas as pd
import os
import fnmatch
import logging
import argparse
from utils.misc import create_folder, asc_sort_tuple
from classes.ProfitPlotter import ProfitPlotter
from config import main_conf as mc
from config import plot_conf as pc
from classes.Plotter import Plotter
from utils.misc import get_logger, csv_maker

logger = get_logger(__name__)
logger.setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description="Load flag to run StockRegression")
parser.add_argument('-i', '--input', type=str, metavar='', required=True,
                    help='Specify the path of the root file')
parser.add_argument('-o', '--output', type=str, metavar='', required=True,
                    help='Specify the output path')
args = parser.parse_args()


def plot_exe(inpt, output):
    """
    Create a general full_df by looping over
    csv files located by input flag.
    :param inpt: args.input
    :param output: output path
    :return: pandas dataframe
    """
    logger.info("in plot execute")
    dfs_list = []
    for csv_file in os.listdir(inpt):
        if fnmatch.fnmatch(csv_file, '*.csv'):
            # create list of dfs
            csv, csv_string = csv_maker(str(inpt), csv_file)
            # print(csv_string)
            df = pd.read_csv(csv, parse_dates=[mc["date_clm"]], index_col=0)
            df_temp = df.copy()
            df_temp = setting_date_as_index(df_temp)
            # plot profit for each stock
            single_model_profit(df_temp, csv_string, output)
            dfs_list.append((df_temp, csv_string))
    logger.info("plot execute done!")
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
    df_temp = df.copy()
    df_temp.set_index("Date", inplace=True)
    return df_temp


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
            print(df_name)

    rf, lr, gbr, knr, lasso, enr, dtr, vef, ve1, veb, ef, eb, ebcv = create_list_of_same_models(dfs_profit_list)

    # make profit plot and create statistical dataframe
    rf_df, rf_stat_dict, rf_stat_df = make_profit_and_create_stat_dict(rf, output, "rf")
    lr_df, lr_stat_dict, lr_stat_df = make_profit_and_create_stat_dict(lr, output, "lr")
    dtr_df, dtr_stat_dict, dtr_stat_df = make_profit_and_create_stat_dict(dtr, output, "dtr")
    gbr_df, gbr_stat_dict, gbr_stat_df = make_profit_and_create_stat_dict(gbr, output, "gbr")
    knr_df, knr_stat_dict, knr_stat_df = make_profit_and_create_stat_dict(knr, output, "knr")
    lasso_df, lasso_stat_dict, lasso_stat_df = make_profit_and_create_stat_dict(lasso, output, "lasso")
    enr_df, enr_stat_dict, enr_stat_df = make_profit_and_create_stat_dict(enr, output, "enr")
    vef_df, vef_stat_dict, vef_stat_df = make_profit_and_create_stat_dict(vef, output, "vot_ens_full")
    ve1_df, ve1_stat_dict, ve1_stat_df = make_profit_and_create_stat_dict(ve1, output, "vot_ens_1")
    veb_df, veb_stat_dict, veb_stat_df = make_profit_and_create_stat_dict(veb, output, "vot_ens_best")
    ref_df, ref_stat_dict, ref_stat_df = make_profit_and_create_stat_dict(ef, output, "reg_ens_full")
    reb_df, reb_stat_dict, reb_stat_df = make_profit_and_create_stat_dict(eb, output, "reg_ens_best")
    rebcv_df, rebcv_stat_dict, rebcv_stat_df = make_profit_and_create_stat_dict(ebcv, output, "reg_ens_best_cv")

    # filling statistical variables
    rf_stat_df = rf_stat_df.join(lr_stat_df["lr"])
    rf_stat_df = rf_stat_df.join(dtr_stat_df["dtr"])
    rf_stat_df = rf_stat_df.join(gbr_stat_df["gbr"])
    rf_stat_df = rf_stat_df.join(knr_stat_df["knr"])
    rf_stat_df = rf_stat_df.join(lasso_stat_df["lasso"])
    rf_stat_df = rf_stat_df.join(enr_stat_df["enr"])
    rf_stat_df = rf_stat_df.join(vef_stat_df["vot_ens_full"])
    rf_stat_df = rf_stat_df.join(ve1_stat_df["vot_ens_1"])
    rf_stat_df = rf_stat_df.join(veb_stat_df["vot_ens_best"])
    rf_stat_df = rf_stat_df.join(ref_stat_df["reg_ens_full"])
    rf_stat_df = rf_stat_df.join(reb_stat_df["reg_ens_best"])
    rf_stat_df = rf_stat_df.join(rebcv_stat_df["reg_ens_best_cv"])
    make_metrics_plot(rf_stat_df, output)
    df_total_list = (lr_df + rf_df + dtr_df + gbr_df + knr_df + lasso_df + enr_df + vef_df + ve1_df +
                     veb_df + ref_df + reb_df + rebcv_df)
    make_profit_plot(df_total_list, output, "total")
    logger.info("Multiple profit done plots done!!")


def make_metrics_plot(df, output):
    """
    Plot metrics table
    :param df: pandas df
    :param output: output path
    :return: None
    """
    pl = Plotter(df, "metrics", output)
    pl.plot_metrics_table()


def make_profit_and_create_stat_dict(model, output, model_string):
    make_profit_plot(model, output, model_string)
    model_df = create_unique_df(model, model_string)
    stat_dict_string = "pyfolio_" + model_string
    stat_dict = make_pyfolio_plot(model_df, output, stat_dict_string)
    # make_pyfolio_plot(model_df, output, stat_dict_string)
    stat_df = pd.DataFrame(stat_dict.items(), columns=["metrics", model_string])
    return model_df, stat_dict, stat_df


def create_unique_df(tup, name):
    lst = []
    lst1 = []
    for item in tup:
        df = item[0]
        lst.append(df)
    df = pd.concat(lst)
    lst1.append((df, name))
    return lst1


def make_profit_plot(model_list, output, model_name):
    """
    It produce profit plot
    :param model_list: model dataframe list
    :param output: output path
    :param model_name: model name
    :return: None
    """
    # logger.info("... in make {} profit plot".format(model_name))
    model = ProfitPlotter(model_list, model_name, output)
    model.make_multiple_profit_plot()
    model.make_multiple_profit_plot_ranking()
    logger.info("{} profit plot done!".format(model_name))


def make_pyfolio_plot(model_list, output, model_name):
    # logger.info("... in make {} pyfolio plot".format(model_name))
    model = ProfitPlotter(model_list, model_name, output)
    stat_dict = model.make_empirical()
    # logger.info("{} plot done!".format(model_name))
    return stat_dict


def profit_shift(df_list):
    """
    It shift profit value by looking at the last value of the
    previous dataframe
    :param df_list: dataframes list
    :return: sorted dataframes list
    """
    shift = 0
    sorted_tuple = []
    for item in df_list:
        df = item[0]
        df_name = item[1]
        df["shifted_long_short_profit"] = df["long_short_profit"]
        df["shifted_long_short_profit"] = df["shifted_long_short_profit"] + shift
        shift *= 0
        shift = df["shifted_long_short_profit"].iloc[-1]
        sorted_tuple.append((df, df_name))
    return sorted_tuple


def create_list_of_same_models(df_list):
    """
    It split the df list in as many list
    as used models.
    :param df_list: list of dataframe
    :return: list
    """
    logger.info("...in create list of same models")
    rf = []
    lr = []
    gbr = []
    knr = []
    lasso = []
    enr = []
    dtr = []
    ve_full = []
    ve_1 = []
    ve_best = []
    re_full = []
    re_best = []
    re_best_cv = []
    for item in df_list:
        df = item[0]
        df_name = item[1]
        if "lr" is df_name:
            lr.append((df, df_name))
            lr = asc_sort_tuple(lr)
            lr = profit_shift(lr)
        if "rf" is df_name:
            rf.append((df, df_name))
            rf = asc_sort_tuple(rf)
            rf = profit_shift(rf)
        if "gbr" is df_name:
            gbr.append((df, df_name))
            gbr = asc_sort_tuple(gbr)
            gbr = profit_shift(gbr)
        if "knr" is df_name:
            knr.append((df, df_name))
            knr = asc_sort_tuple(knr)
            knr = profit_shift(knr)
        if "lasso" is df_name:
            lasso.append((df, df_name))
            lasso = asc_sort_tuple(lasso)
            lasso = profit_shift(lasso)
        if "enr" is df_name:
            enr.append((df, df_name))
            enr = asc_sort_tuple(enr)
            enr = profit_shift(enr)
        if "dtr" is df_name:
            dtr.append((df, df_name))
            dtr = asc_sort_tuple(dtr)
            dtr = profit_shift(dtr)
        if "vot_ens_full" is df_name:
            ve_full.append((df, df_name))
            ve_full = asc_sort_tuple(ve_full)
            ve_full = profit_shift(ve_full)
        if "vot_ens_1" is df_name:
            ve_1.append((df, df_name))
            ve_1 = asc_sort_tuple(ve_1)
            ve_1 = profit_shift(ve_1)
        if "vot_ens_best" is df_name:
            ve_best.append((df, df_name))
            ve_best = asc_sort_tuple(ve_best)
            ve_best = profit_shift(ve_best)
        if "reg_ens_full" is df_name:
            re_full.append((df, df_name))
            re_full = asc_sort_tuple(re_full)
            re_full = profit_shift(re_full)
        if "reg_ens_best" is df_name:
            re_best.append((df, df_name))
            re_best = asc_sort_tuple(re_best)
            re_best = profit_shift(re_best)
        if "reg_ens_best_cv" is df_name:
            re_best_cv.append((df, df_name))
            re_best_cv = asc_sort_tuple(re_best_cv)
            re_best_cv = profit_shift(re_best_cv)
    logger.info("Create list of same models done!")
    return rf, lr, gbr, knr, lasso, enr, dtr, ve_full, ve_1, ve_best, re_full, re_best, re_best_cv


if __name__ == '__main__':
    logger.info("~~~### plot section is now ACTIVE ###~~~")

    # single profit plot
    profit_output = create_folder(args.output, pc["profit_plot_dir"])
    dfs = plot_exe(args.input, profit_output)

    # multiple profit plot
    multiple_profit_plot(dfs, profit_output)
    logger.info("WELL DONE, plot main done!")
