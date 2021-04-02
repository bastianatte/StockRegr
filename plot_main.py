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
from sklearn.metrics import r2_score, mean_squared_error


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
            # print(df_name)

    profit_model_dict = create_list_of_profit_models(dfs_profit_list)

    dataframe_list = []
    dataframe_list_single = []
    dataframe_list_ens = []
    metrics_df = pd.DataFrame(columns=["metrics"])
    for item in profit_model_dict:
        logger.info("item= {}".format(item))
        tup = []
        for item2 in profit_model_dict[item]:
            tup.append((profit_model_dict[item][item2], item2))
        tup = asc_sort_tuple(tup)
        sorted_tup = profit_shift(tup)
        for itm in sorted_tup:
            logger.info("df shape {}, item: {}".format(itm[0].shape, itm[1]))
        mod_df_list, mod_st_df = make_profit_and_create_stat_dict(sorted_tup, output, item)
        dataframe_list += mod_df_list
        if "single" in item:
            dataframe_list_single += mod_df_list
        if "ens" in item:
            dataframe_list_ens += mod_df_list
        metrics_df = metrics_df.merge(mod_st_df, on='metrics', how='outer')

    metrics_df_to_plot = make_transpose(metrics_df)
    # print(metrics_df_to_plot)
    logger.info("n single models: {}, n ens models: {}".format(len(dataframe_list_single), len(dataframe_list_ens)))
    # important plot
    sorted_df_list = sorted(dataframe_list,
                            key= lambda x:float(x[0]['shifted_long_short_profit'].iloc[-1]),
                            reverse=True)
    sorted_df_list_ens = sorted(dataframe_list_ens,
                                key=lambda x: float(x[0]['shifted_long_short_profit'].iloc[-1]),
                                reverse=True)
    sorted_df_list_single = sorted(dataframe_list_single,
                                   key=lambda x: float(x[0]['shifted_long_short_profit'].iloc[-1]),
                                   reverse=True)
    make_profit_plot(sorted_df_list[:10], output, "total")
    make_profit_plot(sorted_df_list_single, output, 'total_base_model')
    make_profit_plot(sorted_df_list_ens[:10], output, 'total_ensemble')
    make_metrics_plot(metrics_df_to_plot, output)
    logger.info("Profit Plot Done!")


def make_transpose(df):
    df_t = df.T
    df_t = df_t.rename(columns={0: 'sharpe_ratio',
                                1: 'annual return',
                                2: 'mean return',
                                3: 'standard dev',
                                4: 'Sortino Ratio',
                                5: 'MaxDD'})
    df_t = df_t.drop(['metrics'])
    df_t = df_t.sort_values(by=['sharpe_ratio'], ascending=False)
    df_t = df_t.reset_index()
    df_t = df_t.rename(columns={'index': 'models'})
    return df_t


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
    return model_df, stat_df


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


def create_list_of_profit_models(df_list):
    model_dict = {}
    model_lst = []
    for item in df_list:
        df_name = item[1]
        df_name = ''.join([i for i in df_name if not i.isdigit()])
        df_name = df_name.replace('profit_', '').replace('--', '')
        model_lst.append(df_name)
    models_names = set(model_lst)
    for name in models_names:
        model_dict[name] = {}
        for item in df_list:
            df = item[0]
            df_name = item[1]
            if name in df_name:
                model_dict[name][df_name] = df
    logger.info("dictionary keys: {}".format(model_dict.keys()))
    return model_dict


def make_score(df_list, output):
    """
    Make global plot long short profit.
    :param df_list: list of pd dfs
    :param output: output path
    :return: None
    """
    logger.info("in score plots")
    for item in df_list:
        df = item[0]
        df_name = item[1]
        if pc['prediction'] in df_name:
            score_ens_df = pd.DataFrame(columns=['model', 'r2', 'mse'])
            score_single_df = pd.DataFrame(columns=['model', 'r2', 'mse'])
            for i in df.columns.to_list():
                if pc['prediction'] in i:
                    scr_r2 = r2_score(df[mc["label"]], df[i])
                    scr_mse = mean_squared_error(df[mc["label"]], df[i])
                    score_single_df = score_single_df.append(
                        {'model': i,
                         'r2': scr_r2,
                         'mse': scr_mse},
                        ignore_index=True
                    )
                if 'ens' in i:
                    ens_scr_r2 = r2_score(df[mc["label"]], df[i])
                    ens_scr_mse = mean_squared_error(df[mc["label"]], df[i])
                    score_ens_df = score_ens_df.append(
                        {'model': i,
                         'r2': ens_scr_r2,
                         'mse': ens_scr_mse},
                        ignore_index=True
                    )
            logger.info("df name: {}, single score df shape: {} ".format(df_name, score_single_df.shape))
            logger.info("df name: {}, ens score df shape: {} ".format(df_name, score_ens_df.shape))
            # single_name = str(df_name) + "_single"
            # ens_name = str(df_name) + "_ens"
            # score_ens_df = score_ens_df.sort_values(by=['r2'], ascending=False)
            # score_single_df = score_single_df.sort_values(by=['r2'], ascending=False)
            # plotter_sng = Plotter(score_single_df, single_name, output)
            # plotter_sng.plot_score_table()
            # plotter_ens = Plotter(score_ens_df, ens_name, output)
            # plotter_ens.plot_ens_score_table()
            plotter_sng = Plotter(score_single_df.sort_values(by=['r2'], ascending=False),
                                  str(df_name) + "_single",
                                  output)
            plotter_sng.plot_score_table()
            plotter_ens = Plotter(score_ens_df.sort_values(by=['r2'], ascending=False),
                                  str(df_name) + "_ens",
                                  output)
            plotter_ens.plot_ens_score_table()


if __name__ == '__main__':
    logger.info("~~~### plot section is now ACTIVE ###~~~")

    # create output folder
    profit_output = create_folder(args.output, pc["profit_plot_dir"])
    score_output = create_folder(args.output, pc['score_dir'])

    # single profit plot
    dfs = plot_exe(args.input, profit_output)

    # multiple profit plot
    multiple_profit_plot(dfs, profit_output)

    # score plots
    make_score(dfs, score_output)

    logger.info("WELL DONE, plot_main.py successfully run!")
