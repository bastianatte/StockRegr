import argparse
import fnmatch
import logging
import os
import pandas as pd
from config import main_conf as mc
import time
from classes.DFSelection import preprocess_df
from datetime import timedelta
from utils.misc import get_logger, create_folder, csv_maker, store_csv
from utils.ranking import rank_exe
from classes.RegressorModels import RegressorModels
from classes.PrefitPlotter import prefit_plotter
from sklearn.metrics import mean_squared_error

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


def create_df(ipt, opt):
    """
    Create a general full_df by looping over
    csv files located by input flag.
    :param ipt: args.input
    :param opt: args.output
    :return: pandas dataframe
    """
    csv_cnt = 0
    dfs = []
    folder = create_folder(opt, "prefit_plot")
    for csv_file in os.listdir(ipt):
        if fnmatch.fnmatch(csv_file, '*.csv'):
            if csv_cnt == 50:
                break
            else:
                csv_cnt += 1
                if csv_cnt % 50 == 0:
                    logger.info("Reading csv... {} stocks done!".format(csv_cnt))
                csv, csv_string = csv_maker(str(ipt), csv_file)
                df_temp = pd.read_csv(csv, parse_dates=[mc["date_clm"]])
                df_temp, scaled_df_temp = preprocess_df(df_temp, csv_string)
                if csv_cnt <= 3:
                    stock_output = create_folder(folder, csv_string)
                    logger.info("csv folder: {}".format(stock_output))
                    if mc["prefit_plot_flag"] is True:
                        prefplot = prefit_plotter(df_temp, stock_output, csv_string)
                        prefplot.prefit_plotter_exe()
                # dfs.append(df_temp)
                dfs.append(scaled_df_temp)
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
    # x = df_splitted[mc["features_TI"]]
    # x = df_splitted[mc["features_LR"]]
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


def create_prediction(df_test, xtrain, ytrain, xtest):
    """
    create prediction for each regression model
    :param df_test: test dataframe
    :param xtrain: X train
    :param ytrain: Y train
    :param xtest: X test
    :return: pred df
    """

    model = RegressorModels(xtrain, ytrain, xtest)
    # single models
    # lasso_mod, pred_lasso = model.lasso_regr()
    # enr_mod, pred_enr = model.elastic_net_regr()
    lr_mod, pred_lr = model.linear_regr()

    # rf_mod, pred_rf = model.random_forest_regr()
    rf_mod, pred_rf = model.random_forest_regr_tun()

    gbr_mod, pred_gbr = model.gradient_boost_regr()
    # gbr_mod, pred_gbr = model.gradient_boost_regr_tun()

    # knr_mod, pred_knr = model.kneighbors_regr()
    knr_mod, pred_knr = model.kneighbors_regr_tun()

    # dtr_mod, pred_dtr = model.decis_tree_regr()
    dtr_mod, pred_dtr = model.decision_tree_regr_tun()

    pred = df_test.copy()

    # ensemble
    # for idx, predictions in model.fitpred_ensemble():
    #     pred[idx] = predictions
    # pred["grid_ens"] = pred_grid_ens

    # single models
    # pred[mc["lasso"]] = pred_lasso
    # pred[mc["enr"]] = pred_enr
    pred[mc["lr"]] = pred_lr
    pred[mc["rf"]] = pred_rf
    pred[mc["gbr"]] = pred_gbr
    pred[mc["knr"]] = pred_knr
    pred[mc["dtr"]] = pred_dtr

    # best model
    pred = select_best_model_per_stock(pred)
    return pred


def create_test_df(train_end):
    """
    Calcuate test windows for the walkforward
    procedure.
    :param train_end: Training end date
    :return: Test start date, test end date
    """
    test_start_date = train_end + timedelta(days=1)
    test_end_date = test_start_date + timedelta(days=365)
    return test_start_date, test_end_date


def create_rank_and_store(df, model_clm, out_path, y_start, y_end, model_name):
    """
    It perform the rank and store csvs for long part, short part and profit.
    :param df: pandas dataframe
    :param model_clm: model
    :param out_path: output path
    :param y_start: begin year
    :param y_end: end year
    :param model_name: model name
    :return: None
    """
    df_long, df_short, df_profit = rank_exe(df, model_clm)
    store_csv(df_long, out_path, f"{model_name}_long-{y_start}-{y_end}")
    store_csv(df_short, out_path, f"{model_name}_short-{y_start}-{y_end}")
    store_csv(df_profit, out_path, f"profit_{model_name}-{y_start}-{y_end}")


def select_best_model_per_stock(preddf):
    """
    It select the best model by comparing the mean
    squared error between all of them.
    It is done for each single stock.
    An updated version, with a new columns with the best
    prediction, of the input prediction
    dataframe is returned.
    :param preddf: prediction dataframe
    :return: prediction dataframe
    """
    prediction_df = preddf.copy()
    score_df = pd.DataFrame(columns=['model', 'mse'])
    df_label = prediction_df.iloc[:, lambda df: df.columns.str.contains(mc['label'], case=False)]
    df_prd = prediction_df.iloc[:, lambda df: df.columns.str.contains('ens|pred', case=False)]
    for clm_name in df_prd.columns:
        mse = mean_squared_error(df_label[mc['label']], df_prd[clm_name])
        score_df = score_df.append(
            {'model': clm_name,
             'mse': mse},
            ignore_index=True
        )
    score_df = score_df.sort_values(by=['mse'], ascending=True)
    # print(score_df.head(2))
    # print(type(score_df['model'].iloc[0]), score_df['model'].iloc[0])
    logger.info("best model found: {}".format(score_df['model'].iloc[0]))
    best_model_name = score_df['model'].iloc[0]
    preddf['best_pred'] = prediction_df[best_model_name]
    return preddf


if __name__ == '__main__':
    logger.info("in main")
    df_pred_list = []
    wnd_cnt = 1
    dataframe = create_df(args.input, args.output)
    dataframe = dataframe.dropna()
    path_csv = create_folder(args.output, "csv")
    path_train_csv = create_folder(path_csv, "Train_Features")
    logger.info("dataframe columns: {}".format(dataframe.columns))
    logger.info("features columns: {}".format(mc["features"]))
    logger.info("{} train windows.".format(len(mc["train_window"])))
    start = time.time()
    for window in mc["train_window"]:
        stock_cnt = 0
        df_pred_list.clear()
        window_start = time.time()
        bad_df_list = []
        y_from = window[0].year
        y_to = window[1].year
        test_start, test_end = create_test_df(window[1])
        test_window = test_end.year - test_start.year
        thresh_raw = make_threshold_lenght(y_from, y_to, test_window)
        logger.info("#### Time windows #{} ####".format(wnd_cnt))
        logger.info("Train from {} to {} #### Test from {} to {}".format(y_from, y_to,
                                                                         test_start.year, test_end.year))
        wnd_cnt += 1
        for stock in set(dataframe[mc["ticker"]].values):
            stock_cnt += 1
            if stock_cnt % 1 == 0:
                logger.info("{} stocks fitted.".format(stock_cnt))
            df_stock_train, x_train, y_train = split_df(dataframe, stock, date_start=window[0], date_end=window[1])
            df_stock_test, x_test, y_test = split_df(dataframe, stock, test_start, test_end)
            if (df_stock_train.shape[0] == 0) or (len(df_stock_train) + len(df_stock_test) < thresh_raw):
                bad_df_list.append(stock)
                continue
            df_pred = create_prediction(df_stock_test, x_train, y_train, x_test)
            df_pred_list.append(df_pred)
            store_csv(x_train, path_train_csv, f"train-{stock}-{y_from}-{y_to}")
        stop = time.time()
        logger.info("{} bad stocks over {}".format(
            len(bad_df_list),
            len(set(dataframe[mc["ticker"]].values))))
        logger.info("Total run time: {} min, model run time: {} min".format(
            (stop-start)/60,
            (stop-window_start)/60))
        # concatenation to a single pred df
        dataframe_pred = pd.concat(df_pred_list)
        # store main csv
        store_csv(dataframe, path_csv, f"general-{y_from}-{y_to}")
        store_csv(dataframe_pred, path_csv, f"pred-{y_from}-{y_to}")
        # create rank for each model and store relevant quantities

        list_clm = dataframe_pred.columns.tolist()
        # print(list_clm)
        for i in range(len(list_clm)):
            if "ens" in list_clm[i]:
                # print(list_clm[i])
                create_rank_and_store(dataframe_pred, list_clm[i], path_csv, y_from, y_to, list_clm[i])
        # single models
        create_rank_and_store(dataframe_pred, mc["best"], path_csv, y_from, y_to, "best_single")
        create_rank_and_store(dataframe_pred, mc["rf"], path_csv, y_from, y_to, "rf_single")
        create_rank_and_store(dataframe_pred, mc["lr"], path_csv, y_from, y_to, "lr_single")
        create_rank_and_store(dataframe_pred, mc["gbr"], path_csv, y_from, y_to, "gbr_single")
        create_rank_and_store(dataframe_pred, mc["knr"], path_csv, y_from, y_to, "knr_single")
        # create_rank_and_store(dataframe_pred, mc["lasso"], path_csv, y_from, y_to, "lasso_single")
        # create_rank_and_store(dataframe_pred, mc["enr"], path_csv, y_from, y_to, "enr_single")
        create_rank_and_store(dataframe_pred, mc["dtr"], path_csv, y_from, y_to, "dtr_single")
        logger.info("Main Analysis Done")
