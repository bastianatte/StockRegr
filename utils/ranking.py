from utils.misc import get_logger
from config import main_conf as mc
from config import ranking_conf as rc
import pandas as pd
import logging

rank_log = get_logger(__name__)
rank_log.setLevel(logging.INFO)


def rank_exe(df, pred_column):
    lsp = 0
    i = 0
    df_daily_profit = pd.DataFrame(columns=[rc["rank_df"]])
    dfs_long = []
    dfs_short = []
    df_rank = df.sort_values([mc["date_clm"], pred_column], ascending=[True, False])
    date_list = list(df_rank.Date.unique())
    date_list.sort()
    for date in date_list[:-1]:
        i += 1
        data_date = df_rank[df_rank.Date == date]
        data_date = data_date.drop_duplicates(subset=[mc["ticker"]])  # is that useful?
        # data_date = data_date.sort_values([mc["label"]], ascending=[False])
        long_part = data_date.iloc[:rc["stocks_number"]]
        short_part = data_date.iloc[-rc["stocks_number"]:]
        dp, lsp = long_short_profit(long_part, short_part, lsp)
        df_daily_profit.loc[i] = [date, lsp, dp]
        print(data_date)
        print("long: ", pred_column, "\n", long_part[["Date", pred_column, mc["label"]]].head(5))
        print("short: ", pred_column, "\n", short_part[["Date", pred_column, mc["label"]]].head(5))
        dfs_long.append(long_part)
        dfs_short.append(short_part)
    long = pd.concat(dfs_long)
    short = pd.concat(dfs_short)
    return long, short, df_daily_profit


def long_short_profit(long, short, lsp):
    """
    Calculate long short daily profit.
    :param long: long df
    :param short: short df
    :param lsp: long short profit
    :return: long df, short df, lsp
    """
    long_daily_profit = (long[mc["label"]]).mean()
    short_daily_profit = ((-1) * short[mc["label"]]).mean()
    daily_profit = rc["long_share"] * long_daily_profit + rc["short_share"] * short_daily_profit
    lsp += daily_profit
    return daily_profit, lsp
