#!/usr/bin/env python
import datetime as dt

main_conf = {
    "columns": ["Date", "Close", "Open", "Volume", "High", "Low"],
    "train_start": dt.datetime.strptime("2005-01-01", "%Y-%m-%d"),
    "train_end": dt.datetime.strptime("2015-01-01", "%Y-%m-%d"),
    "test_start": dt.datetime.strptime("2015-03-01", "%Y-%m-%d"),
    "test_end": dt.datetime.strptime("2018-02-01", "%Y-%m-%d"),
    "ticker": "ticker",
    "date_clm": "Date",
    "rf_clm_name": "rf_pred_next_day_rt",
    "lr_clm_name": "lr_pred_next_day_rt",
    "actual_day_rt_clm": "actual_day_rt",
    "features": ["Close", "Open"],
    "label": "actual_next_day_rt",
    "daily_profit_clm": "daily_profit",
    "ls_profit_clm": "long_short_profit",
}

ranking_conf = {
    "stocks_number": 5,
    "long_share": 0.5,
    "short_share": 0.5,
    "rank_df": ["Date", "long_short_profit", "daily_profit"],
}
