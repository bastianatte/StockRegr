#!/usr/bin/env python
import datetime as dt

main_conf = {
    "columns": ["Date", "Close", "Open", "Volume", "High", "Low"],
    "train_window": [
        (dt.datetime.strptime("2000-01-01", "%Y-%m-%d"), dt.datetime.strptime("2010-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2001-01-01", "%Y-%m-%d"), dt.datetime.strptime("2011-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2002-01-01", "%Y-%m-%d"), dt.datetime.strptime("2012-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2003-01-01", "%Y-%m-%d"), dt.datetime.strptime("2013-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2004-01-01", "%Y-%m-%d"), dt.datetime.strptime("2014-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2005-01-01", "%Y-%m-%d"), dt.datetime.strptime("2015-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2006-01-01", "%Y-%m-%d"), dt.datetime.strptime("2016-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2007-01-01", "%Y-%m-%d"), dt.datetime.strptime("2017-01-01", "%Y-%m-%d")),
    ],
    "ticker": "ticker",
    "date_clm": "Date",
    "rf_clm_name": "rf_pred_next_day_rt",
    "lr_clm_name": "lr_pred_next_day_rt",
    "actual_day_rt_clm": "actual_day_rt",
    "features": ["Close", "Open", "Volume", "High", "Low"],
    "label": "actual_next_day_rt",
    "daily_profit_clm": "daily_profit",
    "ls_profit_clm": "long_short_profit",
}

ranking_conf = {
    "stocks_number": 10,
    "long_share": 0.5,
    "short_share": 0.5,
    "rank_df": ["Date", "long_short_profit", "daily_profit"],
}

plot_conf = {
    "general_csv": "general",
    "pred_csv": "pred",
    "profit_csv": "profit",
    "rf_csv": "rf",
    "lr_csv": "lr",
    "long_csv": "long",
    "short_csv": "short",
    "daily_profit": "daily_profit",
    "plot_dir": "plot",
}
