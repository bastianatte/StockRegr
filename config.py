#!/usr/bin/env python
import datetime as dt

main_conf = {
    "prefit_plot_flag": False,
    "columns": ["Date", "Close", "Open", "Volume", "High", "Low"],
    "train_window": [
        (dt.datetime.strptime("2000-01-01", "%Y-%m-%d"), dt.datetime.strptime("2004-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2001-01-01", "%Y-%m-%d"), dt.datetime.strptime("2005-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2002-01-01", "%Y-%m-%d"), dt.datetime.strptime("2006-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2003-01-01", "%Y-%m-%d"), dt.datetime.strptime("2007-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2004-01-01", "%Y-%m-%d"), dt.datetime.strptime("2008-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2005-01-01", "%Y-%m-%d"), dt.datetime.strptime("2009-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2006-01-01", "%Y-%m-%d"), dt.datetime.strptime("2010-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2007-01-01", "%Y-%m-%d"), dt.datetime.strptime("2011-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2008-01-01", "%Y-%m-%d"), dt.datetime.strptime("2012-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2009-01-01", "%Y-%m-%d"), dt.datetime.strptime("2013-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2010-01-01", "%Y-%m-%d"), dt.datetime.strptime("2014-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2011-01-01", "%Y-%m-%d"), dt.datetime.strptime("2015-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2012-01-01", "%Y-%m-%d"), dt.datetime.strptime("2016-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2013-01-01", "%Y-%m-%d"), dt.datetime.strptime("2017-01-01", "%Y-%m-%d")),
        (dt.datetime.strptime("2014-01-01", "%Y-%m-%d"), dt.datetime.strptime("2018-01-01", "%Y-%m-%d")),
    ],
    "rfr": 0.05,
    "ticker": "ticker",
    "date_clm": "Date",
    "lagged_daily_rt": "lagged_daily_rt",
    "label": "lagged_next_day_rt",
    "label_1": "lagged_daily_rt_1",
    "label_2": "lagged_daily_rt_2",
    "label_3": "lagged_daily_rt_3",
    "label_4": "lagged_daily_rt_4",
    "label_5": "lagged_daily_rt_5",
    "label_6": "lagged_daily_rt_6",
    "label_7": "lagged_daily_rt_7",
    "label_8": "lagged_daily_rt_8",
    "label_9": "lagged_daily_rt_9",
    "label_10": "lagged_daily_rt_10",
    "volatility": "volatility",
    "daily_return": "daily_return",
    "log_return": "log_rt",
    "sharpe_ratio": "sharpe_ratio",
    "rf": "rf_pred",
    "lr": "lr_pred",
    "gbr": "gbr_pred",
    "knr": "knr_pred",
    "lasso": "lasso_pred",
    "enr": "enr_pred",
    "dtr": "dtr_pred",
    "vot_ens": "vot_ens_pred",
    "ensemble": "ensemble_pred",
    "ensemble2": "ensemble2_pred",
    "features": ["Close", "Open", "Volume", "High", "Low",
                 "lagged_daily_rt_1", "lagged_daily_rt_2", "lagged_daily_rt_3",
                 "lagged_daily_rt_4", "lagged_daily_rt_4", "lagged_daily_rt_5",
                 "lagged_daily_rt_6", "lagged_daily_rt_7", "lagged_daily_rt_8",
                 "lagged_daily_rt_9", "ta_ema", "stoch", "ta_proc", "ta_rsi",
                 "macd", "will_r_ind", "disp_5", "disp_10", "adi"],
    "daily_profit_clm": "daily_profit",
    "ls_profit_clm": "long_short_profit",
    "ema_period": 10,
    "proc_period": 14,
    "stochastic_high": 14,
    "stochastic_low": 3,
    "macd_slow": 26,
    "macd_fast": 12,
    "macd_sign": 9,
    "rsi_wind": 14,
}

ranking_conf = {
    "stocks_number": 5,
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
    "profit_plot_dir": "profit_plot",
    "prefit_plot_dir": "prefit_plot",
}

regress_models_conf = {
    "rf_max_features": 1,
}
