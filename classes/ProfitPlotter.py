from utils.misc import get_logger, disc_sort_tuple
import matplotlib.pyplot as plt
import numpy as np
from config import plot_conf as pc
import empyrical as emp
import pyfolio as pf
import os
import logging

plotter_log = get_logger(__name__)
plotter_log.setLevel(logging.INFO)


class ProfitPlotter(object):
    def __init__(self, df_list, plot_name, output):
        self.df_list = df_list
        self.plot_name = plot_name
        self.output = output

    def make_multiple_profit_plot(self):
        """
        it shows cumulative plots.
        :return: None
        """
        for item in self.df_list:
            df = item[0]
            df_name = item[1]
            plt.plot(df.index,
                     df["shifted_long_short_profit"], '-',
                     label=df_name, marker="")
            plt.xticks(rotation=60, fontsize=5)
            plt.yticks(fontsize=10)
        plt.legend(loc='upper left', fontsize='xx-small', frameon=False)
        plt.title("Cumulative Return")
        figname = os.path.join(self.output, self.plot_name + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def make_multiple_profit_total_plot(self):
        """
        it shows cumulative plots.
        :return: None
        """
        for item in self.df_list:
            df = item[0]
            df_name = item[1]
            if pc["rf_csv"] in str(df_name):
                color = "red"
            else:
                color = "green"
            plt.plot(df.index,
                     df["shifted_long_short_profit"], '-',
                     label=df_name, marker="", color=color)
            plt.xticks(rotation=60, fontsize=5)
            plt.yticks(fontsize=10)
        plt.legend(loc='upper left', fontsize='xx-small', frameon=False)
        plt.title("Cumulative Return")
        figname = os.path.join(self.output, self.plot_name + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def make_multiple_profit_plot_ranking(self):
        """
        It plot the cumulative profit obtained at the
        end of the test period.
        :return: None
        """
        tup = []
        for item in self.df_list:
            df = item[0]
            df_name = item[1]
            df_lsp = df.loc[df.index.max(), "long_short_profit"]
            tup.append((df_name, df_lsp))
        sorted_tup = disc_sort_tuple(tup)
        plt.figure(figsize=(5, 12))
        for item in sorted_tup:
            plt.bar(item[0], item[1], color='green')
        plt.xticks(rotation=70, fontsize=12)
        plt.yticks(fontsize=12)
        plt.title("Cumulative Return")
        figname = os.path.join(self.output, self.plot_name + "_c_ranking" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def make_return_tear_sheet_plot(self):
        for item in self.df_list:
            df = item[0]
            daily_profit = df["daily_profit"]
            tear_sheet_rt = pf.create_returns_tear_sheet(daily_profit, return_fig=True)
            figname = os.path.join(self.output, self.plot_name + "_tear_sheet_return" + ".png")
            tear_sheet_rt.savefig(figname, dpi=200)
            # Drawdown
            dd_plot = pf.plot_drawdown_periods(daily_profit, top=10)
            full_tear_sheet_rt_ax = pf.create_full_tear_sheet(daily_profit)
            # print(full_tear_sheet_rt_ax)
            ddfigname = os.path.join(self.output, self.plot_name + "_drawdown" + ".png")
            dd_plot.figure.savefig(ddfigname)

    def make_empirical(self):
        statistics_dict = {}
        for item in self.df_list:
            df = item[0]
            df_name = item[1]
            daily_profit = df["daily_profit"]
            statistics_dict['sharpe ratio'] = round(emp.sharpe_ratio(daily_profit), 4)
            # sharpe ratio
            plt.plot(statistics_dict["sharpe ratio"], '-', label=df_name, marker="")
            plt.title("sharpe ratio")
            sr_figname = os.path.join(self.output, self.plot_name + "_sharpe_ratio.png")
            plt.savefig(sr_figname, dpi=200)
            plt.close()
            # annual return
            statistics_dict["annual returns"] = round(emp.annual_return(daily_profit), 4)
            plt.plot(statistics_dict["annual returns"], '-', label=df_name, marker="")
            plt.title("annual return")
            ar_figname = os.path.join(self.output, self.plot_name + "_annual_ret.png")
            plt.savefig(ar_figname, dpi=200)
            plt.close()
            # mean return
            statistics_dict['mean returns'] = round(daily_profit.mean(), 4)
            plt.plot(statistics_dict["mean returns"], '-', label=df_name, marker="")
            plt.title("mean returns")
            ar_figname = os.path.join(self.output, self.plot_name + "_mean_ret.png")
            plt.savefig(ar_figname, dpi=200)
            plt.close()
            # standard dev p.a. ###### ALERT #######
            statistics_dict['Standard dev p.a.'] = round(emp.annual_volatility(daily_profit), 4)
            # plt.plot(statistics_dict["Standard dev p.a."], '-', label=df_name, marker="")
            # plt.title("Standard dev p.a.")
            # ar_figname = os.path.join(self.output, self.plot_name + "_standard_dev.png")
            # plt.savefig(ar_figname, dpi=200)
            # plt.close()
            # sortino ###### ALERT #######
            statistics_dict['Sortino Ratio'] = round(emp.sortino_ratio(daily_profit), 4)
            # plt.plot(statistics_dict["Sortino Ratio"], '-', label=df_name, marker="")
            # plt.title("Sortino Ratio")
            # ar_figname = os.path.join(self.output, self.plot_name + "_sortino_ratio.png")
            # plt.savefig(ar_figname, dpi=200)
            # plt.close()
            # max dd ###### ALERT #######
            statistics_dict['MaxDD'] = round(emp.max_drawdown(daily_profit), 4)
            # plt.plot(statistics_dict["MaxDD"], '-', label=df_name, marker="")
            # plt.title("MaxDD")
            # ar_figname = os.path.join(self.output, self.plot_name + "_maxdd.png")
            # plt.savefig(ar_figname, dpi=200)
            # plt.close()
        return statistics_dict


