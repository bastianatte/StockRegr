from utils.misc import get_logger, disc_sort_tuple
import matplotlib.pyplot as plt
from config import plot_conf as pc
import os
import datetime
import pandas as pd
import numpy as np
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
        plt.title("Cumulative plot")
        figname = os.path.join(self.output, self.plot_name + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def make_multiple_profit_total_plot(self):
        """
        it shows cumulative plots.
        :return: None
        """
        temp_list = []
        for item in self.df_list:
            df = item[0]
            df_name = item[1]
            if pc["rf_csv"] in str(df_name):
                color = "red"
                tag = "rf"
            else:
                color = "green"
                tag = "lr"
            plt.plot(df.index,
                     df["shifted_long_short_profit"], '-',
                     label=df_name, marker="", color=color)
            plt.xticks(rotation=60, fontsize=5)
            plt.yticks(fontsize=10)
        plt.legend(loc='upper left', fontsize='xx-small', frameon=False)
        plt.title("Total Cumulative plot")
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
        plt.title("Cumulative")
        figname = os.path.join(self.output, self.plot_name + "_ranking" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
