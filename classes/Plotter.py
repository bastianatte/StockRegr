import logging
import os
from utils.misc import get_logger
import matplotlib.pyplot as plt

plotter_log = get_logger(__name__)
plotter_log.setLevel(logging.INFO)


class Plotter(object):
    def __init__(self, df, df_name, output):
        self.df = df
        self.df_name = df_name
        self.output = output

    def plot_profit_template(self):
        self.plot_profit()
        self.plot_cum_profit()

    def plot_profit(self):
        plt.plot(self.df.index,
                 self.df["daily_profit"],
                 label=self.df_name)
        plt.xticks(rotation=45, fontsize=6)
        plt.yticks(fontsize=6)
        plt.legend(loc='upper left', frameon=False)
        figname = os.path.join(self.output, "daily_profit" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_cum_profit(self):
        plt.plot(self.df.index,
                 self.df["long_short_profit"],
                 label=self.df_name)
        plt.xticks(rotation=45, fontsize=6)
        plt.yticks(fontsize=6)
        figname = os.path.join(self.output, "cum_profit" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()


