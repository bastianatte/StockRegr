from utils.misc import get_logger, disc_sort_tuple
import matplotlib.pyplot as plt
import os
import logging

plotter_log = get_logger(__name__)
plotter_log.setLevel(logging.INFO)


class profit_plotter(object):
    def __init__(self, df_list, plot_name, output):
        self.df_list = df_list
        self.plot_name = plot_name
        self.output = output

    def make_multiple_profit_plot(self):
        for item in self.df_list:
            df = item[0]
            df_name = item[1]
            plt.plot(df.index,
                     df["long_short_profit"], '-',
                     label=df_name, marker="")
            plt.xticks(rotation=60, fontsize=5)
            plt.yticks(fontsize=10)
        plt.legend(loc='upper left', fontsize='xx-small', frameon=False)
        plt.title("Cumulative plot")
        figname = os.path.join(self.output, self.plot_name + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def make_multiple_profit_plot_ranking(self):
        tup = []
        for item in self.df_list:
            df = item[0]
            df_name = item[1]
            df_lsp = df.loc[df.index.max(), "long_short_profit"]
            tup.append((df_name, df_lsp))
        sorted_tup = disc_sort_tuple(tup)
        plt.figure(figsize=(6, 8))
        for item in sorted_tup:
            plt.bar(item[0], item[1], color='green')
        plt.xticks(rotation=60, fontsize=6)
        plt.yticks(fontsize=6)
        plt.title("Final cumulative")
        figname = os.path.join(self.output, "ranking" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
