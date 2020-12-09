from utils.misc import get_logger
import matplotlib.pyplot as plt
import os
import logging

plotter_log = get_logger(__name__)
plotter_log.setLevel(logging.INFO)


def make_multiple_profit_plot(df_list, plot_name, output):
    for item in df_list:
        df = item[0]
        df_name = item[1]
        plt.plot(df.index,
                 df["long_short_profit"], '-',
                 label=df_name, marker="")
        plt.xticks(rotation=60, fontsize=5)
        plt.yticks(fontsize=10)
    plt.legend(loc='upper left', fontsize='xx-small', frameon=False)
    plt.title("Cumulative plot")
    figname = os.path.join(output, plot_name + ".png")
    plt.savefig(figname, dpi=200)
    plt.close()


def make_multiple_profit_plot_ranking(df_list, output):
    df_name_list = []
    lsp_list = []
    for item in df_list:
        df = item[0]
        df_name = item[1]
        df_lsp = df.loc[df.index.max(), "long_short_profit"]
        df_name_list.append(df_name)
        lsp_list.append(df_lsp)
    plt.bar(df_name_list, lsp_list, color='green')
    # plt.xticks(rotation=45, fontsize=6)
    plt.xticks(rotation=60, fontsize=6)
    plt.yticks(fontsize=6)
    plt.title("Final cumulative")
    figname = os.path.join(output, "ranking" + ".png")
    plt.savefig(figname, dpi=200)
    plt.close()
