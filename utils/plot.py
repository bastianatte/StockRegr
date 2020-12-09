from utils.misc import get_logger
import matplotlib.pyplot as plt
import os
import logging

plotter_log = get_logger(__name__)
plotter_log.setLevel(logging.INFO)


def make_multiple_profit_plot(df_list, output):
    for item in df_list:
        df = item[0]
        df_name = item[1]
        plt.plot(df.index,
                 df["long_short_profit"], '-',
                 label=df_name, marker="")
        plt.xticks(rotation=45, fontsize=6)
        plt.yticks(fontsize=6)
    plt.legend(loc='upper left', fontsize='xx-small', frameon=False)
    figname = os.path.join(output, "total_daily_profit" + ".png")
    plt.savefig(figname, dpi=200)
    plt.close()

