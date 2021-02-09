import logging
import os
from utils.misc import get_logger
import matplotlib.pyplot as plt
# import plotly.graph_objects as go
# import dataframe_image as dfi
from pandas.plotting import table

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
        """
        It plot daily profit for each model
        window used.
        :return: None
        """
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
        """
        It plot daily profit for each model
        window used.
        :return: None
        """
        plt.plot(self.df.index,
                 self.df["long_short_profit"],
                 label=self.df_name)
        plt.xticks(rotation=45, fontsize=6)
        plt.yticks(fontsize=6)
        figname = os.path.join(self.output, "cum_profit" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_metrics_table(self):
        """
        Makes metrics table plot for each regressor model.
        :return:None
        """
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        ax.axis('tight')
        tabla = ax.table(cellText=round(self.df.values, 4), colLabels=self.df.columns, loc='center')
        tabla.scale(1, 2)
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(6)
        fig.tight_layout()
        figname = os.path.join(self.output, "metrics_table" + ".png")
        plt.savefig(figname, dpi=400)
        plt.close()

    # def plot_metrics_table_2(self):
    #     print(self.df.head(5))
    #     print(self.df.columns, self.df.values)
    #     fig = go.Figure(data=[go.Table(header=dict(values=list(self.df.columns)), cells=dict(values=self.df.values))])
    #     # fig1 = go.Figure(data=[go.Table(header=dict(values=['A Scores', 'B Scores']),
    #     # cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]]))])
    #     figname = os.path.join(self.output, "metrics_table_2" + ".png")
    #     fig.write_image(figname)

    # def plot_metrics_table_3(self):
    #     figname = os.path.join(self.output, "metrics_table_3" + ".png")
    #     dfi.export(self.df, figname)
    #
    # def plot_metrics_table_4(self):
    #     print(self.df.head(5))
    #     fig, ax = plt.subplots(figsize=(14, 3))  # set size frame
    #     # df.index = [item.strftime('%Y-%m-%d') for item in df.index]
    #     ax.xaxis.set_visible(False)  # hide the x axis
    #     ax.yaxis.set_visible(False)  # hide the y axis
    #     ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
    #     tabla = table(ax, self.df, loc='upper right', colWidths=[0.17] * len(self.df.columns))
    #     # tabla.auto_set_font_size(False)  # Activate set fontsize manually
    #     # tabla.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
    #     # tabla.scale(1.2, 1.2)  # change size table
    #     # plt.savefig('table.png', transparent=True)
    #     figname = os.path.join(self.output, "metrics_table_4" + ".png")
    #     plt.savefig(figname, dpi=200)
    #     plt.close()


