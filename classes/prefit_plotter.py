import logging
import os
from utils.misc import get_logger
import matplotlib.pyplot as plt

prefit_plotter_log = get_logger(__name__)
prefit_plotter_log.setLevel(logging.INFO)


class prefit_plotter(object):
    def __init__(self, df, output):
        self.df = df
        self.output = output

    def prefit_plotter_exe(self):
        self.plot_close_vs_open()
        self.plot_high_vs_low()
        self.plot_ema_vs_close()
        self.plot_ema()
        self.plot_adr()
        self.plot_nadr()
        self.plot_volume()
        self.plot_k_fast()
        self.plot_k_slow()
        self.plot_d_fast()
        self.plot_d_slow()

    def plot_close_vs_open(self):
        # self.df.set_index('Date', inplace=True)
        # plt.plot(self.df.index, self.df["Open"], "", linewidth=0.5, label='Open')
        # plt.plot(self.df.index, self.df["Close"], "", linewidth=0.5, label="Close")
        # self.df.sort_index()
        plt.plot(self.df["Open"], linewidth=0.5, label='Open')
        plt.plot(self.df["Close"], linewidth=0.5, label="Close")
        plt.legend(loc='upper left', frameon=False)
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_close_vs_open" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_ema_vs_close(self):
        # self.df.set_index('Date', inplace=True)
        plt.plot(self.df.index, self.df["ema_10"], "--", linewidth=0.5, label='ema10')
        plt.plot(self.df.index, self.df["Close"], "--", linewidth=0.5, label="Close")
        plt.legend(loc='upper left', frameon=False)
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_ema_vs_close" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_high_vs_low(self):
        # self.df.set_index('Date', inplace=True)
        plt.plot(self.df.index, self.df["High"], "--", linewidth=0.5, label='High')
        plt.plot(self.df.index, self.df["Low"], "--", linewidth=0.5, label="Low")
        plt.legend(loc='upper left', frameon=False)
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_high_vs_low" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_ema(self):
        # self.df.set_index('Date', inplace=True)
        plt.plot(self.df.index, self.df["ema_10"], linewidth=0.5, label='ema10')
        plt.legend(loc='upper left', frameon=False)
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_ema" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_volume(self):
        # self.df.set_index('Date', inplace=True)
        plt.plot(self.df.index, self.df["Volume"], linewidth=0.5, label='volume')
        plt.legend(loc='upper left', frameon=False)
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_volume" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_adr(self):
        # self.df.set_index('Date', inplace=True)
        plt.plot(self.df.index, self.df["actual_day_rt"], linewidth=0.5, label='adr')
        plt.legend(loc='upper left', frameon=False)
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_actual_day_rt" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_nadr(self):
        # self.df.set_index('Date', inplace=True)
        plt.plot(self.df.index, self.df["actual_next_day_rt"], linewidth=0.5, label='adr')
        plt.legend(loc='upper left', frameon=False)
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_actual_next_day_rt" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_k_fast(self):
        # plt.plot(self.df.index, self.df["k_fast"], "--", linewidth=0.5, label='k_fast')
        plt.hist(self.df["k_fast"])
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_k_fast" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_k_slow(self):
        # plt.plot(self.df.index, self.df["k_slow"], linewidth=0.5, label='k_slow')
        plt.hist(self.df["k_slow"])
        # plt.legend(loc='upper left', frameon=False)
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_k_slow" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_d_fast(self):
        # plt.hist2d(self.df.index, self.df["d_fast"], linewidth=0.5, label='d_fast')
        plt.hist(self.df["d_fast"])
        # plt.legend(loc='upper left', frameon=False)
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_d_fast" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_d_slow(self):
        # plt.plot(self.df.index, self.df["d_slow"], linewidth=0.5, label='d_slow')
        plt.hist(self.df["d_slow"])
        # plt.legend(loc='upper left', frameon=False)
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_d_slow" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()


