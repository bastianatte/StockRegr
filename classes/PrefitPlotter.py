import logging
import os
from utils.misc import get_logger
import matplotlib.pyplot as plt

prefit_plotter_log = get_logger(__name__)
prefit_plotter_log.setLevel(logging.INFO)


class prefit_plotter(object):
    def __init__(self, df, output, csv_name):
        self.df = df
        self.output = output
        self.csv_name = csv_name

    def prefit_plotter_exe(self):
        prefit_plotter_log.info("Create features plots for {} stock".format(self.csv_name))
        self.plot_close_vs_open()
        self.plot_close()
        self.plot_high_vs_low()
        self.plot_ema_vs_close()
        self.plot_ema()
        self.plot_adr()
        self.plot_andr()
        self.plot_volume()
        self.plot_stoch()
        self.plot_proc()
        self.plot_hist_proc()
        self.plot_rsi()
        self.plot_hist_rsi()
        self.plot_macd()
        self.plot_acc_dist_oscill()
        self.plot_williams()
        self.plot_hist_williams()
        self.plot_disp()
        self.plot_hist_disp()
        self.plot_corr_matrix()

    def plot_close(self):
        # self.df.set_index('Date', inplace=True)
        plt.plot(self.df.index, self.df["Close"], "", linewidth=0.2, label=self.df['ticker'].iloc[-1])
        plt.legend(loc='upper left', frameon=False)
        # plt.rcParams["figure.figsize"] = (140, 0.1)
        # plt.figure(figsize=(200, 10))
        plt.xlabel("year")
        plt.ylabel("price")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_close" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_close_vs_open(self):
        self.df.set_index('Date', inplace=True)
        plt.plot(self.df.index, self.df["Open"], "", linewidth=0.2, label='Open')
        plt.plot(self.df.index, self.df["Close"], "", linewidth=0.2, label="Close")
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        plt.ylabel("price")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_close_vs_open" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_ema_vs_close(self):
        plt.plot(self.df.index, self.df["ta_ema"], "--", linewidth=0.2, label='ta_ema')
        plt.plot(self.df.index, self.df["Close"], "--", linewidth=0.2, label="Close")
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        plt.ylabel("price")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_ema_vs_close" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_high_vs_low(self):
        plt.plot(self.df.index, self.df["High"], "--", linewidth=0.2, label='High')
        plt.plot(self.df.index, self.df["Low"], "--", linewidth=0.2, label="Low")
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        plt.ylabel("price")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_high_vs_low" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_ema(self):
        plt.plot(self.df.index, self.df["ta_ema"], linewidth=0.5, label='ta_ema')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        plt.ylabel("ema")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_ema" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_volume(self):
        plt.plot(self.df.index, self.df["Volume"], linewidth=0.5, label='volume')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        plt.ylabel("Volume")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_volume" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_adr(self):
        plt.plot(self.df.index, self.df["lagged_daily_rt"], linewidth=0.5, label='adr')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        plt.ylabel("adr")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_lagged_daily_rt" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_andr(self):
        plt.plot(self.df.index, self.df["lagged_next_day_rt"], linewidth=0.5, label='andr')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        plt.ylabel("andr")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_lagged_next_day_rt" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_stoch(self):
        plt.hist(self.df["stoch"], bins=200, label='ta_stoch')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("stochastic values")
        plt.ylabel("counts")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_stoch" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_proc(self):
        plt.plot(self.df.index, self.df["ta_proc"], linewidth=0.2, label='ta_ROC')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_ROC" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_hist_proc(self):
        plt.hist(self.df["ta_proc"], label='ta_ROC', bins=200, alpha=0.3)
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("values")
        plt.ylabel("counts")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_hist_ROC" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_rsi(self):
        plt.plot(self.df.index, self.df["ta_rsi"], linewidth=0.2, label='RSI')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_rsi" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_hist_rsi(self):
        plt.hist(self.df["ta_rsi"], bins=400, label='RSI')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("values")
        plt.ylabel("counts")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_hist_rsi" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_macd(self):
        plt.plot(self.df.index, self.df["macd"], linewidth=0.2, label='macd')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        # plt.ylabel("andr")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_macd" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_williams(self):
        plt.plot(self.df.index, self.df["will_r_ind"], linewidth=0.2, label='will %R')
        plt.legend(loc='upper left', frameon=False)
        plt.ylabel("will")
        plt.xlabel("year")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_williams" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_hist_williams(self):
        plt.hist(self.df["will_r_ind"], bins=400, label='will %R')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("values")
        plt.ylabel("counts")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_hist_williams" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_disp(self):
        plt.plot(self.df.index, self.df["disp_5"], linewidth=0.2, label='disp_5')
        plt.plot(self.df.index, self.df["disp_10"], linewidth=0.2, label='disp_10')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_disparity" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_hist_disp(self):
        plt.hist(self.df["disp_5"], bins=400, alpha=0.6, label='disp_5')
        plt.hist(self.df["disp_10"], bins=400, alpha=0.6, label='disp_10')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("values")
        plt.ylabel("counts")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_hist_disparity" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_acc_dist_oscill(self):
        plt.plot(self.df.index, self.df["adi"], linewidth=0.2, label='ta_adi')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_acc_dist_oscill" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    def plot_corr_matrix(self):
        f = plt.figure(figsize=(19, 15))
        plt.matshow(self.df.corr(), fignum=f.number)
        plt.xticks(range(self.df.select_dtypes(['number']).shape[1]),
                   self.df.select_dtypes(['number']).columns,
                   fontsize=14, rotation=90)
        plt.yticks(range(self.df.select_dtypes(['number']).shape[1]),
                   self.df.select_dtypes(['number']).columns, fontsize=14)
        cb = plt.colorbar()
        cb.ax.tick_params(labelsize=14)
        plt.title('Correlation Matrix', fontsize=16)

        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_corr_matrix" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()

    # def plot_stoch_fast_slow(self):
    #     plt.hist(self.df["k_fast"], bins=200, label='k_fast', alpha=0.5)
    #     plt.hist(self.df["k_slow"], bins=200, label='k_slow', alpha=0.5)
    #     plt.hist(self.df["d_fast"], bins=200, label='d_fast', alpha=0.5)
    #     plt.hist(self.df["d_slow"], bins=200, label='d_slow', alpha=0.5)
    #     plt.legend(loc='upper left', frameon=False)
    #     plt.ylim(0, 70)
    #     plt.xlabel("stochastic values")
    #     plt.ylabel("counts")
    #     figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_stoch_fast_slow" + ".png")
    #     plt.savefig(figname, dpi=200)
    #     plt.close()
    #     prefit_plotter_log.info("plot stochastic fast slow done!")

    # def plot_acc_dist_oscill_2(self):
    #     plt.plot(self.df.index, self.df["adi_2"], linewidth=0.2, label='adi')
    #     plt.legend(loc='upper left', frameon=False)
    #     plt.xlabel("year")
    #     figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_acc_dist_oscill2" + ".png")
    #     plt.savefig(figname, dpi=200)
    #     plt.close()
    #     prefit_plotter_log.info("plot adi2 done!")
