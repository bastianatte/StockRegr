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
        self.plot_andr()
        self.plot_volume()
        self.plot_stoch()
        self.plot_stoch_fast_slow()
        self.plot_proc()
        self.plot_hist_proc()
        self.plot_rsi()
        self.plot_macd()
        self.plot_acc_dist_oscill()
        self.plot_acc_dist_oscill_2()
        self.plot_williams()
        self.plot_hist_williams()
        self.plot_disp()

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
        prefit_plotter_log.info("plot Close vs Open done!")

    def plot_ema_vs_close(self):
        plt.plot(self.df.index, self.df["ema"], "--", linewidth=0.2, label='ema')
        plt.plot(self.df.index, self.df["ta_ema"], "--", linewidth=0.2, label='ta_ema')
        plt.plot(self.df.index, self.df["Close"], "--", linewidth=0.2, label="Close")
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        plt.ylabel("price")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_ema_vs_close" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        prefit_plotter_log.info("plot Close vs ema done!")

    def plot_high_vs_low(self):
        plt.plot(self.df.index, self.df["High"], "--", linewidth=0.2, label='High')
        plt.plot(self.df.index, self.df["Low"], "--", linewidth=0.2, label="Low")
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        plt.ylabel("price")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_high_vs_low" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        prefit_plotter_log.info("plot high vs low done!")

    def plot_ema(self):
        plt.plot(self.df.index, self.df["ema"], "--", linewidth=0.2, label='ema')
        plt.plot(self.df.index, self.df["ta_ema"], linewidth=0.5, label='ta_ema')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        plt.ylabel("ema")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_ema" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        prefit_plotter_log.info("plot ema done!")

    def plot_volume(self):
        plt.plot(self.df.index, self.df["Volume"], linewidth=0.5, label='volume')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        plt.ylabel("Volume")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_volume" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        prefit_plotter_log.info("plot Volume done!")

    def plot_adr(self):
        plt.plot(self.df.index, self.df["actual_day_rt"], linewidth=0.5, label='adr')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        plt.ylabel("adr")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_actual_day_rt" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        prefit_plotter_log.info("plot acd done!")

    def plot_andr(self):
        plt.plot(self.df.index, self.df["actual_next_day_rt"], linewidth=0.5, label='andr')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        plt.ylabel("andr")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_actual_next_day_rt" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        prefit_plotter_log.info("plot next adr done!")

    def plot_stoch(self):
        plt.hist(self.df["stoch"], bins=200, label='ta_stoch')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("stochastic values")
        plt.ylabel("counts")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_stoch" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        prefit_plotter_log.info("plot stochastic done!")

    def plot_stoch_fast_slow(self):
        plt.hist(self.df["k_fast"], bins=200, label='k_fast', alpha=0.5)
        plt.hist(self.df["k_slow"], bins=200, label='k_slow', alpha=0.5)
        plt.hist(self.df["d_fast"], bins=200, label='d_fast', alpha=0.5)
        plt.hist(self.df["d_slow"], bins=200, label='d_slow', alpha=0.5)
        plt.legend(loc='upper left', frameon=False)
        plt.ylim(0, 70)
        plt.xlabel("stochastic values")
        plt.ylabel("counts")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_stoch_fast_slow" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        prefit_plotter_log.info("plot stochastic fast slow done!")

    def plot_proc(self):
        plt.plot(self.df.index, self.df["proc"], linewidth=0.2, label='ROC')
        plt.plot(self.df.index, self.df["ta_proc"], linewidth=0.2, label='ta_ROC')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_ROC" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        prefit_plotter_log.info("plot ROC done!")

    def plot_hist_proc(self):
        plt.hist(self.df["proc"], label='ROC', bins=200, alpha=0.3)
        plt.hist(self.df["ta_proc"], label='ta_ROC', bins=200, alpha=0.3)
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("values")
        plt.ylabel("counts")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_hist_ROC" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        prefit_plotter_log.info("plot hist ROC done!")

    def plot_rsi(self):
        plt.plot(self.df.index, self.df["ta_rsi"], linewidth=0.2, label='RSI')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_rsi" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        prefit_plotter_log.info("plot rsi done!")

    def plot_macd(self):
        plt.plot(self.df.index, self.df["macd"], linewidth=0.2, label='macd')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        # plt.ylabel("andr")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_macd" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        prefit_plotter_log.info("plot macd done!")

    def plot_williams(self):
        plt.plot(self.df.index, self.df["will_r_ind"], linewidth=0.2, label='williams')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("will_r_index")
        plt.ylabel("will")
        plt.xlabel("year")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_williams" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        prefit_plotter_log.info("plot williams done!")

    def plot_hist_williams(self):
        plt.hist(self.df["will_r_ind"], bins=200)
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("will_r_index")
        plt.ylabel("counts")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_hist_williams" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        prefit_plotter_log.info("plot hist williams done!")

    def plot_disp(self):
        plt.plot(self.df.index, self.df["disp_5"], linewidth=0.2, label='disp_5')
        plt.plot(self.df.index, self.df["disp_10"], linewidth=0.2, label='disp_10')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_disparity" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        prefit_plotter_log.info("plot disparity done!")

    def plot_acc_dist_oscill(self):
        plt.plot(self.df.index, self.df["adi"], linewidth=0.2, label='ta_adi')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_acc_dist_oscill" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        prefit_plotter_log.info("plot adi done!")

    def plot_acc_dist_oscill_2(self):
        plt.plot(self.df.index, self.df["adi_2"], linewidth=0.2, label='adi')
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel("year")
        figname = os.path.join(self.output, self.df['ticker'].iloc[-1] + "_acc_dist_oscill2" + ".png")
        plt.savefig(figname, dpi=200)
        plt.close()
        prefit_plotter_log.info("plot adi2 done!")
