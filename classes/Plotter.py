import logging
import os
from utils.misc import get_logger
import matplotlib.pyplot as plt

plotter_log = get_logger(__name__)
plotter_log.setLevel(logging.INFO)


class Plotter(object):
    def __init__(self, df, output):
        self.df = df
        self.output = output

    def plot_open(self):
        plot = self.df["Open"].hist()
        plot.get_figure()
        figname = os.path.join(self.output, "histo" + ".png")
        plt.savefig(figname)
