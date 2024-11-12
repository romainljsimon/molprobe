import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import extrapol as ex
import colors as col


class AutoCorrelationFile:
    def __init__(self, path_corr_file='', temp=None, onset_time=1,  start=0, end=0, extrapolation=False):
        self.path_corr_file = path_corr_file
        self.onset_time = onset_time
        self.start = start
        self.end = end
        self.extrapolation = extrapolation
        self.temp = temp
        self.corr_array, self.t_array = np.array([]), np.array([])
        self.prepare_file()
        if extrapolation:
            self.calculate_extrapolation()

    def prepare_file(self):
        data = np.genfromtxt(self.path_corr_file)
        mask = np.any(np.isnan(data), axis=1)
        data = data[~mask]
        idx = np.argsort(data[:, 0])
        data = data[idx]
        self.end = len(data) - self.end
        self.t_array = data[self.start:self.end , 0] / self.onset_time
        self.corr_array = data[self.start:self.end , 1]
        

    def plot_time_corr(self, ax, marker='^', ms=5, ylabel=''):
        ax.plot(self.t_array, self.corr_array, marker=marker, ms=ms, label=str(self.temp))
        ax.plot(self.t_array, np.full(len(self.t_array), np.exp(-1)), c='black')
        ax.set_xlabel(r'$t$')
        ax.set_ylabel(ylabel)

    def get_relaxation_time(self, crit=np.exp(-1)):
        index = 0
        for j, elt in enumerate(self.corr_array):
            if elt < crit:
                index = j
                break
        
        if index != 0:
            x1, x2 = self.t_array[index], self.t_array[index-1]
            y1, y2 = self.corr_array[index], self.corr_array[index-1]
            a = (y2 - y1) / np.log10(x2/x1)
            b = y2 - a * np.log10(x2)
            t_relax = 10**((crit - b) / a)
        else:
            t_relax = np.nan
        return t_relax