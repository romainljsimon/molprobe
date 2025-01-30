import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import molprobe.fit_functions as ex
import molprobe.colors as col


class AutoCorrelationFile:
    def __init__(self, path_corr_file='', n_particles=None, temp=None, onset_time=1,  start=0, end=0, extrapolation=False):
        self.path_corr_file = path_corr_file
        self.onset_time = onset_time
        self.start = start
        self.end = end
        self.extrapolation = extrapolation
        self.temp = temp
        self.corr_array, self.chi4_array, self.self_array, self.t_array = np.array([]), np.array([]), np.array([]), np.array([])
        self.n_particles = n_particles
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
        if len(data.T) > 2:
            self.chi4_array = self.n_particles * (data[self.start:self.end, 2] - self.corr_array**2)
        if len(data.T) > 3:
            self.self_array = (data[self.start:self.end, 3] - self.corr_array**2)

    def plot_time_corr(self, ax, marker='^', ms=5, xlabel=r'$t$', ylabel='',color='blue', linewidth=1, xlabelpad=1, ylabelpad=1):
        ax.plot(self.t_array, self.corr_array, marker=marker, ms=ms, label=str(self.temp), linewidth=linewidth, color=color, zorder=0)
        #ax.plot(self.t_array, np.full(len(self.t_array), np.exp(-1)), c='black')
        ax.set_xlabel(xlabel, labelpad=xlabelpad)
        ax.set_ylabel(ylabel, labelpad=ylabelpad)
    
    def plot_time_chi4(self, ax, marker='^', ms=5, xlabel=r'$t$', ylabel='',color='blue', linewidth=1, xlabelpad=1, ylabelpad=1):
        ax.plot(self.t_array, self.chi4_array, marker=marker, ms=ms, label=str(self.temp), linewidth=linewidth, color=color, zorder=0)
        #ax.plot(self.t_array, np.full(len(self.t_array), np.exp(-1)), c='black')
        ax.set_xlabel(xlabel, labelpad=xlabelpad)
        ax.set_ylabel(ylabel, labelpad=ylabelpad)
    
    def plot_time_chi4onself(self, ax, marker='^', ms=5, xlabel=r'$t$', ylabel='',color='blue', linewidth=1, xlabelpad=1, ylabelpad=1):
        ax.plot(self.t_array, self.chi4_array / self.self_array, marker=marker, ms=ms, label=str(self.temp), linewidth=linewidth, color=color, zorder=0)
        #ax.plot(self.t_array, np.full(len(self.t_array), np.exp(-1)), c='black')
        ax.set_xlabel(xlabel, labelpad=xlabelpad)
        ax.set_ylabel(ylabel, labelpad=ylabelpad)


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
    


    def im_integral(self, w):
        integ = 0
        def derivative(f_x, x):
            derivative_array = []
            y0, x0 = f_x[0], x[0]
            for y, x in zip(f_x[1:], x[1:]):
                der = (y - y0) / (x - x0)
                derivative_array.append(der)
                y0, x0 = y, x
            return np.array(derivative_array)
        derivative_corr = derivative(self.corr_array, np.log10(self.t_array))
        for i, elt in enumerate(derivative_corr):
            t = self.t_array[i+1]
            integ -= elt * w * t / (1 + (w*t)**2) * np.log10(t / self.t_array[i])
        return integ
    
    @property
    def w_range(self):
        return 2 * np.pi / self.t_array
    
    @property
    def dynamic_susceptibility(self):
        dynamic_susceptibility = []
        for w in self.w_range:
            dynamic_integral = self.im_integral(w)
            dynamic_susceptibility.append(dynamic_integral)
        return dynamic_susceptibility
    
    def plot_dynamic_susceptibility(self, ax, marker='D', ms=5, xlabel=r'$\omega$', ylabel=r"$\chi^{''}(\omega)$", color='blue', linewidth=1, xlabelpad=1, ylabelpad=1):
        ax.plot(self.w_range, self.dynamic_susceptibility, marker=marker, ms=ms, label=str(self.temp), linewidth=linewidth, color=color, zorder=0)
        ax.set_xlabel(xlabel, labelpad=xlabelpad)
        ax.set_ylabel(ylabel, labelpad=ylabelpad)