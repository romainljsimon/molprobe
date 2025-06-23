import numpy as np
import scipy
import scipy.optimize
import molprobe.fit_functions as ex
import molprobe.colors as col
import glob
import re
import pyfftlog
import os

natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]

class TauFile:
    def __init__(self, path_tau='', legend='', onset_time=1,  start=0, end=0, rescale_temp=1,
                 tg_parab=None, onset_temp=None, extrapolation=False, tts=0, marker='s', ms=5, 
                 color='b', inverse_temp=True, inverse_y=False):
        self.path_tau = path_tau
        self.legend = legend
        self.onset_time = onset_time
        self.start = start
        self.end = end
        self.tts = tts
        self.extrapolation = extrapolation
        self.marker = marker
        self.ms = ms
        self.color = color
        self.temp_array, self.t_array = np.array([]), np.array([])
        self.rescale_temp = rescale_temp
        self.onset_temp = onset_temp
        self.inverse_temp = inverse_temp
        self.inverse_y = inverse_y
        self.prepare_file()
        self.tg_parab = tg_parab
        self.t_extrapol_parab, self.temp_extrapol_parab = [], []
        self.activation_energy = []
        if extrapolation:
            self.calculate_extrapolation()

    def prepare_file(self):
        if os.path.splitext(self.path_tau)[1] == '.csv':
            data = np.genfromtxt(self.path_tau, delimiter=',')
        else:
            data = np.genfromtxt(self.path_tau)
        mask = np.any(np.isnan(data), axis=1)
        data = data[~mask]
        if self.inverse_temp:
            idx = np.argsort(data[:, 0])
        else:
            idx = np.argsort(1 / data[:, 0])
        data = data[idx]
        self.end = len(data) - self.end
        self.tts = len(data) - self.tts
        
        if self.inverse_temp:
            self.temp_array = 1 / data[self.start:self.end , 0]
        else:
            self.temp_array = data[self.start:self.end , 0]
        self.temp_array /= self.rescale_temp
        self.t_array = data[self.start:self.end , 1] / self.onset_time
        if self.inverse_y:
            self.t_array = 1 / self.t_array
    
    def calculate_extrapolation(self):
        mask = self.t_array > 1
        temp_array_fit_parab, relaxation_array_fit_parab = self.temp_array[mask], self.t_array[mask] 
        self.popt_parab, _ = scipy.optimize.curve_fit(ex.func_parab, temp_array_fit_parab, 
                                                      np.log(relaxation_array_fit_parab), 
                                                      bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))
        f_parab = lambda x: ex.func_parab(x, *self.popt_parab) - np.log(10**12)
        #self.tg_parab = scipy.optimize.root(f_parab , 1/np.array([0.8])).x[0]
        self.tg_parab = scipy.optimize.brentq(f_parab, 0.01, np.max(self.temp_array))
        self.temp_extrapol_parab = np.linspace(self.tg_parab, np.max(self.temp_array))
        self.t_extrapol_parab = np.exp(ex.func_parab(self.temp_extrapol_parab, *self.popt_parab))

        

    def plot_tau(self, ax, color, markeredgecolor=None):
        if self.extrapolation:    
            x_array = 1 / self.temp_extrapol_parab
            ax.plot([1/self.tg_parab, 1/self.tg_parab], [1e-2, 1e12], '--', color=self.color)
            ax.plot(x_array, self.t_extrapol_parab, color=self.color, label='Parabolic Extrapolation')
        
        if markeredgecolor is None:
            markeredgecolor = col.adjust_lightness(self.color)
        ax.plot(1 / self.temp_array[:self.tts], self.t_array[:self.tts], marker=self.marker, ms=self.ms, color=color,  
                 markeredgecolor=markeredgecolor, label=self.legend)
        if self.tts < len(self.temp_array):
            ax.plot(1 / self.temp_array[self.tts:], self.t_array[self.tts:], marker='s', markerfacecolor='none', 
                    markersize=self.ms, markeredgecolor=col.adjust_lightness('blue'), label='TTS')
        ax.set_xlabel(r'$1/ T$')
        ax.set_yscale('log')
        ax.set_ylabel(r'$\tau / \tau_o$')

    def calculate_activation_energy(self):
        mask_high_temp = self.temp_array > self.onset_temp
        inv_temp_array = 1. / self.temp_array 
        tau_array_fit, inv_temp_array_fit = self.t_array[mask_high_temp], inv_temp_array[mask_high_temp]   
        ln_tau_array_fit = np.log(tau_array_fit)
        popt, _ = scipy.optimize.curve_fit(ex.func_high_temp, inv_temp_array_fit, ln_tau_array_fit, bounds = ([-np.inf, 0], [np.inf, np.inf]))
        self.activation_energy = (np.log(self.t_array) - popt[0]) / inv_temp_array / popt[1] 
        return popt
    
    def calculate_fragility(self, fragility_type):
        if fragility_type == 'angell':
            if self.extrapolation != True:
                self.calculate_extrapolation()
                
        elif fragility_type == 'activation_energy':
                self.calculate_activation_energy()


    def plot_fragility(self, ax, fragility_type='angell'):
        self.calculate_fragility(fragility_type)
        
        if fragility_type == 'angell':
            
            
            ax.plot([1, 1], [1e-2, 1e12], '--', color=self.color)
            x_array = self.tg_parab / self.temp_extrapol_parab
            ax.plot(x_array, self.t_extrapol_parab, color=self.color, label='Parabolic Extrapolation')
        
            ax.plot(self.tg_parab /  self.temp_array[:self.tts], self.t_array[:self.tts], marker=self.marker, ms=self.ms, color=self.color,  
                    markeredgecolor=col.adjust_lightness(self.color), label=self.legend)
            if self.tts < len(self.temp_array):
                ax.plot(self.tg_parab /  self.temp_array[self.tts:], self.t_array[self.tts:], color='none', 
                        marker='s', markerfacecolor='none', markersize=self.ms, 
                        markeredgecolor=col.adjust_lightness('blue'), label='TTS')
            ax.set_xlabel(r'$T_g / T$')
            ax.set_ylabel(r'$\tau / \tau_o$')
            
        elif fragility_type == 'activation_energy':
            ax.plot(self.onset_temp / self.temp_array, self.activation_energy, marker=self.marker, 
                    ms=self.ms, color=self.color,  markeredgecolor=col.adjust_lightness(self.color), 
                    label=self.legend)
            ax.set_xlabel(r'$T_o / T$')
            ax.set_ylabel(r'$E(T) / E_{\infty}$')
            
class Decoupling:
    def __init__(self, TauFile1, TauFile2, inverse_x=False, inverse_y=False):
        self.TauFile1, self.TauFile2 = TauFile1, TauFile2
        self.tau_array_1, self.tau_array_2 = [], []
        self.inverse_x, self.inverse_y = inverse_x, inverse_y
        self.temp_array = []
        self.prepare_files()

    def prepare_files(self):
        self.temp_array = [elt for elt in self.TauFile1.temp_array if elt in self.TauFile2.temp_array]
        self.temp_array = np.array(self.temp_array)
        mask1, mask2 = np.isin(self.TauFile1.temp_array, self.temp_array), np.isin(self.TauFile2.temp_array, self.temp_array)
        self.tau_array_1, self.tau_array_2 = self.TauFile1.t_array[mask1], self.TauFile2.t_array[mask2]

    def plot(self, ax, rescale_y=1, xlabel='', ylabel='', label='', marker='s', ms=5, color='b'):
        if self.inverse_y:
            ax.plot(self.tau_array_1, 1 / self.tau_array_2 / rescale_y, marker=marker, ms=ms, color=color, label=label)
        else:
            ax.plot(self.tau_array_1, self.tau_array_2, marker=marker, ms=ms, color=color, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)


class CorrFolder:
    def __init__(self, path_folder='./', corr='fsqt', onset_time=1,  start=0, end=0):
        self.path_folder = path_folder
        self.corr = corr
        self.onset_time = onset_time
        self.start = start
        self.end = end
        self.temp_array, self.corr_array = [], []
        self.fht_array = []
        self.prepare_folder()
    
    def prepare_folder(self):
        path_files = f'{self.path_folder}/{self.corr}*'
        file_list = np.array(sorted(glob.glob(path_files), key=natsort))[::-1]
        file_list = np.array([elt for elt in file_list if '.png' not in elt])
        file_list = file_list[self.start: len(file_list) - self.end]
        self.temp_array = np.array([float(re.findall(r"[-+]?(?:\d*\.*\d+)", s)[-1]) for s in file_list])
        for elt in file_list:
            elt_array = np.genfromtxt(elt)
            elt_array[:, 0] /= self.onset_time
            self.corr_array.append(elt_array)
        #self.corr_array = np.array(self.corr_array)

    def extract_tau(self, filename, fstar=np.exp(-1)):
        """
        Find first root of f=f(x) for data sets.

        Given two lists x and f, it returns the value of xstar for which
        f(xstar) = fstar. Raises an ValueError if no root is found.
        """
        tau_array = []
        for j, corr in enumerate(self.corr_array):
            s = corr[0, 1] - fstar
            for i in range(len(corr)):
                if (corr[i, 1] - fstar) * s < 0.0:
                    # Linear interpolation
                    dxf = (corr[i, 1] - corr[i-1, 1]) / (corr[i, 0] - corr[i-1, 0])
                    
                    xstar = corr[i-1, 0] + (fstar - corr[i-1, 1]) / dxf
                    tau_array.append([1./self.temp_array[j], xstar])
                    break
        # We get to the end and cannot find the root
        np.savetxt(filename, np.array(tau_array))

    def compute_chi_fftlog(self, logt, c_t):
        n=len(logt)
        dlogr=(logt.max()-logt.min())/n
        dlnr = dlogr*np.log(10.0)
        kr, xsave = pyfftlog.fhti(n, mu=-1/2, dlnr=dlnr, q=0, kr=1, kropt=1)
        ak = pyfftlog.fftl(c_t.copy(), xsave, tdir=1)
        logrc = (logt.max()+logt.min())/2
        logkc = np.log10(kr) - logrc
        nc = (n + 1)/2.0
        k = 10**(logkc + (np.arange(1, n+1) - nc)*dlogr)
        ak = np.abs((ak-ak[-1])*k)
        ak = ak[2: -2]
        
        # compute vertical scale
        freq = k / (2*np.pi)
        df = freq - np.roll(freq, 1)
        df[0] = 0
        eps0_self_init = 2/np.pi*np.cumsum(ak/freq*df)[-1]
        eps0_self = c_t[0]
        ak * eps0_self / eps0_self_init
        print(ak)
        return k, ak
    
    def fft_log(self):
        self.fht_array = []
        for elt in self.corr_array:
            time, corr = elt[:, 0], elt[:, 1]
            omega, fftlog = self.compute_chi_fftlog(np.log10(time), corr)
            self.fht_array.append(np.column_stack((omega, fftlog)))
            break
        self.fht_array = np.array(self.fht_array)
            
    def plot_corr(self, ax):
        for corr, temp in zip(self.corr_array, self.temp_array):
            ax.plot(corr[:, 0], corr[:, 1], label=temp)
        ax.legend()
        ax.set_xscale('log')
    
    def plot_fft(self, ax):
        for fht, temp in zip(self.fht_array, self.temp_array):
            ax.plot(fht[:, 0], fht[:, 1], label=temp)
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')