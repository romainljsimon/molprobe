import numpy as np
import scipy
import scipy.optimize
import molprobe.fit_functions as fit
import molprobe.colors as col
import glob
import re
import os 

natsort = lambda s: [int(t) if t.isdigit() else t.lower() for t in re.split('(\d+)', s)]

class MSDFolder:
    """
    A class to handle and plot Mean Squared Displacement (MSD) data from a folder containing MSD files.

    Parameters
    ----------
    path_folder : str, optional
        Path to the folder containing MSD data files. Default is './'.
    msd : str, optional
        Type of MSD calculation (e.g., 'cm' for center-of-mass MSD). Default is 'cm'.
    onset_time : float, optional
        Onset time used to scale the time axis in MSD data. Default is 1.
    start : int, optional
        Starting index for selecting files from the sorted list of files in the folder. Default is 0.
    end : int, optional
        Ending index for selecting files from the sorted list of files in the folder. Default is 0.

    Attributes
    ----------
    path_folder : str
        Path to the folder containing MSD data files.
    msd : str
        Type of MSD calculation (e.g., 'cm' for center-of-mass MSD).
    onset_time : float
        Onset time used to scale the time axis in MSD data.
    start : int
        Starting index for selecting files from the sorted list of files in the folder.
    end : int
        Ending index for selecting files from the sorted list of files in the folder.
    temp_array : list of float
        Array of temperatures extracted from file names.
    msd_array : list of numpy.ndarray
        List of arrays containing MSD data for each file.

    Methods
    -------
    prepare_folder()
        Prepares and loads MSD data from files in the specified folder.
    plot_msd(ax)
        Plots the MSD data on the provided matplotlib axis.
    """
    
    def __init__(self, path_folder='./', msd='cm', onset_time=1, start=0, end=0):
        self.path_folder = path_folder
        self.msd = msd
        self.onset_time = onset_time
        self.start = start
        self.end = end
        self.temp_array, self.msd_array = [], []
        self.prepare_folder()
    
    def prepare_folder(self):
        """
        Loads and processes MSD data files from the specified folder.

        The method:
        - Filters files based on the `msd` type specified.
        - Sorts files in natural order (e.g., '1', '2', '10', etc.) based on temperature information in file names.
        - Selects files based on `start` and `end` parameters, applies onset time scaling to the time column, and
          stores data in `temp_array` and `msd_array`.

        Raises
        ------
        FileNotFoundError
            If no files matching the pattern are found in the specified folder.
        """
        path_files = f'{self.path_folder}/{self.msd}*msd*'
        file_list = np.array(sorted(glob.glob(path_files), key=natsort))[::-1]
        file_list = np.array([elt for elt in file_list if '.png' not in elt])
        file_list = np.array([elt for elt in file_list if '.py' not in elt])
        file_list = file_list[self.start: len(file_list) - self.end]
        print(file_list)
        self.temp_array = np.array([float(re.findall(r"[-+]?(?:\d*\.*\d+)", s)[-1]) for s in file_list])
        idx = np.argsort(self.temp_array)[::-1]
        self.temp_array, file_list = self.temp_array[idx], file_list[idx]
        
        for i, elt in enumerate(file_list):
            elt_array = np.genfromtxt(elt)
            elt_array[:, 0] /= self.onset_time[i]
            self.msd_array.append(elt_array)
        #self.msd_array = np.array(self.msd_array)

    def calc_diff(self, filename='diff.txt'):
        diff_array = []
        for temp, t_msd in zip(self.temp_array, self.msd_array):
            t = t_msd[:, 0]
            msd = t_msd[:, 1]
            popt, _ = scipy.optimize.curve_fit(fit.linear, t[-5:], msd[-5:], bounds=([0,0],[np.inf, np.inf]))
            diff_coeff = popt[0] / 6
            diff_array.append([1 / temp, diff_coeff])
        diff_array = np.array(diff_array)
        np.savetxt(os.path.join(self.path_folder, filename), diff_array)

    def plot_msd(self, ax):
        """
        Plots the Mean Squared Displacement (MSD) data on a given matplotlib axis.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis on which the MSD data is to be plotted.

        Notes
        -----
        Each dataset in `msd_array` is plotted with the temperature value from `temp_array` as its label.
        """
        for msd, temp in zip(self.msd_array, self.temp_array):
            ax.plot(msd[:, 0], msd[:, 1], label=temp)
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('time')
        ax.set_ylabel('MSD')

    def plot_msd_div(self, ax, diff_array):
        """
        Plot Mean Squared Displacement (MSD) data normalized by the factor 6*D*t on a given matplotlib axis.

        The function plots MSD data for each temperature, dividing by \(6 \cdot D \cdot t\), where \(D\) 
        is the diffusion coefficient from `diff_array` and \(t\) is time. At long times, if the diffusion 
        coefficient \(D\) is accurate, the plotted values should converge to 1, indicating correct 
        diffusion scaling.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis object on which the normalized MSD data will be plotted.
        diff_array : 2d numpy arrau
            An array where each line is: (temperature, diffusion coefficient), used for scaling MSD data.

        Notes
        -----
        For each dataset in `msd_array`, the temperature from `temp_array` is matched with the diffusion 
        coefficient in `diff_array` to perform normalization. The plot is labeled with temperature values 
        and uses a logarithmic scale for both x and y axes.
        """
        for msd, temp, diff in zip(self.msd_array, self.temp_array, diff_array):
            if np.isclose(temp, 1 / diff[0]):

                div = 6 * msd[:, 0] * diff[1]
                ax.plot(msd[:, 0], msd[:, 1] / div, label=temp)
        ax.legend()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('time')
        ax.set_ylabel('MSD')