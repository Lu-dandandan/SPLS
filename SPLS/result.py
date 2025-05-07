import numpy as np   
import pandas as pd       
import matplotlib.pyplot as plt     
from matplotlib import rcParams; rcParams["figure.dpi"] = 250 
from SPLS.constants import A_up,A_down  
import matplotlib.gridspec as gridspec
import os 
from pathlib import Path 

class Result_LinearSearch:
    """
    A class for storing and visualizing the results of the linear search.

    Attributes:
    -----------
    dlnL : numpy.ndarray
        A 2D array of log-likelihood values computed during the linear search. 
        The shape should match the grid of sampled transit durations and mid-transit times.
    
    tm_sam : numpy.ndarray
        A 1D array of the sampled mid-transit times (in days).
    
    d_sam : numpy.ndarray
        A 1D array of the sampled transit durations (in days).
    
    Methods:
    --------
    plot: visualizing the results of the linear search.

    """
    def __init__(self, dlnL, tm_sam, d_sam):

        self.dlnL = dlnL
        self.tm_sam = tm_sam
        self.d_sam = d_sam

    def plot2d(self):
        """
        Plots a 2D contour plot of the log-likelihood difference values (`dlnL`) as a function 
        of sampled mid-transit times (`tm_sam`) and transit durations (`d_sam`).
        """
        d_sam_hour = self.d_sam * 24
        fig, ax = plt.subplots(figsize=(10, 3.5))
        im = ax.contourf(self.dlnL, cmap=plt.cm.Oranges, levels=25)
        ax.set_xticks(np.linspace(0, len(self.tm_sam) - 1, 10), ['%.0f'%i for i in np.linspace(self.tm_sam[0], self.tm_sam[-1], 10)])
        range1 = np.linspace(0, len(d_sam_hour) - 1, 9)[[0, 4, 6, 7, 8]]
        range2 = np.logspace(np.log10(d_sam_hour[0]), np.log10(d_sam_hour[-1]), 9)[[0, 4, 6, 7, 8]]
        ax.set_yticks([i for i in range1], ['%.3f'%i for i in range2])
        ax.set_xlabel(r'$t_m$ [day]', fontsize=12)
        ax.set_ylabel(r'$d$ [hour]', fontsize=12)
        ax.tick_params(axis='x', direction='in') 
        ax.tick_params(axis='y', direction='in')  
        cbar = fig.colorbar(im)
        cbar.set_label(r'$\Delta \ln \mathcal{L}$')
        plt.show()
    def plot1d(self):
        """
        Plots a 1D line plot of the log-likelihood difference values (`dlnL`) as a function of 
        the sampled mid-transit times (`tm_sam`).

        This plot is useful when there is only one transit duration sample.
        """
        fig, ax = plt.subplots()
        ax.plot(self.tm_sam,np.squeeze(self.dlnL), c='k', lw=0.5)
        ax.set_xlabel(r'$t_m$ [day]', fontsize=12)
        ax.set_ylabel(r'$\Delta \ln \mathcal{L}$', fontsize=12)
        ax.tick_params(axis='x', direction='in') 
        ax.tick_params(axis='y', direction='in')  
        plt.show()
    def plot(self):
        """
        Determines whether to plot a 2D or 1D plot based on the number of transit 
        duration samples (`d_sam`). If more than one transit duration is sampled, 
        a 2D plot is created. Otherwise, a 1D plot is generated.
        """
        if len(self.d_sam) > 1:
            self.plot2d()
        else:
            self.plot1d()

class Result_Periodogram:
    """
    A class to store and visualize results related to the periodogram analysis.

    Attributes:
    -----------
    d_sam : numpy.ndarray
        Sampled transit durations.

    P_sam : numpy.ndarray
        Sampled periods.

    dlnL_arr : numpy.ndarray
        Array of log-likelihood difference values corresponding to sampled periods.

    SDE_arr : numpy.ndarray
        Array of signal detection efficiencies (SDE) corresponding to sampled periods.

    P_best : float
        The best period derived from the analysis (in days).

    d_best : float
        The best transit duration (in days).

    tm0_best : float
        The best mid-transit time (in days).

    SDE_best : float
        The highest signal detection efficiency (SDE).

    n_seg : int
        The number of segments used for fitting at the best parameters.

    in_num : int
        The number of data points in transits at the best parameters.

    cut_num : int
        The number of data points in segments at the best parameters.

    depth : float
        The transit depth at the best parameters.

    depth_sigma : float
        The sigma of the transit depth at the best parameters.

    depth_snr : float
        The signal-to-noise ratio (SNR) of the transit depth at the best parameters.

    t_cut_list : list
        List of time of fitted segments at the best parameters.

    phase_cut_list : list
        List of phase of fitted segments at the best parameters.

    f_cut_list : list
        List of flux of fitted segments at the best parameters.

    model_t_cut_list : list
        List of model time of fitted segments at the best parameters.

    model_phase_cut_list : list
        List of model phase of fitted segments at the best parameters.

    model_f_cut_list : list
        List of model flux of fitted segments at the best parameters.

    phase_cut_sort : numpy.ndarray
        Sorted phase of fitted segments at the best parameters.

    de_f_cut_sort : numpy.ndarray
        Sorted flux residuals after subtracting trend model corresponding to sorted phase cuts.

    model_phase_cut_sort : numpy.ndarray
        Sorted model phase of fitted segments at the best parameters.

    model_de_f_cut_sort : numpy.ndarray
        Sorted model flux residuals after subtracting trend model corresponding to sorted model phase cuts.


    t : numpy.ndarray
        Time data for the light curve.

    f : numpy.ndarray
        Flux data for the light curve.

    Methods:
    --------
    plots: plotting various figures such as period vs. log-likelihood,
    period vs. signal detection efficiency (SDE), phase curves, and time curves.

    """

    def __init__(self, d_sam, P_sam, dlnL_arr, SDE_arr, 
                 P_best, d_best, tm0_best, SDE_best, 
                 n_seg, in_num, cut_num, depth, depth_sigma, depth_snr,
                 t_cut_list, phase_cut_list, f_cut_list, model_t_cut_list, model_phase_cut_list, model_f_cut_list, 
                 phase_cut_sort, de_f_cut_sort, model_phase_cut_sort, model_de_f_cut_sort,
                 t, f):
        
        # parameters about periodogram
        self.d_sam = d_sam
        self.P_sam = P_sam
        self.dlnL_arr = dlnL_arr
        self.SDE_arr = SDE_arr

        # best parameters
        self.P_best = P_best
        self.d_best = d_best
        self.tm0_best = tm0_best 
        self.SDE_best = SDE_best

        # The relative parameters at the best parameters
        self.number_segments_at_best_parameters = n_seg
        self.data_number_in_transits = in_num
        self.data_number_in_segments = cut_num 
        self.depth_best = depth
        self.depth_sigma_best = depth_sigma 
        self.depth_snr = depth_snr

        # data used for plots
        self.t_cut_list = t_cut_list
        self.phase_cut_list = phase_cut_list 
        self.f_cut_list = f_cut_list
        self.model_t_cut_list = model_t_cut_list 
        self.model_phase_cut_list = model_phase_cut_list
        self.model_f_cut_list = model_f_cut_list
        self.phase_cut_sort = phase_cut_sort
        self.de_f_cut_sort = de_f_cut_sort
        self.model_phase_cut_sort = model_phase_cut_sort
        self.model_de_f_cut_sort = model_de_f_cut_sort 
        self.t = t
        self.f = f

    def Pdsam_region(self,dmin,dmax,pmin,pmax):
        """
        Plots the sampling range in the period-duration space.
        """
        fig,ax = plt.subplots()
        dir_name = os.path.dirname(__file__)
        addre = Path(dir_name)/'NASA_Pd.csv'
        data = pd.read_csv(addre) 
        P = np.array(data['P[day]'])
        d = np.array(data['d[hour]'])
        ax.scatter(P, d / 24, alpha=0.8, s=5, c='#616F8C', edgecolor='none') 
        ax.set_xscale('log') 
        ax.set_yscale('log') 
        ax.tick_params(axis='both', which='both', direction='in')  # 调整主刻度
        ax.tick_params(axis='both', which='minor', direction='in')  # 调整次刻度
        ax.set_xlabel(r'$P$ [day]') 
        ax.set_ylabel(r'$d$ [day]') 

        # two lines
        pp = np.arange(0.2,1000,0.1)
        d_up = A_up*pp**(1/3.0)
        d_down = A_down*pp**(1/3.0)
        ax.plot(pp,d_up,'#716DF2', linestyle='--',linewidth=1.2)
        ax.plot(pp,d_down,'#716DF2', linestyle='--',linewidth=1.2)

        corners1 = [
            (pmin, dmin),   # Bottom-left
            (pmax, dmin),  # Bottom-right
            (pmax, dmax),    # Top-right
            (pmin, dmax)      # Top-left
        ]
        x_coords, y_coords = zip(*corners1)
        ax.fill(x_coords, y_coords, color='none',linewidth=1.7, alpha=1, edgecolor='#F2A679',label='Sampling range in the linear search')
        ax.legend(fontsize=7)
        plt.show()
        
    def P_dlnL2(self, ax):
        """
        Plots period vs. log-likelihood difference (`dlnL_arr`).
        """
        ax.plot(self.P_sam, self.dlnL_arr, c='k', lw=0.5, zorder=1)
        ax.set_xscale('log')
        ax.set_xlabel('Period [d]')
        ax.set_title(r'P-$\Delta \ln \mathcal{L}$')
        ax.set_ylabel(r'$\Delta \ln \mathcal{L}$')

        ax.axvline(self.P_best, alpha=0.3, lw=2.5, c='#fcaa3a', zorder=0)
        n = 2
        alias = n * self.P_best
        if alias <= self.P_sam[-1]:
            ax.axvline(alias, alpha=0.4, lw=1, linestyle="dashed", c='#fcaa3a', zorder=0)
            n += 1
            alias = n * self.P_best
        n = 2
        alias = self.P_best / n
        if alias >= self.P_sam[0]:
            ax.axvline(alias, alpha=0.4, lw=1, linestyle="dashed", c='#fcaa3a', zorder=0)
            n += 1
            alias = self.P_best / n
        ax.tick_params(axis='both', direction='in') 
        ax.tick_params(axis='both', which='minor', direction='in') 

    def P_SDE(self, ax):
        """
        Plots period vs. signal detection efficiency (`SDE_arr`).
        """
        ax.plot(self.P_sam, self.SDE_arr, c='k', lw=0.5, zorder=1)
        ax.set_xscale('log')
        ax.set_xlabel('Period [d]')
        ax.set_title('P-SDE')
        ax.set_ylabel('SDE')

        ax.axvline(self.P_best, alpha=0.3, lw=2.5, c='#fcaa3a', zorder=0)
        n = 2
        alias = n * self.P_best
        if alias <= self.P_sam[-1]:
            ax.axvline(alias, alpha=0.4, lw=1, linestyle="dashed", c='#fcaa3a', zorder=0)
            n += 1
            alias = n * self.P_best
        n = 2
        alias = self.P_best / n
        if alias >= self.P_sam[0]:
            ax.axvline(alias, alpha=0.4, lw=1, linestyle="dashed", c='#fcaa3a', zorder=0)
            n += 1
            alias = self.P_best / n

        ax.tick_params(axis='both', direction='in') 
        ax.tick_params(axis='both', which='minor', direction='in')  

    def phase_curve1(self, ax): 
        """
        Plots the phase curve within the windows used for fitting. 
        The phase light curves of different segments are vertically shifted for clarity and shows the simultaneous fitting of both background trends and the signal.
        """ 

        f0 = self.f_cut_list[0]
        ax.scatter(self.phase_cut_list[0], f0, c='black', s=10, alpha=0.5)
        model_f0 = self.model_f_cut_list[0]
        ax.plot(self.model_phase_cut_list[0], model_f0)
        fmin_before = np.min(f0)

        for k in range(1, len(self.phase_cut_list)):
            f0 = np.copy(self.f_cut_list[k])
            model_f0 = np.copy(self.model_f_cut_list[k])
            fmax_after = np.max(f0)
            delta = np.abs(fmin_before-fmax_after)
            f0 -= delta
            model_f0 -= delta
            fmin_before = np.min(f0)
            ax.scatter(self.phase_cut_list[k], f0, c='black', s=10, alpha=0.5)
            ax.plot(self.model_phase_cut_list[k], model_f0)
        ax.set_xlabel('Phase')
        ax.set_ylabel('Flux')   
        ax.ticklabel_format(useOffset=False)


    def phase_curve2(self, ax):
        """
        Plots the phase curve within the windows used for fitting. 
        The data and model subtracts the fitted trend component from the periodic model.     
        """ 
        ax.scatter(self.phase_cut_sort, self.de_f_cut_sort, c='black', s=10, alpha=0.3)
        ax.plot(self.model_phase_cut_sort, self.model_de_f_cut_sort, c='r', lw=2)
        ax.set_xlabel('Phase')
        ax.set_ylabel('Flux')
        ax.tick_params(axis='both', direction='in') 
        ax.tick_params(axis='both', which='minor', direction='in') 
    
    def time_curve(self, ax):
        """
        Plots the flux as a function of time.
        """
        ax.scatter(self.t, self.f, c='black', s=1, alpha=0.4)
        for k in range(len(self.model_t_cut_list)):
            ax.plot(self.model_t_cut_list[k], self.model_f_cut_list[k], lw=1.5, c='r') 
        ax.set_xlim(np.min(self.t), np.max(self.t))
        ax.set_xlabel('Time [day]')
        ax.set_ylabel('Flux') 
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(axis='both', direction='in') 
        ax.tick_params(axis='both', which='minor', direction='in') 

    def plots(self):
        """
        Creates and displays multiple plots for analyzing the data, including the sampling region,
        P-`dlnL`, P-SDE, phase curves, and time curve.
        """
        dmin = self.d_sam[0]
        dmax = self.d_sam[-1]
        pmin = self.P_sam[0]
        pmax = self.P_sam[-1]
        self.Pdsam_region(dmin, dmax, pmin, pmax)

        fig = plt.figure(figsize=(8, 9))
        gs = gridspec.GridSpec(3, 2) 
        ax1 = plt.subplot(gs[0]) 
        self.P_dlnL2(ax1)
        ax2 = plt.subplot(gs[1])
        self.P_SDE(ax2)
        ax3 = plt.subplot(gs[2])
        self.phase_curve1(ax3)
        ax4 = plt.subplot(gs[3])
        self.phase_curve2(ax4)
        ax5 = plt.subplot(gs[2, :])
        self.time_curve(ax5)
        fig.tight_layout()
