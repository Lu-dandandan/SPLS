import numpy as np   
import matplotlib.pyplot as plt       
from matplotlib import rcParams; rcParams["figure.dpi"] = 250   
from SPLS.step1_linear import get_chi2_base 
from SPLS.funX_sig_and_trend import sele_trend 

class DefaultTrendOrder:
    """
    A class to determine the best trend order for a given light curve by evaluating the 
    chi-squared fit for various trend orders and selecting the most appropriate one based 
    on the log Bayes Factor (lnBF) for each segment. The optimal global trend order is selected
    based on a quantile from the distribution of trend orders across all segments.

    Attributes:
    -----------
    ts : list of numpy.ndarray
        The time series data for multiple segments (in days).
    fs : list of numpy.ndarray
        The flux corresponding to each time segment in ts.
    dfs : list of numpy.ndarray
        The uncertainties (errors) of flux for each time segment in ts.
    trend_order_max : int
        The maximum allowed trend order for model fitting.
    dmax : float
        The maximum sampled transit duration in days.
    half_window : float 
        Half the size of the time window used for slicing the data (in days).
    OS_seg : int
        The oversampling parameter for segmentation.

    Methods:
    --------
    _compute_chi2(trend_order, t_cut, inv_df_cut, wF_cut):
        Computes the chi-squared value for a given trend order and data segment.
        
    _best_trend_one_seg(t_cut, inv_df_cut, wF_cut, num):
        Determines the best trend order for a single segment by comparing the chi-squared 
        values for different trend orders using the log Bayes Factor (lnBF).
        
    best_trend_order(quantile_value):
        Computes the best trend order for the entire dataset based on the specified quantile 
        value of trend orders computed for each segment. Returns the best trend order and 
        the array of trend orders for each segment.
    """

    def __init__(self, ts, fs, dfs, trend_order_max, dmax, half_window, OS_seg):
        self.ts, self.fs, self.dfs = ts, fs, dfs 
        self.dmax = dmax
        self.half_window = half_window 
        self.trend_order_max = trend_order_max
        self.OS_seg = OS_seg

    def _compute_chi2(self, trend_order, t_cut, inv_df_cut, wF_cut):
        """
        Compute the chi-squared value for a given trend order over a window of data.

        Parameters:
        -----------
        trend_order : int 
            The polynomial order for the background trend. 
        t_cut : numpy.ndarray  
            Sliced time data (in days).
        inv_df_cut : numpy.ndarray
            Inverse of the uncertainty for the sliced data.
        wF_cut : numpy.ndarray
            Weighted flux for the sliced data.

        Returns:
        --------
        chi2_base : numpy.ndarray
            Computed chi-squared value.
        """
        get_X_trend_cut = sele_trend(trend_order)   
        X_trend_cut = get_X_trend_cut(t_cut)
        wF_cut = np.asarray(wF_cut, dtype=np.float64)
        chi2_base = get_chi2_base(X_trend_cut, inv_df_cut, wF_cut)
        return chi2_base
    
    def _best_trend_one_seg(self, t_cut, inv_df_cut, wF_cut, num):
        """
        Compute the best trend order for a window of data.

        Parameters:
        -----------
        t_cut : numpy.ndarray
            Sliced time data (in days).
        inv_df_cut : numpy.ndarray
            Inverse uncertainty for the sliced data.
        wF_cut : numpy.ndarray
            Weighted flux for the sliced data.
        num : int
            Number of data points in the segment.

        Returns:
        --------
        best_trend : int
            Best trend order for this segment.
        """

        chi2_before = self._compute_chi2(0, t_cut, inv_df_cut, wF_cut) # Compute chi-squared for trend order 0
        for j in range(1, self.trend_order_max + 1):
            chi2_after = self._compute_chi2(j, t_cut, inv_df_cut, wF_cut) 
            # Calculate log Bayes Factor (lnBF) to decide if the trend order can be increased
            lnBF = 0.5 * (chi2_before - chi2_after) - 0.5 * np.log(num)
            if lnBF < 5.0:
                return j - 1 # If lnBF is less than 5, increasing trend order does not improve the model
            chi2_before = np.copy(chi2_after)
        return self.trend_order_max # If all trend orders fail, return the maximum trend order
    
    def best_trend_order(self, quantile_value):
        """
        Compute and return the best trend order for the entire dataset.

        Parameters:
        -----------
        quantile_value : float
            Quantile value to determine the best trend order (0-1).

        Returns:
        --------
        best_trend : int
            Best trend order determined from the data.
        best_trend_arr : numpy.ndarray
            Array of trend orders computed for each segment of data.
        """ 

        best_trend_arr = []
        step = self.dmax / self.OS_seg

        # Iterate through each data set (ts, fs, dfs)
        for i in range(len(self.ts)):
            t = self.ts[i]
            f = self.fs[i]
            df = self.dfs[i]
            inv_df = (1 / df)[:, None] 
            wF = f / df 

            for tm in np.arange(t[0], t[-1], step):
                cut_idx = np.where((t >= (tm - self.half_window)) & (t <= (tm + self.half_window)))[0]
                cut_num0 = len(cut_idx)
                if cut_num0 < (self.trend_order_max + 1) * 2 + 2:
                    continue # Skip if there are too few data points in the segment

                # Normalize the time data for the current segment
                t_cut_ori = t[cut_idx]
                t1 = t_cut_ori[0]
                t2 = t_cut_ori[-1]  
                t_cut = (2.0 * t_cut_ori - (t1 + t2)) / (t2 - t1)
                inv_df_cut = inv_df[cut_idx]
                wF_cut = wF[cut_idx]

                # Compute the best trend order for this segment
                best_trend0 = self._best_trend_one_seg(t_cut, inv_df_cut, wF_cut, cut_num0)
                best_trend_arr.append(best_trend0)

        # Compute the final best trend order using the specified quantile
        best_trend = int(np.quantile(best_trend_arr, quantile_value))
        print(f'Default best trend order is {best_trend} for {len(best_trend_arr)} segments in the model comparison.')
        plt.figure()
        plt.hist(best_trend_arr, bins = np.arange(-0.5, 4.5), facecolor = '0.7')
        plt.axvline(np.quantile(best_trend_arr, 0.9), c='r', lw = 1)
        plt.xlabel('Polynomial order')
        plt.ylabel('Number of segments')
        plt.xticks(ticks = [ii for ii in range(int(self.trend_order_max + 1))], labels=['%d'%iii for iii in range(int(self.trend_order_max + 1))])
        plt.tick_params(axis='x', direction='in')  
        plt.tick_params(axis='y', direction='in') 
        return best_trend