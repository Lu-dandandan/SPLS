import numpy as np
from SPLS.step0_segmentation import get_segs_time, get_segs_time_flux
from SPLS.step0_trend_model_comparison import DefaultTrendOrder
from SPLS.sampling import dp_fun, P_grid
from SPLS.funX_sig_and_trend import sele_sig, sele_trend, sele_global_trend
from SPLS.step1_linear import sele_fitting_2D
from SPLS.result import Result_LinearSearch, Result_Periodogram
from SPLS.step2_periodic_global import sele_fitting_global, sele_result_fitting, sele_fun_fold_map
from SPLS.constants import ori_kernel_size
from tqdm import tqdm 
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from time import time

class SPLeastSquares:   
    """
    SPLeastSquares (Segmented-Polynomial-fitting Least Squares) is a class for detecting exoplanet transits. 
    It simultaneously fits planetary transits and background trends using a segmented double polynomial model. 
    
    Methods:
    --------
    step0_segment: Segments the time series data into multiple segments based on time and flux gaps.
    
    step0_default_trend_order: Determines the default optimal polynomial order for the background trend component.
    
    step1_pre_default_dsam1: Generates a default set of transit durations sampled uniformly in logarithmic space.
    
    step1_linear_search: Performs a linear search to compute the log-likelihood over a grid of sampled transit durations and mid-transit times.
    
    step2_pre_default_Psam: Generates a default set of period samples.
    
    step2_periodogram: Performs a periodogram analysis including periodic search and global fitting.
    """ 

    def __init__(self, t, f, df):
        """
        t : numpy.ndarray
            Time array (in days). Must have the same cadence, without overlaps. Gaps are allowed and will be handled later.
        f : numpy.ndarray
            Flux array corresponding to time `t`.
        df : numpy.ndarray
            Uncertainty of the flux measurements.
        """  
          
        self.t_ori, self.f_ori, self.df_ori = self._sort_by_time(t, f, df) 
    
    def step0_segment(self, flux_gap=False, gap_time_threshold=4.5, gap_min_num_in_one_seg=10, gap_delta_flux_mad_threshold=5): 
        """
        Segment the time series data into multiple segments based on time and flux gaps.

        The segmentation is based on the following criteria:
        1. Time gaps: The time gap between consecutive segments exceeds `gap_time_threshold * cadence`. Each segment must contain at least 
       `gap_min_num_in_one_seg` data points.
        2. Flux gaps (optional): If `flux_gap` is set to True, the segmentation is further refined based on flux 
        gaps using a median absolute deviation (MAD) threshold (`gap_delta_flux_mad_threshold`).
        
        Parameters:
        -----------
        flux_gap : bool, optional
            If True, further segmenting is performed based on flux gaps (default is False).
        gap_time_threshold : float, optional
            Threshold multiplier for detecting time gaps. The gap between consecutive segments must exceed 
            `gap_time_threshold * cadence` (default is 4.5).
        gap_min_num_in_one_seg : int, optional
            The minimum number of data points required in each segment (default is 10).
        gap_delta_flux_mad_threshold : float, optional
            The threshold for flux gap detection, based on the median absolute deviation (default is 5).
        """

        ts1, fs1, dfs1, self.dt_median = get_segs_time(self.t_ori, self.f_ori, self.df_ori, gap_time_threshold, gap_min_num_in_one_seg)
        if len(ts1) == 0:
            raise ValueError("No segments meet the required criteria. Please adjust the parameters and data.")
        self.gap_time_threshold = gap_time_threshold

        # If flux gap detection is enabled, refine the segmentation based on flux gaps
        if flux_gap:
            ts2, fs2, dfs2 = get_segs_time_flux(ts1, fs1, dfs1, gap_delta_flux_mad_threshold, gap_time_threshold, gap_min_num_in_one_seg, self.dt_median)
            self.ts, self.fs, self.dfs = ts2, fs2, dfs2
        else:
            self.ts, self.fs, self.dfs = ts1, fs1, dfs1

        self.t, self.f, self.df = np.concatenate(self.ts), np.concatenate(self.fs), np.concatenate(self.dfs)
        self.time_span = self.t[-1] - self.t[0]
        self.dtmin = np.min(np.ediff1d(self.t))

    def step0_default_trend_order(self, dmax, window, 
                                  quantile_value=0.9, trend_order_max=3, OS_seg=12):
        """
        Determine the default optimal polynomial order for the background trend component.

        This method uses the `DefaultTrendOrder` class to determine the global best trend order. The optimal trend order is selected
        based on a quantile from the distribution of trend orders across all finer segments.

        Parameters:
        -----------
        dmax : float
            The maximum sampled duration (in days).
        window : float
            The size of the window used to segment the time series data (in days). It must be larger than dmax.
        quantile_value : float, optional
            The quantile value used to determine the best trend order from the distribution of trend orders 
            found across all segments (0-1, default is 0.9).
        trend_order_max : int, optional
            The maximum allowed trend order that will be considered in the analysis. The default value is 3, 
            meaning that the polynomial order can range from 0 to 3.
        OS_seg : int, optional
            The oversampling parameter for segmentation. A higher value increases the number of segments in the model comparison. (default is 12)

        Returns:
        --------
        trend_order : int
            The optimal trend order selected based on the model comparison and provided quantile value.
        """ 

        object = DefaultTrendOrder(self.ts, self.fs, self.dfs, trend_order_max, dmax, 0.5*window, OS_seg)
        trend_order = object.best_trend_order(quantile_value)
        return trend_order

    def step1_pre_default_dsam1(self, sig_order, dmax, dmin=None, OS_d=15):
        """
        Generate a default set of transit durations sampled uniformly in logarithmic space.

        Parameters:
        -----------
        sig_order : int
            The polynomial order of the signal component, used to compute the default minimum duration (`dmin`).
        dmax : float
            The maximum sampled duration (in days).
        dmin : float, optional
            The minimum sampled duration. If not provided, it is calculated based on `sig_order` 
            and the minimum time step (dtmin) (in days).
        OS_d : int, optional
            The number of points to sample between `dmin` and `dmax` (default is 15).

        Returns:
        --------
        dsam : numpy.ndarray
            Sampled transit durations which are uniform in log space between `dmin` and `dmax` (in days).
        """        

        # default dmin 
        if dmin is None: 
            dmin = (sig_order / 2.0 + 2.0) * self.dtmin 
        
        # Ensure that dmin is less than dmax
        if dmin >= dmax:
            raise ValueError(f"dmin {dmin} is greater than dmax. Please adjust the parameters.")
        return np.logspace(np.log10(dmin), np.log10(dmax), OS_d)

    def step1_linear_search(self, trend_order, sig_order, d_sam, window, 
                            max_workers, Pmin_step1, OS_P=2, 
                            dPmin=None,
                            Rs=1, Ms=1,
                            OS_tm=5,
                            A_limit=True
                            ):        
        """
        Perform a linear search to compute the log-likelihood (`dlnL`) over a grid of sampled transit durations (`d_sam`) and mid-transit times (`tm_sam`).
        This method allows for both sequential and parallel computations. 
        
        Parameters:
        -----------
        trend_order : int
            The polynomial order of the background trend component.
        sig_order : int
            The polynomial order of the signal component.
        d_sam : numpy.ndarray
            Sampled transit durations (in days). 
        window : float
            The window size (in days). It must be larger than dmax.
        max_workers : int
            The number of parallel workers to use. If 1, the computation is performed sequentially.
        Pmin1_step1 : float
            The minimum sampled period (in days). It must be no less than the window size.
        OS_P : int, optional
            The oversampling factor for the sampled periods. Default is 2 (range 2-5).
            If default sampled periods will be used, this parameter is required.
        dPmin : float, optional
            The minimum period step for the search (in days). If not provided, it is calculated based on `Rs`, `Ms`, and the time span.
            If default sampled periods will be used, this parameter should be the default value.
        Rs : float, optional
            The radius of the star in solar radii (Rsun). Default is 1.
        Ms : float, optional
            The mass of the star in solar masses (Msun). Default is 1.
        OS_tm : int, optional
            The oversampling factor for mid-transit times. Default is 5.
        A_limit : bool, optional
            Whether to apply a limit to the signal coefficients during fitting. Default is `True`.
        
        Returns:
        --------
        Result_LinearSearch : object
            A custom object containing the computed log-likelihood values (`dlnL`), the sampled mid-transit times 
            (`tm_sam`), and the sampled transit durations (`d_sam`).
            

        Notes:
        ------
        - If default period sampling is adopted, the parameters `Pmin_step1`, `OS_P`, `Rs`, `Ms` need to be considered. `dPmin` used in this case is the default value.
        - if custom period sampling is used by the user, the parameters `Pmin_step1`, `dPmin` need to be considered. `dPmin` should be provided by the user.
        """
        t_start = time()

        self.sig_order = sig_order
        self.trend_order = trend_order  
        self.Rs = Rs
        self.Ms = Ms
        self.OS_P = OS_P

        self.OS_tm = OS_tm 
        self.limit = A_limit
        self.Pmin_step1 = Pmin_step1

        if type(d_sam) != np.ndarray:
            d_sam = np.array([d_sam])
        d_sam = np.sort(d_sam)
        dmax = d_sam[-1]
        dmin = np.around(d_sam[0], 12)

        # 1. Check the duration samples, specifically the minimum duration
        dmin_default = np.round((sig_order / 2.0 + 2.0) * self.dtmin, 12) 
        if dmin < dmin_default:
            print(f'dmin {dmin} is smaller than the default, automatically set to the default value {dmin_default}')
            d_sam = d_sam[d_sam >= dmin_default]
            dmin = d_sam[0]
            if len(d_sam) == 0:
                raise ValueError("Reinput d_sam, no d_sam is larger than dmin_default")
        size_d = len(d_sam) 
        print(f'Sample {size_d} transit durations from {dmin} day to {dmax} day.')
        self.d_sam = d_sam


        # 2. Check the window size. Ensure the input window size is greater than dmax.
        if window <= dmax:
            raise ValueError("Inputted window size must be greater than dmax.")
        self.window = window

        # 3. Check the minimum period.
        if Pmin_step1 < window:
            print('Inputted Pmin is smaller than the window size, automatically set to the window size.')
            Pmin_step1 = np.copy(window) 
        # if Pmin_step1 > self.time_span / 2.0:
        #     raise ValueError("Pmin1 is larger than time_span/2, please reinput")

        # 4. tm_sam 
        if dPmin is None:
            dPmin = dp_fun(Rs, Ms, self.time_span, OS_P, Pmin_step1)
        self.tm_gap = np.minimum(dmin / OS_tm, dPmin)      
        self.tm_sam = np.arange(self.t[0], self.t[-1], self.tm_gap)
        print(f'Sample {len(self.tm_sam)} mid-transit times') 

        # Set up the fitting functions 
        self.get_X_sig_cut = sele_sig(sig_order) 
        self.get_X_trend_cut = sele_trend(trend_order)
        fitting_2D = sele_fitting_2D(A_limit)

        # 5. Compute the log-likelihood (`dlnL`) using either sequential or parallel computation
        if max_workers == 1: 
            pbar = tqdm(total=len(self.ts))
            self.dlnL = fitting_2D(0, self.ts, self.fs, self.dfs,
                                   size_d, d_sam, self.tm_sam,
                                   0.5 * window,
                                   sig_order, trend_order,
                                   self.get_X_sig_cut, self.get_X_trend_cut)
            for i in range(1, len(self.ts)): 
                dlnL_here = fitting_2D(i, self.ts, self.fs, self.dfs,
                                   size_d, d_sam, self.tm_sam,
                                   0.5 * window,
                                   sig_order, trend_order,
                                   self.get_X_sig_cut, self.get_X_trend_cut)
                self.dlnL = np.hstack((self.dlnL, dlnL_here)) 
                pbar.update(1)
            pbar.close()
        else: 
            i_arr = np.arange(0, len(self.ts), dtype=int)
            fun_i = partial(fitting_2D,
                            seg_ts=self.ts,
                            seg_fs=self.fs,
                            seg_dfs=self.dfs,
                            size_d=size_d,
                            d_sam=d_sam,
                            tm_sam=self.tm_sam,
                            half_window=0.5*window,
                            sig_order=sig_order,
                            trend_order=trend_order,
                            get_X_sig_cut=self.get_X_sig_cut,
                            get_X_trend_cut=self.get_X_trend_cut)
            with ProcessPoolExecutor(max_workers = max_workers) as exe:
                result = list(tqdm(exe.map(fun_i, i_arr), total = len(i_arr))) 
            self.dlnL = np.concatenate(result, axis=1) 

        t_end = time()  
        print('Linear search cost time', (t_end - t_start) / 60, 'min')
        return Result_LinearSearch(self.dlnL, self.tm_sam, self.d_sam)

    def step2_pre_default_Psam(self, Pmin_step2=None, Pmax_step2=None, min_num_transit=3):
        """
        Generate a default set of period samples (`P_sam`).

        Parameters:
        -----------
        Pmin_step2 : float, optional
            The minimum period for the sampling (in days). It must be no less than `Pmin_step1`. If not provided, it is set to the value of `Pmin_step1` (default is `None`).
        Pmax_step2 : float, optional
            The maximum period for the sampling (in days). If not provided, it is calculated as `time_span / (min_num_transit - 1)` (default is `None`).
        min_num_transit : int, optional
            The minimum number of transits required. Used to calculate the maximum period (default is 3).

        Returns:
        --------
        P_sam : numpy.ndarray
            Sampled period which is between `Pmin_step2` and `Pmax_step2`.
        """

        # Default minimum period
        if Pmin_step2 is None: 
            Pmin_step2 = self.Pmin_step1
        
        # Default maximum period
        if Pmax_step2 is None:
            Pmax_step2 = self.time_span / (min_num_transit - 1)

        # Generate the period samples
        P_sam = P_grid(self.Rs, self.Ms, self.time_span, self.OS_P, Pmin_step2, Pmax_step2)   
        return P_sam

    def step2_periodogram(self, P_sam, min_num_transit=3, max_workers=1, d_limit=True):
        """
        Perform a periodogram analysis including periodic search and global fitting. 
        
        Periodic search is to find the best transit duration and first mid-transit time for each sampled period. 
        Global fitting is applying a rigorous fitting to construct a periodogram. 
        Finally, obtain the best period, transit duration and first mid-transit time of signal at the peak.

        This method allows for both sequential and parallel computations. 
        But sequential computation is recommended, because the parallel computation is often slower due to the large memory usage.

        Parameters:
        -----------
        P_sam : numpy.ndarray
            The sampled periods (in days). 
        min_num_transit : int, optional
            The minimum number of transits required (default is 3).
        max_workers : int, optional
            The number of parallel workers to use. Default is 1, which is also recommended.
        d_limit : bool, optional
            Whether to apply a further limit on sampled duration in this step for acceleration (default is `True`).
        
        Returns:
        --------
        Result_Periodogram : object
            An object containing the results of the periodogram analysis, including:
            - results about sampled period-duration region (`d_sam`, `P_sam`)
            - periodogram results(`dlnL_arr`, `SDE_arr`)
            - best parameters (e.g., `P_best`, `d_best`, `tm0_best`, `SDE_best`)
            - model fitting results (e.g., `t_cut_list`, `phase_cut_list`, `f_cut_list`)
        """

        t_start = time()

        # Select fitting functions
        get_global_X_trend_cut = sele_global_trend(self.trend_order)
        fitting_global = sele_fitting_global(self.limit)
        result_fitting = sele_result_fitting(self.limit)
        fun_fold_map = sele_fun_fold_map(d_limit)

        P_sam = np.sort(P_sam)
        Pmin2 = P_sam[0]
        Pmax2 = P_sam[-1]

        # Check sampled periods
        if Pmax2 > self.time_span / (min_num_transit - 1):
            print('Inputted Pmax2 is larger than the time_span / min_num_transit, automatically set to time_span / min_num_transit.')
            P_sam = np.copy(P_sam[P_sam <= self.time_span / (min_num_transit - 1)])            
        if Pmin2 < self.Pmin_step1:
            print('Inputted Pmin2 is smaller than the first step Pmin1, automatically set to the first step Pmin1')
            P_sam = np.copy(P_sam[P_sam >= self.Pmin_step1])
        if len(P_sam) == 0:
            raise ValueError("Reinput P_sam, no P_sam remains.")
 
        # Adjust period samples.
        P_sam = (P_sam // self.tm_gap) * self.tm_gap
        print(f'Sample {len(P_sam)} periods from {Pmin2} day to {Pmax2} day.')

        cri0 = self.gap_time_threshold * self.dt_median
        inv_df = (1 / self.df)[:, None]
        wF = self.f / self.df
        wF = np.asarray(wF, dtype=np.float64)

        dlnL_arr1 = []
        dlnL_arr2 = [] 
        ds = []
        tm0s = []

        # Sequential computation
        if max_workers == 1:
            pbar1 = tqdm(total=len(P_sam))
            for P in P_sam: 
                dlnLmax1, dlnLmax2, d_best0, tm0_best0 = fitting_global(P,
                                                                    self.d_sam,
                                                                    self.tm_sam,
                                                                    self.dlnL,
                                                                    self.tm_gap,
                                                                    self.OS_tm,
                                                                    self.get_X_sig_cut,
                                                                    get_global_X_trend_cut,
                                                                    cri0,
                                                                    0.5*self.window,
                                                                    self.t,
                                                                    inv_df,
                                                                    wF,
                                                                    self.trend_order,
                                                                    self.sig_order,
                                                                    min_num_transit,
                                                                    fun_fold_map)     
                dlnL_arr1.append(dlnLmax1) 
                dlnL_arr2.append(dlnLmax2) 
                ds.append(d_best0)
                tm0s.append(tm0_best0)
                pbar1.update(1)
            pbar1.close()

        # Parallel computation
        else:
            fun_P = partial(fitting_global,
                            d_sam=self.d_sam,
                            tm_sam=self.tm_sam,
                            dlnL=self.dlnL,
                            tm_gap=self.tm_gap,
                            OS_tm=self.OS_tm,
                            get_X_sig_cut=self.get_X_sig_cut,
                            get_X_trend_cut=get_global_X_trend_cut,
                            cri0=cri0,
                            half_window=0.5*self.window,
                            t=self.t,
                            inv_df=inv_df,
                            wF=wF,
                            trend_order=self.trend_order,
                            sig_order=self.sig_order,
                            min_num_transit=min_num_transit,
                            fun_fold_map=fun_fold_map)
            with ProcessPoolExecutor(max_workers=max_workers) as exe:
                result = list(tqdm(exe.map(fun_P, P_sam), total=len(P_sam))) 
            for k in result:
                dlnL_arr1.append(k[0]) 
                dlnL_arr2.append(k[1]) 
                ds.append(k[2])
                tm0s.append(k[3])
        dlnL_arr1 = np.array(dlnL_arr1)
        dlnL_arr2 = np.array(dlnL_arr2)
        ds = np.array(ds)
        tm0s = np.array(tm0s)

        # Compute SDE metrics
        _, _, SDE_arr = self._metrics(dlnL_arr2, self.OS_P) 

        # Identify the best parameters based on the SDE
        best_idx = np.nanargmax(SDE_arr) 
        P_best = P_sam[best_idx]
        d_best = ds[best_idx] 
        tm0_best = tm0s[best_idx]  
        SDE_best = SDE_arr[best_idx]
        print('P_best', P_best, 'day')
        print('d_best', d_best * 24, 'hours')
        print('tm0_best', tm0_best, 'day')
        print('SDE_best', SDE_best) 

        t_end = time() 
        print('Periodic search and global fitting cost time', (t_end - t_start) / 60, 'min')

        # Perform final fitting using the best parameters
        t_cut_list, phase_cut_list, f_cut_list, model_t_cut_list, model_phase_cut_list, model_f_cut_list, phase_cut_sort, de_f_cut_sort, model_phase_cut_sort, model_de_f_cut_sort, n_seg, in_num,cut_num,depth,depth_sigma,depth_snr = result_fitting(P_best,
                                                                                                                                                                                                                                                       d_best, 
                                                                                                                                                                                                                                                       tm0_best,
                                                                                                                                                                                                                                                       self.t,
                                                                                                                                                                                                                                                       self.f,
                                                                                                                                                                                                                                                       0.5*self.window,
                                                                                                                                                                                                                                                       self.trend_order,
                                                                                                                                                                                                                                                       cri0,
                                                                                                                                                                                                                                                       self.get_X_sig_cut,
                                                                                                                                                                                                                                                       get_global_X_trend_cut,
                                                                                                                                                                                                                                                       inv_df,
                                                                                                                                                                                                                                                       wF,
                                                                                                                                                                                                                                                       self.sig_order,
                                                                                                                                                                                                                                                       self.dtmin)

        return Result_Periodogram(self.d_sam, P_sam, dlnL_arr2, SDE_arr, 
                                  P_best, d_best, tm0_best, SDE_best, 
                                  n_seg, in_num, cut_num, depth, depth_sigma, depth_snr,
                                  t_cut_list, phase_cut_list, f_cut_list, model_t_cut_list, model_phase_cut_list, model_f_cut_list, 
                                  phase_cut_sort, de_f_cut_sort, model_phase_cut_sort, model_de_f_cut_sort,
                                  self.t, self.f)       
 
    def _sort_by_time(self, t, f, df):
        """
        Sort the time, flux, and flux uncertainty arrays based on time `t`.
        """
        sorted_indices = np.argsort(t)
        t_sorted = t[sorted_indices]
        f_sorted = f[sorted_indices]
        df_sorted = df[sorted_indices]
        return t_sorted, f_sorted, df_sorted
    def _sliding_median(self, data, kernel):
        """
        Compute the sliding-window median of a 1D array, handling NaNs, and pad the ends.

        Parameters
        ----------
        data : numpy.ndarray 
            1D array data, may contain NaNs.
        kernel : int
            Size of the sliding window (must be <= len(data)).

        Returns
        -------
        med : numpy.ndarray
            1D array of length `len(data)`, where each central position holds the median
            of the corresponding window in `data`. The end positions, where a full window
            cannot be centered, are filled with the first and last valid median values.
        """

        idx = np.arange(kernel) + np.arange(len(data) - kernel + 1)[:, None]
        idx = idx.astype(np.int64) 
        med = []

        # For each window, compute median ignoring NaNs; if all are NaN, result is NaN
        for kk in range(idx.shape[0]): 
            arr = data[idx[kk]]
            if np.all(np.isnan(arr)):
                med.append(np.nan)
            else:
                med.append(np.nanmedian(arr))
        med = np.array(med)

        # Determine padding lengths at front and end
        first_values = med[0]
        last_values = med[-1]
        missing_values = len(data) - len(med) 
        values_front = int(missing_values * 0.5)
        values_end = missing_values - values_front

        # Pad the beginning with the first median value, and the end with the last
        med = np.append(np.full(values_front, first_values), med)
        med = np.append(med, np.full(values_end, last_values)) 
        return med

    def _metrics(self, dlnL, OS):
        """
        Compute signal detection efficiency (SDE) metrics.

        Parameters
        ----------
        dlnL :  numpy.ndarray
            Array of ΔlnL (change in log-likelihood) values, may contain NaNs.
        OS : int 
            The oversampling factor for periods used to determine the sliding median kernel size.

        Returns
        -------
        normalized_dlnL : numpy.ndarray
            The ΔlnL array shifted and scaled to the range [0, 1].
        SDE_raw : numpy.ndarray
            The raw signal detection efficiency: (normalized_dlnL - mean) / std.
        SDE : numpy.ndarray
            The detrended SDE, obtained by subtracting a sliding-median trend (when data
            length permits) and re-normalizing to zero mean and unit standard deviation.
        """

        # Shift to zero minimum and scale to [0, 1]
        shift_dlnL = dlnL - np.nanmin(dlnL)
        normalized_dlnL = shift_dlnL / np.nanmax(shift_dlnL)

        # Compute raw SDE
        SDE_raw = (normalized_dlnL - np.nanmean(normalized_dlnL)) / np.nanstd(normalized_dlnL) 
        
       # Detrend SDE with sliding median if enough data points
        kernel = OS * ori_kernel_size
        if kernel % 2 == 0:
            kernel = kernel + 1
        if len(SDE_raw) > 2 * kernel:
            slide_median = self._sliding_median(SDE_raw, kernel)
            SDE_ori = SDE_raw - slide_median

            # Recompute SDE so that mean = 0
            SDE = (SDE_ori - np.nanmean(SDE_ori)) / np.nanstd(SDE_ori) 
        else:
            SDE = SDE_raw 

        return normalized_dlnL, SDE_raw, SDE