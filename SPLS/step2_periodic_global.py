import numpy as np
from SPLS.sampling import d_grid
from numba import njit 

def fun(f):
    return f

def sele_fitting_global(A_limit):
    if A_limit:
        return fun(fitting_global_limitA) 
    else:
        return fun(fitting_global_no_limitA)

def sele_result_fitting(A_limit):
    if A_limit:
        return fun(result_fitting_limitA)
    else:
        return fun(result_fitting_no_limitA)

def sele_fun_fold_map(d_limit):
    if d_limit:
        return fun(fold_map_d_constraint)
    else:
        return fun(fold_map_no_d_constraint)

def fold_map_no_d_constraint(P, d_sam, tm_sam, dlnL, tm_gap, OS_tm, min_num_transit):
    """
    Perform the folding of the result of linear search based on a given period.

    Parameters
    ----------
    P : float
        The period for folding the data (in days).
    d_sam : numpy.ndarray 
        Sampled durations (in days). This array must be sorted in ascending order (from smallest to largest).
    tm_sam : numpy.ndarray
        The sampled mid-transit times (in days).
    dlnL : numpy.ndarray
        The log-likelihood difference values of linear search.
    tm_gap : float
        The time interval between consecutive time samples.
    OS_tm : int
        The oversampling factor for mid-transit times.
    min_num_transit: int
        The minimum number of transits required for the fitting.

    Returns
    -------
    d_best0 : float
        The best duration corresponding to the maximum log-likelihood difference.
    tm0_best0 : float
        The best mid-transit time corresponding to the maximum log-likelihood difference.
    dlnLmax0 : float
        The maximum log-likelihood difference value for the folded result.

    Notes
    -------
    This function has no further restriction on the sampled durations based on the period P
    """

    row = int(P // tm_gap) 

    # Enlarge the resolution of tm0
    tm_gap_new = d_sam[0] / OS_tm
    nn = tm_gap_new // tm_gap 

    # Number of mid-transit times of a set of P, tm0
    col = int(dlnL.shape[1] // row + 1) 
    num_d = dlnL.shape[0]

    # Add NaN values to ensure that the grid shape can be folded
    add = int(row - dlnL.shape[1] % row)  
    dlnL = np.hstack((dlnL, np.full((num_d, add), np.nan)))
    dlnL = np.reshape(dlnL, (num_d, col, row)) 

    # New matrix with the new resolution
    if (nn != 0) and (nn != 1):
        idx_sele = np.arange(0, row, nn, dtype=int)
        dlnL = dlnL[:, :, idx_sele]

    # Identify at least 3 transits, row_indices are for d, col are for tm0
    row_indices, col_indices  = np.where(np.count_nonzero(~np.isnan(dlnL), axis=1) < min_num_transit)
    dlnL[row_indices, :, col_indices] = np.nan

    # Sum tmj to construct a grid of tm0 and d
    dlnL_co = np.nansum(dlnL, axis=1) 
    dlnLmax0 = np.nanmax(dlnL_co)
    if dlnLmax0 == 0.0: # less than 3 transits
        d_best0 = np.nan
        tm0_best0 = np.nan
        dlnLmax0 = np.nan
    else:
        idx_d, idx_tm0 = np.unravel_index(np.nanargmax(dlnL_co), dlnL_co.shape)
        d_best0 = d_sam[idx_d]
        if (nn != 0) and ((nn != 1)):
            tm0_sam = tm_sam[: row][idx_sele] 
            tm0_best0 = tm0_sam[idx_tm0]
        else: 
            tm0_sam = tm_sam[: row] 
            tm0_best0 = tm0_sam[idx_tm0]
    return d_best0, tm0_best0, dlnLmax0

def fold_map_d_constraint(P, d_sam, tm_sam, dlnL, tm_gap, OS_tm, min_num_transit):
    """
    Perform the folding of the result of linear search based on a given period.

    Parameters
    ----------
    P : float
        The period for folding the data (in days).
    d_sam : numpy.ndarray 
        Sampled durations (in days). This array must be sorted in ascending order (from smallest to largest).
    tm_sam : numpy.ndarray
        The sampled mid-transit times (in days).
    dlnL : numpy.ndarray
        The log-likelihood difference values of linear search.
    tm_gap : float
        The time interval between consecutive time samples.
    OS_tm : int
        The oversampling factor for mid-transit times.
    min_num_transit: int
        The minimum number of transits required for the fitting.

    Returns
    -------
    d_best0 : float
        The best duration corresponding to the maximum log-likelihood difference.
    tm0_best0 : float
        The best mid-transit time corresponding to the maximum log-likelihood difference.
    dlnLmax0 : float
        The maximum log-likelihood difference value for the folded result.

    Notes
    -------
    This function has a further restriction on the sampled durations based on the period P
    
    """

    d_sam2, idx_dd = d_grid(P, d_sam) 
    dlnL2 = dlnL[idx_dd, :]  # Select the corresponding log-likelihood values for the new sampled durations
    # Number of tm0 
    row = int(P // tm_gap) 

    # Enlarge the resolution of tm0
    tm_gap_new = d_sam2[0] / OS_tm
    nn = tm_gap_new // tm_gap 

    # Number of mid-transit times of a set of P, tm0
    col = int(dlnL2.shape[1] // row + 1) 
    num_d = dlnL2.shape[0]

    # Add NaN values to ensure that the grid shape can be folded
    add = int(row - dlnL2.shape[1] % row)  
    dlnL2 = np.hstack((dlnL2, np.full((num_d, add), np.nan)))
    dlnL2 = np.reshape(dlnL2, (num_d, col, row)) 

    # New matrix with the new resolution
    if (nn != 0) and (nn != 1):
        idx_sele = np.arange(0, row, nn, dtype=int)
        dlnL2 = dlnL2[:, :, idx_sele]

    # Identify at least 3 transits, row_indices are for d, col are for tm0
    row_indices, col_indices  = np.where(np.count_nonzero(~np.isnan(dlnL2), axis=1) < min_num_transit)
    dlnL2[row_indices, :, col_indices] = np.nan

    # Sum tmj to construct a grid of tm0 and d
    dlnL_co = np.nansum(dlnL2, axis=1) 
    dlnLmax0 = np.nanmax(dlnL_co)
    if dlnLmax0 == 0.0: # less than 3 transits
        d_best0 = np.nan
        tm0_best0 = np.nan
        dlnLmax0 = np.nan
    else:
        idx_d, idx_tm0 = np.unravel_index(np.nanargmax(dlnL_co), dlnL_co.shape)
        d_best0 = d_sam2[idx_d]
        if (nn != 0) and ((nn != 1)):
            tm0_sam = tm_sam[: row][idx_sele] 
            tm0_best0 = tm0_sam[idx_tm0]
        else: 
            tm0_sam = tm_sam[: row] 
            tm0_best0 = tm0_sam[idx_tm0]
    return d_best0, tm0_best0, dlnLmax0

@njit
def get_phase(t, tm, P): 
    u = (t - tm) / P + 0.5
    group = np.floor(u) 
    return u - group, group  # Transit occurs at phase(0-1) = 0.5 

@njit
def get_wX(X_sig_cut, X_trend_cut, inv_df_cut): 
    X_cut = np.hstack((X_sig_cut, X_trend_cut))
    wX_cut = X_cut * inv_df_cut 
    return wX_cut 

def get_XwX(X_sig_cut, X_trend_cut, inv_df_cut):
    X_cut = np.hstack((X_sig_cut, X_trend_cut))
    wX_cut = X_cut * inv_df_cut 
    return X_cut, wX_cut 

@njit
def get_covar(wX_cut):
    xtx = wX_cut.T @ wX_cut
    return np.linalg.inv(xtx)

@njit 
def get_L(wX_cut, wF_cut): 
    var = get_covar(wX_cut)
    L = var @ wX_cut.T @ wF_cut
    return L

@njit
def get_chi2_base(X_trend_cut, inv_df_cut, wF_cut):
    wX_base = X_trend_cut * inv_df_cut
    L = get_L(wX_base, wF_cut)
    wR_base = wF_cut - wX_base @ L
    return wR_base.T @ wR_base

@njit
def selection_fixed_window(phase, group, q, half_phase_window, trend_order, P, cri0, valid_mask, cut_index):
    """
    This function processes data based on the given input arrays `phase` and `group` within a 
    fixed window centered around phase = 0.5. Within the fitting window, the segment must have at least one point in the transit region, at least trend + 1 points on both sides, and no gaps; otherwise, discard the segment.
    
    Parameters:
    ----------
    phase : numpy.ndarray
        Array of phase values.
    group : numpy.ndarray 
        Array representing the group of each phase value.
    q : float
        The phase duration of the transit.
    half_phase_window : float
        Half the width of the phase window used to cut the data.
    trend_order : int
        The polynomial order of trend component.
    P : float
        Sampled period (in days).
    cri0 : float 
        The base critical value for time gap detection.
    valid_mask : numpy.ndarray
        A boolean mask indicating valid data points.
    cut_index : numpy.ndarray
        Indices of the data points to be cut.

    Returns:
    ---------
    valid_cut_index : numpy.ndarray
        Indices of the valid data points within the selected window.
    valid_group_cut : numpy.ndarray
        The corresponding group array for the valid data points.
    valid_phase_cut : numpy.ndarray
        The corresponding phase values for the valid data points.
    g_uni_final : numpy.ndarray 
        Unique group identifiers for the valid data points.
    
    """

    # Identify the indices that fall within the phase window centered around 0.5
    group_cut = group[cut_index] 
    phase_cut = phase[cut_index] 
    g_uni = np.unique(group_cut) 

    cri = cri0 / P # time gap threshold

    # Process each unique group
    for s in g_uni:
        idx = np.where(group_cut == s)[0]
        phase_cut_seg = phase_cut[idx]  

        # check
        in_idx = np.where((phase_cut_seg >= 0.5 - q / 2.0) & (phase_cut_seg <= 0.5 + q / 2.0))[0]
        in_num = len(in_idx)

        if in_num < 1: 
            valid_mask[idx] = False
            continue
        
        inbool0_left = ((phase_cut_seg >= 0.5 - half_phase_window) & (phase_cut_seg <= 0.5 - q / 2.0))
        left_num = np.sum(inbool0_left)
        cut_num = len(phase_cut_seg)
        if left_num < trend_order + 1:
            valid_mask[idx] = False   
            continue 
        if (cut_num-left_num-in_num) < trend_order + 1:
            valid_mask[idx] = False   
            continue

        # Check for gaps larger than the critical value   
        d_phase = np.diff(phase_cut_seg) 
    
        # If there's a gap larger than the critical threshold, split the segment
        if np.max(d_phase) > cri:  
            split_idx = np.where(d_phase > cri)[0] # Find positions where gap exceeds threshold. At least one gap
            
            # More than 3 gaps, discard this segment
            if len(split_idx) > 3: 
                valid_mask[idx] = False
                continue

            # One gap, split into two segments
            elif len(split_idx) == 1:  
                # print(1)
                aa = (split_idx + 1)[0]
                a = phase_cut_seg[split_idx]    
                b = phase_cut_seg[split_idx + 1]

                if (a < (0.5 - q / 2.0)) and (b < (0.5 - q / 2)): 
                    # Check if left side satisfies the trend condition
                    inbool0_left = ((phase_cut_seg >= phase_cut_seg[aa]) & (phase_cut_seg <= (0.5 - q / 2.0)))
                    left_num = np.sum(inbool0_left) 
                    if left_num < trend_order + 1:
                        valid_mask[idx] = False 
                        continue
                    else: 
                        valid_mask[idx[:aa]] = False # Remove the leftmost segment
                        continue
                elif (a > (0.5 + q / 2.0)) and (b > (0.5 + q / 2.0)):
                    # Check if right side satisfies the trend condition
                    inbool0_right = ((phase_cut_seg >= (0.5 + q / 2.0)) & (phase_cut_seg < phase_cut_seg[aa]))
                    right_num = np.sum(inbool0_right)
                    if right_num < trend_order + 1:
                        valid_mask[idx] = False
                        continue 
                    else: 
                        valid_mask[idx[aa:]] = False # Remove the rightmost segment
                        continue
                else:
                    valid_mask[idx] = False
                    continue 
            else: # Case with two gaps
                # print(2)
                edge1 = split_idx[0]
                edge2 = split_idx[1]  
                a = phase_cut_seg[edge1]    
                b = phase_cut_seg[edge1 + 1]
                c = phase_cut_seg[edge2]    
                dd = phase_cut_seg[edge2 + 1] 

                if dd < (0.5 - q / 2.0):  
                   # Check left side for trend condition
                    inbool0_left = ((phase_cut_seg >= phase_cut_seg[edge2 + 1]) & (phase_cut_seg <= 0.5 - q / 2.0))
                    left_num = np.sum(inbool0_left)
                    if left_num < trend_order + 1:
                        valid_mask[idx] = False    
                        continue 
                    else: 
                        valid_mask[idx[: edge2 + 1]] = False
                        continue
                elif a > (0.5 + q / 2.0):
                    # Check right side for trend condition
                    inbool0_right = ((phase_cut_seg >= (0.5 + q / 2.0)) & (phase_cut_seg <= phase_cut_seg[edge1]))
                    right_num = np.sum(inbool0_right)
                    if right_num < trend_order + 1:
                        valid_mask[idx] = False   
                        continue
                    else: 
                        valid_mask[idx[edge1 + 1:]] = False
                        continue

                elif (b < (0.5 - q / 2.0)) and (c > (0.5 + q / 2.0)):
                    # Remove both sides of the segment 
                    inbool0_left = ((phase_cut_seg >= phase_cut_seg[edge1 + 1]) & (phase_cut_seg <= 0.5 - q / 2.0))
                    left_num = np.sum(inbool0_left) 
                    if left_num < trend_order + 1:
                        valid_mask[idx] = False     
                        continue 
                    inbool0_right = ((phase_cut_seg >= (0.5 + q / 2.0)) & (phase_cut_seg <= phase_cut_seg[edge2]))
                    right_num = np.sum(inbool0_right)
                    if right_num < trend_order + 1:
                        valid_mask[idx] = False   
                        continue
                    xx = np.concatenate((idx[: edge1 + 1], idx[edge2 + 1:]))
                    valid_mask[xx] = False
                    continue
                else:
                    valid_mask[idx] = False  
                    continue          

    # Return the valid cut indices, corresponding groups, phases, and unique groups                 
    valid_cut_index = cut_index[valid_mask]
    valid_group_cut = group_cut[valid_mask]
    valid_phase_cut = phase_cut[valid_mask]
    g_uni_final = np.unique(valid_group_cut)
    return valid_cut_index, valid_group_cut, valid_phase_cut, g_uni_final


@njit
def get_X0_trend(phase_cut_len, g_uni_final, group_cut):
    """
    Creates a binary matrix (X0) based on the group assignments. 
    For each unique group in 'g_uni_final', the corresponding rows in 'X0' 
    are marked with 1 at positions where the group matches in 'group_cut'.
    """

    X0 = np.zeros((phase_cut_len, len(g_uni_final)))
    for a, s in enumerate(g_uni_final):
        s_idx = np.where(group_cut == s)[0]
        X0[s_idx, a] = 1
    return X0



def fitting_global_no_limitA(P, d_sam, tm_sam, dlnL, tm_gap, OS_tm, get_X_sig_cut,
                             get_X_trend_cut, cri0, half_window, t, inv_df, wF, trend_order, sig_order, min_num_transit, fun_fold_map):   
    """
    For a given period P and without limiting signal coefficients, this function performs:
    - A periodic search to find optimal transit duration (d) and mid-transit time (tm0),
    - A global least-squares fitting to improve fit quality.
    
    Parameters:
    ----------
    P : float
        The sampled period (in days).
    d_sam : numpy.ndarray
        The sampled transit duration (in days).
    tm_sam : numpy.ndarray
        The sampled mid-transit times (in days).
    dlnL : numpy.ndarray
        The log-likelihood difference.
    tm_gap : float
        The interval of sampled mid-transit times.
    OS_tm : float
        The oversampling factor for mid-transit times.
    get_X_sig_cut : function
        Function to compute the signal design matrix for the phase cut.
    get_X_trend_cut : function
        Function to compute the trend design matrix for the phase cut.
    cri0 : float
        The critical value used for time gap criteria.
    half_window : float
        Half of the window size (in days).
    t : numpy.ndarray
        Times.
    inv_df : numpy.ndarray
        Inverse uncertainty of flux.
    wF : numpy.ndarray
        The weight flux array.
    trend_order : int
        The order of the trend to be considered in the fitting.
    sig_order : int
        The order of the signal to be considered in the fitting.
    min_num_transit: int
        The minimum number of transits required for the fitting.
    fun_fold_map: function
        Function to perform the folding of the result of linear search based on a given period.

    Returns:
    -------
    dlnLmax1 : float
        The maximum log-likelihood difference of periodic search.
    dlnLmax2 : float
        The maximum log-likelihood difference of the global fitting.
    d : float
        The best-fit d parameter (in days).
    tm0 : float
        The best-fit tm0 parameter (in days).

    """  

    # Find the best d and tm0 for this period in the periodic search.
    d, tm0, dlnLmax1 = fun_fold_map(P, d_sam, tm_sam, dlnL, tm_gap, OS_tm, min_num_transit)

    if np.isnan(dlnLmax1): 
        dlnLmax2 = np.nan
    else: 
        q = d / P 
        half_phase_window = half_window / P

        # Get the phase and group information for the given tm0 and P
        phase, group = get_phase(t, tm0, P)

        # Find valid data points within the fixed window
        cut_index = np.where((phase >= 0.5 - half_phase_window) & (phase <= 0.5 + half_phase_window))[0] 
        valid_mask = np.ones(len(cut_index), dtype=bool) # Initialize mask
        cut_index, group_cut, phase_cut, g_uni_final = selection_fixed_window(phase, group, q, half_phase_window, trend_order, P, cri0, valid_mask, cut_index)

        phase_cut = (2.0 * phase_cut - 1.0) / q
        # Prepare the design matrices and other necessary data for fitting
        inbool = (phase_cut >= -1.0) & (phase_cut <= 1.0)
        y_cut = np.where(inbool, phase_cut, 1.0)
        X_sig_cut = get_X_sig_cut(y_cut) 
        X0 = get_X0_trend(len(phase_cut), g_uni_final, group_cut) 
        X_trend_cut = get_X_trend_cut(phase_cut.reshape(len(phase_cut), 1), X0)
        inv_df_cut = inv_df[cut_index]
        wX_cut = get_wX(X_sig_cut, X_trend_cut, inv_df_cut)
        wF_cut = wF[cut_index]
        L = get_L(wX_cut, wF_cut)
        # 求chi2_base 
        chi2_base = get_chi2_base(X_trend_cut, inv_df_cut, wF_cut)
        wR = wF_cut - wX_cut @ L
        chi2 = wR.T @ wR
        dlnLmax2 = 0.5 * (chi2_base - chi2)
    return dlnLmax1, dlnLmax2, d, tm0  


def fitting_global_limitA(P, d_sam, tm_sam, dlnL, tm_gap, OS_tm, get_X_sig_cut, 
                          get_X_trend_cut, cri0, half_window, t, inv_df, wF, trend_order, sig_order, min_num_transit, fun_fold_map):     
    """
    For a given period P and with limiting signal coefficients, this function performs:
    - A periodic search to find optimal transit duration (d) and mid-transit time (tm0),
    - A global least-squares fitting to improve fit quality.
    
    Parameters:
    ----------
    P : float
        The sampled period (in days).
    d_sam : numpy.ndarray
        The sampled transit duration (in days).
    tm_sam : numpy.ndarray
        The sampled mid-transit times (in days).
    dlnL : numpy.ndarray
        The log-likelihood difference.
    tm_gap : float
        The interval of sampled mid-transit times.
    OS_tm : float
        The oversampling factor for mid-transit times.
    get_X_sig_cut : function
        Function to compute the signal design matrix for the phase cut.
    get_X_trend_cut : function
        Function to compute the trend design matrix for the phase cut.
    cri0 : float
        The critical value used for time gap criteria.
    half_window : float
        Half of the window size (in days).
    t : numpy.ndarray
        Times.
    inv_df : numpy.ndarray
        Inverse uncertainty of flux.
    wF : numpy.ndarray
        The weight flux array.
    trend_order : int
        The order of the trend to be considered in the fitting.
    sig_order : int
        The order of the signal to be considered in the fitting.
    min_num_transit: int
        The minimum number of transits required for the fitting.
    fun_fold_map: function
        Function to perform the folding of the result of linear search based on a given period.

    Returns:
    -------
    dlnLmax1 : float
        The maximum log-likelihood difference of periodic search.
    dlnLmax2 : float
        The maximum log-likelihood difference of the global fitting.
    d : float
        The best-fit d parameter (in days).
    tm0 : float
        The best-fit tm0 parameter (in days).

    """ 

    d, tm0, dlnLmax1 = fun_fold_map(P, d_sam, tm_sam, dlnL, tm_gap, OS_tm, min_num_transit)

    if np.isnan(dlnLmax1):
        dlnLmax2 = np.nan
    else:
        q = d / P 
        half_phase_window = half_window / P  

        # Get the phase and group information for the given tm0 and P
        phase, group = get_phase(t, tm0, P)

        # Find valid data points within the fixed window
        cut_index = np.where((phase >= 0.5 - half_phase_window) & (phase <= 0.5 + half_phase_window))[0]
        valid_mask = np.ones(len(cut_index), dtype=bool) # Initialize mask
        cut_index, group_cut, phase_cut, g_uni_final = selection_fixed_window(phase, group, q, half_phase_window, trend_order, P, cri0, valid_mask, cut_index) 
        
        phase_cut = (2.0 * phase_cut - 1) / q
        # Prepare the design matrices and other necessary data for fitting
        inbool = (phase_cut >= -1.0) & (phase_cut <= 1.0)
        y_cut = np.where(inbool, phase_cut, 1.0)
        X_sig_cut = get_X_sig_cut(y_cut) 
        X0 = get_X0_trend(len(phase_cut), g_uni_final, group_cut) 
        X_trend_cut = get_X_trend_cut(phase_cut.reshape(len(phase_cut), 1), X0)
        inv_df_cut = inv_df[cut_index]
        wX_cut = get_wX(X_sig_cut, X_trend_cut, inv_df_cut)
        wF_cut = wF[cut_index]
        L = get_L(wX_cut, wF_cut)

        if sig_order == 2:
            if L[0] < 0.0:
                chi2 = np.nan
            else:
                wR = wF_cut - wX_cut @ L
                chi2 = wR.T @ wR
        if sig_order == 4: 
            if (L[0] > 0.0) and (L[1] > 0.0): 
                wR = wF_cut - wX_cut @ L 
                chi2 = wR.T @ wR
            elif (L[0] > 0) and (L[0] / L[1] < -2.0):  
                wR = wF_cut - wX_cut @ L
                chi2 = wR.T @ wR 
            elif ((L[0] / L[1] > -2.0) & (L[0] > 0) & (L[1] < 0)):
                xin = -2.0 * phase_cut ** 2 + phase_cut ** 4
                xout = -1.0 
                x2 = np.where(inbool, xin, xout)
                wX_cut = get_wX(x2[:, None], X_trend_cut, inv_df_cut)
                L = get_L(wX_cut, wF_cut) 
                if L[0] > 0.0:
                    chi2 = np.nan
                else: 
                    wR = wF_cut - wX_cut @ L
                    chi2 = wR.T @ wR
            else:
                chi2 = np.nan 

        chi2_base = get_chi2_base(X_trend_cut, inv_df_cut, wF_cut)
        dlnLmax2 = 0.5 * (chi2_base - chi2)
    return dlnLmax1, dlnLmax2, d, tm0  


def result_fitting_no_limitA(P, d, tm0, t, f, half_window, trend_order, cri0, 
                             get_X_sig_cut, get_X_trend_cut, inv_df, wF, sig_order, dtmin):
    """
    Perform segmented model fitting at the best parameters without limiting signal coefficients.

    Parameters
    ----------
    P : float
        The sampled orbital period (in days).
    d : float
        The sampled transit duration (in days).
    tm0 : float
        The sampled mid-transit time (in days).
    t : numpy.ndarray
        Time array of the observations (in days).
    f : numpy.ndarray
        Flux array.
    half_window : float
        Half width of the time window around each transit (in days).
    trend_order : int
        Polynomial order of the trend model to fit.
    cri0 : float
        Critical threshold used during data selection.
    get_X_sig_cut : function
        Function that returns the design matrix for the signal model.
    get_X_trend_cut : function
        Function that returns the design matrix for the trend model.
    inv_df : numpy.ndarray
        Inverse of the uncertainty of the flux measurements.
    wF : numpy.ndarray
        Weighted flux array.
    sig_order : int
        Polynomial order of the signal component.
    dtmin : float
        Minimum time interval in the time series.

    Returns
    -------
    t_cut_list : list of numpy.ndarray
        Time arrays for each fitted segment.
    phase_cut_list : list of numpy.ndarray
        Phase values for each segment.
    f_cut_list : list of numpy.ndarray
        Observed flux for each segment.
    model_t_cut_list : list of numpy.ndarray
        Model time arrays.
    model_phase_cut_list : list of numpy.ndarray
        Model phase arrays.
    model_f_cut_list : list of numpy.ndarray
        Model flux arrays.
    phase_cut_sort : numpy.ndarray
        Sorted phase arrays.
    de_f_cut_sort : numpy.ndarray
        Sorted flux arrays after subtracting trend component.
    model_phase_cut_sort : numpy.ndarray
        Sorted model phase arrays.
    model_de_f_cut_sort : numpy.ndarray
        Sorted model flux arrays after subtracting trend component.
    n_seg : int
        Number of separate phase segments fitted.
    in_num : int
        Number of in-window data points (phase in [-1, 1]).
    cut_num : int
        Total number of data points used.
    depth : float
        Estimated transit depth.
    depth_sigma : float
        Uncertainty in the estimated depth.
    depth_snr : float
        Signal-to-noise ratio of the depth estimate.
    """

    q = d / P
    half_phase_window = half_window / P

    # Get phase and group index for each timestamp
    phase, group = get_phase(t, tm0, P)

    # Select all data points within the phase window
    cut_index = np.where((phase >= 0.5 - half_phase_window) & (phase <= 0.5 + half_phase_window))[0] 
    valid_mask = np.ones(len(cut_index), dtype=bool) # Initialize mask
    
    # Further refine selection using a fixed window selection method
    cut_index, group_cut, phase_cut_ori, g_uni_final = selection_fixed_window(phase, group, q, half_phase_window, trend_order, P, cri0, valid_mask, cut_index)

    phase_cut = (2.0 * phase_cut_ori - 1.0) / q
    
    t_cut = t[cut_index]
    f_cut = f[cut_index]

    # Build signal design matrix
    inbool = (phase_cut >= -1.0) & (phase_cut <= 1.0)
    y_cut = np.where(inbool, phase_cut, 1.0)
    X_sig_cut = get_X_sig_cut(y_cut) 

    # Build trend design matrix
    X0 = get_X0_trend(len(phase_cut), g_uni_final, group_cut) 
    X_trend_cut = get_X_trend_cut(phase_cut.reshape(len(phase_cut), 1), X0)
    
    # Build weighted design matrix
    inv_df_cut = inv_df[cut_index]
    wX_cut = get_wX(X_sig_cut, X_trend_cut, inv_df_cut)
    wF_cut = wF[cut_index]
    
    # Fit the model: calculate covariance and coefficients
    covar = get_covar(wX_cut) 
    L = covar @ wX_cut.T @ wF_cut

    if sig_order == 2:
        L_sig = np.array([L[0]])
    if sig_order == 4:
        L_sig = L[0: 2]

    nL_trend = (X_trend_cut.shape)[1] 
    L_trend = L[-nL_trend:]
    L_trend_reshape = L_trend.reshape(int(trend_order + 1), len(L_trend) // (trend_order + 1)) 

    phase_cut_list = []
    f_cut_list = [] 
    t_cut_list = []
    
    model_t_cut_list = []
    model_phase_cut_list = []
    model_f_cut_list = []

    model_de_f_cut_list = []
    aa = 0
    for s in g_uni_final:
        idx = np.where(group_cut==s)
        phase_cut_list.append(phase_cut_ori[idx])
        t0 = t_cut[idx]
        t_cut_list.append(t0)
        f_cut_list.append(f_cut[idx])
        model_t0 = np.arange(t0[0], t0[-1], 0.1 * dtmin) 
        model_t_cut_list.append(model_t0)

        model_phase0, _ = get_phase(model_t0, tm0, P)
        model_phase_cut_list.append(model_phase0)
        model_phase0_fit = (2.0 * model_phase0 - 1.0) / q
        inbool_model = (model_phase0_fit >= -1.0) & (model_phase0_fit <= 1.0)
        y_cut_model = np.where(inbool_model, model_phase0_fit, 1.0)
        X_sig_cut_model = get_X_sig_cut(y_cut_model) 

        X_trend_cut_model = np.vstack([model_phase0_fit ** i for i in range(trend_order + 1)]).T
        X_model = np.hstack((X_sig_cut_model, X_trend_cut_model))
        L_trend_here = L_trend_reshape[: , aa]
        L_here = np.concatenate((L_sig, L_trend_here))
        model_f0 = X_model @ L_here
        model_f_cut_list.append(model_f0)
        model_f_trend0 = X_trend_cut_model @ L_trend_here
        model_de_f0 = model_f0 - model_f_trend0
        model_de_f_cut_list.append(model_de_f0)
        aa += 1

    f_trend = X_trend_cut @ L_trend
    de_f_cut = f_cut - f_trend 

    # Sort for plotting
    idx2 = np.argsort(phase_cut_ori)
    phase_cut_sort = phase_cut_ori[idx2] 
    de_f_cut_sort = de_f_cut[idx2] 

    model_phase_cut = np.concatenate(model_phase_cut_list) 
    model_de_f_cut = np.concatenate(model_de_f_cut_list)
    idx2 = np.argsort(model_phase_cut)
    model_phase_cut_sort = model_phase_cut[idx2]
    model_de_f_cut_sort = model_de_f_cut[idx2]
    model_de_f_max = np.max(model_de_f_cut_sort)
    model_de_f_cut_sort = model_de_f_cut_sort - model_de_f_max + 1
    de_f_cut_sort = de_f_cut_sort - model_de_f_max + 1

    # Additional outputs
    n_seg = len(g_uni_final)
    in_num = np.sum(inbool)
    cut_num = len(cut_index)

    nn = int(sig_order / 2.0) 
    covar_cut = covar[: nn, : nn]   
    diff_arr = np.array([1] * nn)
    depth = np.sum(L[: nn]) 
    depth_sigma = (diff_arr @ covar_cut @ diff_arr.T) ** 0.5
    depth_snr = depth / depth_sigma
        
    return t_cut_list, phase_cut_list, f_cut_list, model_t_cut_list, model_phase_cut_list, model_f_cut_list, phase_cut_sort, de_f_cut_sort, model_phase_cut_sort, model_de_f_cut_sort, n_seg, in_num, cut_num, depth, depth_sigma, depth_snr

def result_fitting_limitA(P, d, tm0, t, f, half_window, trend_order, cri0, 
                          get_X_sig_cut, get_X_trend_cut, inv_df, wF, sig_order, dtmin):
    """
    Perform segmented model fitting at the best parameters with limiting signal coefficients.

    Parameters
    ----------
    P : float
        The sampled orbital period (in days).
    d : float
        The sampled transit duration (in days).
    tm0 : float
        The sampled mid-transit time (in days).
    t : numpy.ndarray
        Time array of the observations (in days).
    f : numpy.ndarray
        Flux array.
    half_window : float
        Half width of the time window around each transit (in days).
    trend_order : int
        Polynomial order of the trend model to fit.
    cri0 : float
        Critical threshold used during data selection.
    get_X_sig_cut : function
        Function that returns the design matrix for the signal model.
    get_X_trend_cut : function
        Function that returns the design matrix for the trend model.
    inv_df : numpy.ndarray
        Inverse of the uncertainty of the flux measurements.
    wF : numpy.ndarray
        Weighted flux array.
    sig_order : int
        Polynomial order of the signal component.
    dtmin : float
        Minimum time interval in the time series.

    Returns
    -------
    t_cut_list : list of numpy.ndarray
        Time arrays for each fitted segment.
    phase_cut_list : list of numpy.ndarray
        Phase values for each segment.
    f_cut_list : list of numpy.ndarray
        Observed flux for each segment.
    model_t_cut_list : list of numpy.ndarray
        Model time arrays.
    model_phase_cut_list : list of numpy.ndarray
        Model phase arrays.
    model_f_cut_list : list of numpy.ndarray
        Model flux arrays.
    phase_cut_sort : numpy.ndarray
        Sorted phase arrays.
    de_f_cut_sort : numpy.ndarray
        Sorted flux arrays after subtracting trend component.
    model_phase_cut_sort : numpy.ndarray
        Sorted model phase arrays.
    model_de_f_cut_sort : numpy.ndarray
        Sorted model flux arrays after subtracting trend component.
    n_seg : int
        Number of separate phase segments fitted.
    in_num : int
        Number of in-window data points (phase in [-1, 1]).
    cut_num : int
        Total number of data points used.
    depth : float
        Estimated transit depth.
    depth_sigma : float
        Uncertainty in the estimated depth.
    depth_snr : float
        Signal-to-noise ratio of the depth estimate.
    """

    q = d / P
    half_phase_window = half_window / P

    # Get phase and group index for each timestamp
    phase, group = get_phase(t, tm0, P)

    # Select all data points within the phase window
    cut_index = np.where((phase >= 0.5 - half_phase_window) & (phase <= 0.5 + half_phase_window))[0] 
    valid_mask = np.ones(len(cut_index), dtype=bool) 

    # Further refine selection using a fixed window selection method
    cut_index, group_cut, phase_cut_ori, g_uni_final = selection_fixed_window(phase, group, q, half_phase_window, trend_order, P, cri0, valid_mask, cut_index)

    phase_cut = (2.0 * phase_cut_ori - 1.0) / q
    
    t_cut = t[cut_index]
    f_cut = f[cut_index]

    # Build signal design matrix
    inbool = (phase_cut >= -1.0) & (phase_cut <= 1.0)
    y_cut = np.where(inbool, phase_cut, 1.0)
    X_sig_cut = get_X_sig_cut(y_cut) 

    # Build trend design matrix
    X0 = get_X0_trend(len(phase_cut), g_uni_final, group_cut) 
    X_trend_cut = get_X_trend_cut(phase_cut.reshape(len(phase_cut), 1), X0)

    # Build weighted design matrix
    inv_df_cut = inv_df[cut_index]
    wX_cut = get_wX(X_sig_cut, X_trend_cut, inv_df_cut)
    wF_cut = wF[cut_index]
    
    # Fit the model: calculate covariance and coefficients
    covar = get_covar(wX_cut) 
    L = covar @ wX_cut.T @ wF_cut

    if sig_order == 2:
        L_sig = np.array([L[0]])
    if sig_order == 4:
        L_sig = L[0: 2]
    bb = 0
    if sig_order == 4:
        if ((L[0] / L[1] > -2.0) & (L[0] > 0.0) & (L[1] < 0.0)): 
            yyin = -2.0 * phase_cut ** 2 + phase_cut ** 4
            yyout = -1.0
            y2 = np.where(inbool, yyin, yyout)
            X_sig_cut2 = y2.reshape(len(y2), 1)
            wX_cut = get_wX(X_sig_cut2, X_trend_cut, inv_df_cut)
            covar = get_covar(wX_cut) 
            L = covar @ wX_cut.T @ wF_cut 
            depth = -np.copy(L[0]) 
            depth_sigma = np.copy(covar[0, 0])**0.5
            bb = 1
            L_sig = np.array([L[0]])
        else:
            nn = int(sig_order / 2.0) 
            covar_cut = covar[: nn, : nn]   
            diff_arr = np.array([1] * nn)
            depth = np.sum(L[: nn]) 
            depth_sigma = (diff_arr @ covar_cut @ diff_arr.T) ** 0.5
    else:
        nn = int(sig_order / 2.0) 
        covar_cut = covar[: nn, : nn]   
        diff_arr = np.array([1] * nn)
        depth = np.sum(L[: nn]) 
        depth_sigma = (diff_arr @ covar_cut @ diff_arr.T) ** 0.5    

    nL_trend = (X_trend_cut.shape)[1] 
    L_trend = L[-nL_trend:]
    L_trend_reshape = L_trend.reshape(int(trend_order + 1), len(L_trend) // (trend_order + 1)) 

    phase_cut_list = []
    f_cut_list = [] 
    t_cut_list = []
    
    model_t_cut_list = []
    model_phase_cut_list = []
    model_f_cut_list = []

    model_de_f_cut_list = []
    aa = 0
    if bb == 0:
        for s in g_uni_final:
            idx = np.where(group_cut == s)
            phase_cut_list.append(phase_cut_ori[idx])
            t0 = t_cut[idx]
            t_cut_list.append(t0)
            f_cut_list.append(f_cut[idx])
            model_t0 = np.arange(t0[0], t0[-1], 0.1 * dtmin) 
            model_t_cut_list.append(model_t0)

            model_phase0, _ = get_phase(model_t0, tm0, P)
            model_phase_cut_list.append(model_phase0)
            model_phase0_fit = (2.0 * model_phase0 - 1.0) / q
            inbool_model = (model_phase0_fit >= -1.0) & (model_phase0_fit <= 1.0)
            y_cut_model = np.where(inbool_model, model_phase0_fit, 1.0)
            X_sig_cut_model = get_X_sig_cut(y_cut_model) 

            X_trend_cut_model = np.vstack([model_phase0_fit ** i for i in range(trend_order + 1)]).T
            X_model = np.hstack((X_sig_cut_model, X_trend_cut_model))
            L_trend_here = L_trend_reshape[:, aa]
            L_here = np.concatenate((L_sig, L_trend_here))
            model_f0 = X_model @ L_here
            model_f_cut_list.append(model_f0)
            model_f_trend0 = X_trend_cut_model @ L_trend_here
            model_de_f0 = model_f0 - model_f_trend0
            model_de_f_cut_list.append(model_de_f0)
            aa += 1
    else:
        for s in g_uni_final:
            idx = np.where(group_cut == s)
            phase_cut_list.append(phase_cut_ori[idx])
            t0 = t_cut[idx]
            t_cut_list.append(t0)
            f_cut_list.append(f_cut[idx])
            model_t0 = np.arange(t0[0], t0[-1], 0.1 * dtmin)
            model_t_cut_list.append(model_t0)

            model_phase0,_ = get_phase(model_t0, tm0, P)
            model_phase_cut_list.append(model_phase0)
            model_phase0_fit = (2.0 * model_phase0 - 1.0) / q
            inbool_model = (model_phase0_fit >= -1.0) & (model_phase0_fit <= 1.0)

            yyin_model = -2.0 * model_phase0_fit ** 2 + model_phase0_fit ** 4
            yyout_model = -1.0
            y2_model = np.where(inbool_model, yyin_model, yyout_model)
            X_sig_cut_model = y2_model.reshape(len(y2_model), 1)            
            X_trend_cut_model = np.vstack([model_phase0_fit ** i for i in range(trend_order + 1)]).T
            X_model = np.hstack((X_sig_cut_model, X_trend_cut_model))
            L_trend_here = L_trend_reshape[: , aa]
            L_here = np.concatenate((L_sig, L_trend_here))
            model_f0 = X_model @ L_here
            model_f_cut_list.append(model_f0)
            model_f_trend0 = X_trend_cut_model @ L_trend_here
            model_de_f0 = model_f0 - model_f_trend0
            model_de_f_cut_list.append(model_de_f0)
            aa += 1

    f_trend = X_trend_cut @ L_trend
    de_f_cut = f_cut - f_trend 

    # Sort for plotting
    idx2 = np.argsort(phase_cut_ori)
    phase_cut_sort = phase_cut_ori[idx2] 
    de_f_cut_sort = de_f_cut[idx2] 

    model_phase_cut = np.concatenate(model_phase_cut_list) 
    model_de_f_cut = np.concatenate(model_de_f_cut_list)
    idx2 = np.argsort(model_phase_cut)
    model_phase_cut_sort = model_phase_cut[idx2]
    model_de_f_cut_sort = model_de_f_cut[idx2]
    model_de_f_max = np.max(model_de_f_cut_sort)
    print(np.max(model_de_f_cut_sort))
    model_de_f_cut_sort = model_de_f_cut_sort - model_de_f_max + 1
    de_f_cut_sort = de_f_cut_sort - model_de_f_max + 1

    # Additional outputs
    n_seg = len(g_uni_final)
    in_num = np.sum(inbool)
    cut_num = len(cut_index)

    depth_snr = depth / depth_sigma
        
    return t_cut_list, phase_cut_list, f_cut_list, model_t_cut_list, model_phase_cut_list, model_f_cut_list, phase_cut_sort, de_f_cut_sort, model_phase_cut_sort, model_de_f_cut_sort, n_seg, in_num, cut_num, depth, depth_sigma, depth_snr

