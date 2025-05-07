
import numpy as np 
from numba import njit


def fun(f):
    return f

def sele_fitting_2D(A_limit): 
    if A_limit:
        return fun(fitting_2D_limitA)
    else:
        return fun(fitting_2D_no_limitA)

@njit
def get_wX(X_sig_cut, X_trend_cut, inv_df_cut):
    # Construct weighted design matrix from signal and trend components
    X_cut = np.hstack((X_sig_cut, X_trend_cut))
    return X_cut * inv_df_cut # Weight each row by 1/sigma

@njit
def get_covar(wX_cut):
    # Compute inverse of X^T X (used for least-squares estimation)
    xtx = wX_cut.T @ wX_cut
    return np.linalg.inv(xtx)

@njit
def get_L(wX_cut, wF_cut): 
    # Compute least-squares solution L = (X^T X)^(-1) X^T y
    var = get_covar(wX_cut)
    return var @ wX_cut.T @ wF_cut

@njit
def get_chi2_base(X_trend_cut, inv_df_cut, wF_cut):
    # Compute chi-squared of baseline model using trend components only
    wX_base = X_trend_cut * inv_df_cut
    L = get_L(wX_base, wF_cut)
    wR_base = wF_cut - wX_base @ L # residual
    return wR_base.T @ wR_base


def fitting_2D_no_limitA(i, seg_ts, seg_fs, seg_dfs,
                        size_d, d_sam, tm_sam,
                        half_window,
                        sig_order, trend_order,
                        get_X_sig_cut, get_X_trend_cut):
    """
    Compute the log-likelihood improvement ΔlnL across a 2D grid without the limitation of signal coefficients.
    For a given segment of data, using weighted least squares regression.

    Parameters
    ----------
    i : int
        Index of the current data segment.
    seg_ts : list of numpy.array
        List of timestamp arrays.
    seg_fs : list of numpy.array
        List of flux arrays.
    seg_dfs : list of numpy.array
        List of uncertainty arrays (standard deviation).
    size_d : int
        Number of sampled transit duration.
    d_sam : numpy.array
        Sampled transit duration (in days).
    tm_sam : numpy.array
        Sampled mid-transit times (in days).
    half_window : float
        Half of the fixed time window around each center time for extracting data (in days).
    sig_order : int
        Polynomial order of the signal component.
    trend_order : int
        Polynomial order of the trend component.
    get_X_sig_cut : function
        Function to generate the design matrix for the signal component given x_cut.
    get_X_trend_cut : function
        Function to generate the design matrix for the trend component given t_cut.

    Returns
    -------
    dlnL_here : numpy.ndarray (size_d × len(tm_sam_in_each_segment))
        Matrix of ΔlnL values comparing the full model with signal vs. trend-only model,
        for each d × time combination. NaNs are used where data is insufficient.
    """

    seg_t0 = seg_ts[i]  
    seg_f0 = seg_fs[i]
    seg_df0 = seg_dfs[i]
    inv_df = (1.0 / seg_df0)[:, None]
    wF = seg_f0 / seg_df0

    # Time mask for current segment's time window
    idx_tm2 = np.where((tm_sam >= seg_t0[0]) & (tm_sam <= seg_t0[-1]))[0]
    tm_sam_here = tm_sam[idx_tm2]

    # Preallocate dlnL result matrix
    dlnL_here = np.full((size_d, len(idx_tm2)), np.nan) 

    for k in range(size_d):
        d = d_sam[k]
        half_d = 0.5 * d

        for j in range(len(tm_sam_here)):
            tm = tm_sam_here[j]

            # Cut data within window centered at tm
            cut_idx = np.where((seg_t0 >= (tm - half_window)) & (seg_t0 <= (tm + half_window)))[0]
            cut_num = len(cut_idx)

            t_cut_ori = seg_t0[cut_idx]
            left_mask = (t_cut_ori < (tm - half_d))  
            left_num = np.sum(left_mask)  
            
            in_mask = ((t_cut_ori > (tm - half_d)) & (t_cut_ori < (tm + half_d)))  
            in_num = np.sum(in_mask) 

            if in_num < (sig_order / 2.0 + 2.0):
                continue
            if left_num < trend_order + 1.0:
                continue 
            if (cut_num-in_num - left_num) < trend_order + 1.0:
                continue 
            
            # # Normalize time values in window to [-1, 1] 
            t_cut = 2.0 * (t_cut_ori - tm) / d  
            inv_df_cut = inv_df[cut_idx]
            wF_cut = wF[cut_idx]             
    
            inbool = (t_cut >= -1.0) & (t_cut <= 1.0)
            x_cut = np.where(inbool, t_cut, 1.0)

            # Construct design matrices
            X_sig_cut = get_X_sig_cut(x_cut) 
            X_trend_cut = get_X_trend_cut(t_cut)

            # Compute weighted design matrix and solve for L
            wX_cut = get_wX(X_sig_cut, X_trend_cut, inv_df_cut)
            wF_cut = np.asarray(wF_cut, dtype=np.float64)
            L = get_L(wX_cut, wF_cut)

            # Compute likelihood improvement (Δ lnL)
            chi2_base = get_chi2_base(X_trend_cut, inv_df_cut, wF_cut)
            wR = wF_cut - wX_cut @ L
            chi2 = wR.T @ wR 
            dlnL_here[k, j] = 0.5 * (chi2_base - chi2)

    # If not the first segment, pad on the left with NaNs
    if i != 0: 
        idx_tm1 = np.where((tm_sam > seg_ts[i - 1][-1]) & (tm_sam < seg_t0[0]))[0]
        dlnL_add = np.full((size_d, len(idx_tm1)), np.nan)
        dlnL_here = np.hstack((dlnL_add, dlnL_here))
    return dlnL_here 



def fitting_2D_limitA(i, seg_ts, seg_fs, seg_dfs, 
                    size_d, d_sam, tm_sam, 
                    half_window,
                    sig_order, trend_order,
                    get_X_sig_cut, get_X_trend_cut):
    
    """
    Compute the log-likelihood improvement ΔlnL across a 2D grid with the limitation of signal coefficients.
    For a given segment of data, using weighted least squares regression.

    Parameters
    ----------
    i : int
        Index of the current data segment.
    seg_ts : list of numpy.ndarray
        List of timestamp arrays (in days).
    seg_fs : list of numpy.ndarray
        List of flux arrays.
    seg_dfs : list of numpy.ndarray
        List of uncertainty arrays (standard deviation).
    size_d : int
        Number of sampled transit duration.
    d_sam : numpy.ndarray
        Sampled transit duration (in days).
    tm_sam : numpy.ndarray
        Sampled mid-transit times (in days).
    half_window : float
        Half of the fixed time window around each center time for extracting data (in days).
    sig_order : int
        Polynomial order of the signal component.
    trend_order : int
        Polynomial order of the trend component.
    get_X_sig_cut : function
        Function to generate the design matrix for the signal component given x_cut.
    get_X_trend_cut : function
        Function to generate the design matrix for the trend component given t_cut.

    Returns
    -------
    dlnL_here : numpy.ndarray (size_d × len(tm_sam_in_each_segment))
        Matrix of ΔlnL values comparing the full model with signal vs. trend-only model,
        for each d × time combination. NaNs are used where data is insufficient.
    """
    seg_t0 = seg_ts[i]  
    seg_f0 = seg_fs[i]
    seg_df0 = seg_dfs[i]
    inv_df = (1.0 / seg_df0)[:, None]
    wF = seg_f0 / seg_df0

    # Time mask for current segment's time window
    idx_tm2 = np.where((tm_sam >= seg_t0[0]) & (tm_sam <= seg_t0[-1]))[0]
    tm_sam_here = tm_sam[idx_tm2]

    # Preallocate dlnL result matrix 
    dlnL_here = np.full((size_d, len(idx_tm2)), np.nan) 
    
    for k in range(size_d):
        d = d_sam[k]
        half_d = 0.5 * d

        for j in range(len(tm_sam_here)):
            tm = tm_sam_here[j]

            # Cut data within window centered at tm
            cut_idx = np.where((seg_t0 >= (tm - half_window)) & (seg_t0 <= (tm + half_window)))[0]
            cut_num = len(cut_idx)

            t_cut_ori = seg_t0[cut_idx] 
            left_mask = (t_cut_ori < (tm - half_d))  
            left_num = np.sum(left_mask)

            in_mask = ((t_cut_ori > (tm - half_d)) & (t_cut_ori < (tm + half_d)))  
            in_num = np.sum(in_mask)  

            if in_num < (sig_order / 2.0 + 2.0):
                continue
            if left_num < trend_order + 1.0:
                continue 
            if (cut_num - in_num - left_num) < trend_order + 1.0:
                continue 
            
            # Normalize time values in window to [-1, 1]
            t_cut = 2.0 * (t_cut_ori - tm) / d  
            inv_df_cut = inv_df[cut_idx]
            wF_cut = wF[cut_idx]             
       
            inbool = (t_cut >= -1.0) & (t_cut <= 1.0)
            x_cut = np.where(inbool, t_cut, 1.0)

            # Construct design matrices
            X_sig_cut = get_X_sig_cut(x_cut) 
            X_trend_cut = get_X_trend_cut(t_cut)

            # Compute weighted design matrix and solve for L 
            wX_cut = get_wX(X_sig_cut, X_trend_cut, inv_df_cut)
            wF_cut = np.asarray(wF_cut, dtype=np.float64)
            L = get_L(wX_cut, wF_cut)

            chi2_base = get_chi2_base(X_trend_cut, inv_df_cut, wF_cut)

            # limit signal coefficients A
            if sig_order == 2: 
                if L[0] < 0.0:
                    continue
                else:
                    wR = wF_cut - wX_cut @ L
                    chi2 = wR.T @ wR
            if sig_order == 4:  
                if (L[0] > 0.0) and (L[1] > 0.0):
                    wR = wF_cut - wX_cut @ L 
                    chi2 = wR.T @ wR
                elif (L[0] > 0.0) and (L[0] / L[1] < -2.0):  
                    wR = wF_cut - wX_cut @ L
                    chi2 = wR.T @ wR 
                elif (L[0] / L[1] > -2.0) and (L[0] / L[1] < 0.0):
                    xin = -2.0 * t_cut ** 2 + t_cut ** 4
                    xout = -1.0 
                    x2 = np.where(inbool, xin, xout)
                    wX_cut = get_wX(x2[:, None], X_trend_cut, inv_df_cut)
                    L = get_L(wX_cut, wF_cut) # A4 only 
                    if L[0] > 0.0:
                        continue
                    wR = wF_cut - wX_cut @ L
                    chi2 = wR.T @ wR
                else:
                    continue
            dlnL_here[k, j] = 0.5 * (chi2_base - chi2)

    # If not the first segment, pad on the left with NaNs    
    if i != 0: 
        idx_tm1 = np.where((tm_sam > seg_ts[i - 1][-1]) & (tm_sam < seg_t0[0]))[0]
        dlnL_add = np.full((size_d, len(idx_tm1)), np.nan)
        dlnL_here = np.hstack((dlnL_add, dlnL_here))
    return dlnL_here 