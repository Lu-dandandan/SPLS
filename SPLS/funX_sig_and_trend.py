import numpy as np 
from numba import njit

@njit
def fun(f):
    return f

@njit
def sig_order2(x_cut):
    """
    Generate a 2D feature array with one column: x^2.

    Parameters
    ----------
    x_cut : 1D numpy array
        Input array of values.

    Returns
    -------
    X_sig_cut : 2D numpy array of shape (n, 1)
        Output array where each row is [x_i^2].
    """
    n = len(x_cut)
    X_sig_cut = np.empty((n, 1))
    for i in range(n):
        X_sig_cut[i, 0] = x_cut[i] * x_cut[i]
    return X_sig_cut

@njit
def sig_order4(x_cut):
    """
    Generate 2D feature array with columns: [x^2, x^4].

    Parameters
    ----------
    x_cut : 1D array
        Input vector.

    Returns
    -------
    X_sig_cut : 2D array of shape (n, 2)
        Feature array with columns [x^2, x^4].
    """ 
    n = len(x_cut)
    X_sig_cut = np.empty((n, 2))
    for i in range(n):
        x2 = x_cut[i] * x_cut[i]
        X_sig_cut[i, 0] = x2
        X_sig_cut[i, 1] = x2 * x2
    return X_sig_cut

@njit
def sele_sig(sig_order): 
    if sig_order == 2:
        return fun(sig_order2)
    elif sig_order == 4:
        return fun(sig_order4)
    else:  
        raise ValueError("The provided sig_order is out of range. Please provide an integer in [2, 4].")


@njit
def trend_order0(t_cut):
    n = len(t_cut)
    X_trend = np.empty((n, 1)) 
    for i in range(n):
        X_trend[i, 0] = 1.0
    return X_trend

@njit
def trend_order1(t_cut):  
    n = len(t_cut)
    X_trend = np.empty((n, 2)) 
    for i in range(n):
        t = t_cut[i]
        X_trend[i, 0] = 1.0
        X_trend[i, 1] = t
    return X_trend

@njit
def trend_order2(t_cut):
    n = len(t_cut)
    X_trend = np.empty((n, 3)) 
    for i in range(n):
        t = t_cut[i]
        X_trend[i, 0] = 1.0
        X_trend[i, 1] = t
        X_trend[i, 2] = t * t
    return X_trend

@njit
def trend_order3(t_cut):
    n = len(t_cut)
    X_trend = np.empty((n, 4)) 
    for i in range(n):
        t = t_cut[i]
        X_trend[i, 0] = 1.0
        X_trend[i, 1] = t
        X_trend[i, 2] = t * t
        X_trend[i, 3] = t * t * t
    return X_trend

@njit
def trend_order4(t_cut):
    n = len(t_cut)
    X_trend = np.empty((n, 5)) 
    for i in range(n):
        t = t_cut[i]
        X_trend[i, 0] = 1.0
        X_trend[i, 1] = t
        X_trend[i, 2] = t * t
        X_trend[i, 3] = t * X_trend[i, 2]
        X_trend[i, 4] = t * X_trend[i, 3]
    return X_trend

@njit
def trend_order5(t_cut):
    n = len(t_cut)
    X_trend = np.empty((n, 6)) 
    for i in range(n):
        t = t_cut[i]
        X_trend[i, 0] = 1.0
        X_trend[i, 1] = t
        X_trend[i, 2] = t * t
        X_trend[i, 3] = t * X_trend[i, 2]
        X_trend[i, 4] = t * X_trend[i, 3]
        X_trend[i, 5] = t * X_trend[i, 4] 
    return X_trend

@njit
def trend_order6(t_cut):
    n = len(t_cut)
    X_trend = np.empty((n, 7)) 
    for i in range(n):
        t = t_cut[i]
        X_trend[i, 0] = 1.0
        X_trend[i, 1] = t
        X_trend[i, 2] = t * t
        X_trend[i, 3] = t * X_trend[i, 2]
        X_trend[i, 4] = t * X_trend[i, 3]
        X_trend[i, 5] = t * X_trend[i, 4]
        X_trend[i, 6] = t * X_trend[i, 5]
    return X_trend

@njit
def trend_order7(t_cut):
    n = len(t_cut)
    X_trend = np.empty((n, 8)) 
    for i in range(n):
        t = t_cut[i]
        X_trend[i, 0] = 1.0
        X_trend[i, 1] = t
        X_trend[i, 2] = t * t
        X_trend[i, 3] = t * X_trend[i, 2]
        X_trend[i, 4] = t * X_trend[i, 3]
        X_trend[i, 5] = t * X_trend[i, 4]
        X_trend[i, 6] = t * X_trend[i, 5]
        X_trend[i, 7] = t * X_trend[i, 6]
    return X_trend


@njit
def trend_order8(t_cut):
    n = len(t_cut)
    X_trend = np.empty((n, 9)) 
    for i in range(n):
        t = t_cut[i]
        X_trend[i, 0] = 1.0
        X_trend[i, 1] = t
        X_trend[i, 2] = t * t
        X_trend[i, 3] = t * X_trend[i, 2]
        X_trend[i, 4] = t * X_trend[i, 3]
        X_trend[i, 5] = t * X_trend[i, 4]
        X_trend[i, 6] = t * X_trend[i, 5]
        X_trend[i, 7] = t * X_trend[i, 6]
        X_trend[i, 8] = t * X_trend[i, 7]
    return X_trend

@njit
def sele_trend(trend_order): 
    if trend_order == 0:
        return fun(trend_order0)
    elif trend_order == 1:
        return fun(trend_order1)
    elif trend_order == 2:
        return fun(trend_order2)
    elif trend_order == 3:
        return fun(trend_order3)
    elif trend_order == 4:
        return fun(trend_order4) 
    elif trend_order == 5:
        return fun(trend_order5)
    elif trend_order == 6:
        return fun(trend_order6)
    elif trend_order == 7:
        return fun(trend_order7)
    elif trend_order == 8:
        return fun(trend_order8) 
    else:
        raise ValueError("The provided trend_order is out of range. Please provide an integer in [0, 1, 2, 3, 4, 5, 6, 7, 8].")




@njit
def global_trend_order0(phase_cut,X0):
    return X0

@njit
def global_trend_order1(phase_cut,X0): 
    n, m = X0.shape
    X_trend = np.empty((n, m * 2))
    X_trend[:, :m] = X0
    X_trend[:, m:2*m] = X0 * phase_cut
    return X_trend

@njit
def global_trend_order2(phase_cut,X0):
    n, m = X0.shape
    X_trend = np.empty((n, m * 3))
    X_trend[:, :m] = X0
    X_trend[:, m:2*m] = X0 * phase_cut
    X_trend[:, 2*m:3*m] = X_trend[:, m:2*m]* phase_cut
    return X_trend

@njit
def global_trend_order3(phase_cut,X0):
    n, m = X0.shape
    X_trend = np.empty((n, m * 4))
    X_trend[:, :m] = X0
    X_trend[:, m:2*m] = X0 * phase_cut
    X_trend[:, 2*m:3*m] = X_trend[:, m:2*m] * phase_cut
    X_trend[:, 3*m:4*m] = X_trend[:, 2*m:3*m] * phase_cut
    return X_trend


@njit
def global_trend_order4(phase_cut,X0):
    n, m = X0.shape
    X_trend = np.empty((n, m * 5))
    X_trend[:, :m] = X0
    X_trend[:, m:2*m] = X0 * phase_cut
    X_trend[:, 2*m:3*m] = X_trend[:, m:2*m] * phase_cut
    X_trend[:, 3*m:4*m] = X_trend[:, 2*m:3*m] * phase_cut
    X_trend[:, 4*m:5*m] = X_trend[:, 3*m:4*m] * phase_cut
    return X_trend

@njit
def global_trend_order5(phase_cut,X0):
    n, m = X0.shape
    X_trend = np.empty((n, m * 6))
    X_trend[:, :m] = X0
    X_trend[:, m:2*m] = X0 * phase_cut
    X_trend[:, 2*m:3*m] = X_trend[:, m:2*m] * phase_cut
    X_trend[:, 3*m:4*m] = X_trend[:, 2*m:3*m] * phase_cut
    X_trend[:, 4*m:5*m] = X_trend[:, 3*m:4*m] * phase_cut
    X_trend[:, 5*m:6*m] = X_trend[:, 4*m:5*m] * phase_cut
    return X_trend

@njit
def global_trend_order6(phase_cut,X0):
    n, m = X0.shape
    X_trend = np.empty((n, m * 7))
    X_trend[:, :m] = X0
    X_trend[:, m:2*m] = X0 * phase_cut
    X_trend[:, 2*m:3*m] = X_trend[:, m:2*m] * phase_cut
    X_trend[:, 3*m:4*m] = X_trend[:, 2*m:3*m] * phase_cut
    X_trend[:, 4*m:5*m] = X_trend[:, 3*m:4*m] * phase_cut
    X_trend[:, 5*m:6*m] = X_trend[:, 4*m:5*m] * phase_cut
    X_trend[:, 6*m:7*m] = X_trend[:, 5*m:6*m] * phase_cut
    return X_trend

@njit
def global_trend_order7(phase_cut,X0):
    n, m = X0.shape
    X_trend = np.empty((n, m * 8))
    X_trend[:, :m] = X0
    X_trend[:, m:2*m] = X0 * phase_cut
    X_trend[:, 2*m:3*m] = X_trend[:, m:2*m] * phase_cut
    X_trend[:, 3*m:4*m] = X_trend[:, 2*m:3*m] * phase_cut
    X_trend[:, 4*m:5*m] = X_trend[:, 3*m:4*m] * phase_cut
    X_trend[:, 5*m:6*m] = X_trend[:, 4*m:5*m] * phase_cut
    X_trend[:, 6*m:7*m] = X_trend[:, 5*m:6*m] * phase_cut
    X_trend[:, 7*m:8*m] = X_trend[:, 6*m:7*m] * phase_cut
    return X_trend

@njit
def global_trend_order8(phase_cut,X0):
    n, m = X0.shape
    X_trend = np.empty((n, m * 9))
    X_trend[:, :m] = X0
    X_trend[:, m:2*m] = X0 * phase_cut
    X_trend[:, 2*m:3*m] = X_trend[:, m:2*m] * phase_cut
    X_trend[:, 3*m:4*m] = X_trend[:, 2*m:3*m] * phase_cut
    X_trend[:, 4*m:5*m] = X_trend[:, 3*m:4*m] * phase_cut
    X_trend[:, 5*m:6*m] = X_trend[:, 4*m:5*m] * phase_cut
    X_trend[:, 6*m:7*m] = X_trend[:, 5*m:6*m] * phase_cut
    X_trend[:, 7*m:8*m] = X_trend[:, 6*m:7*m] * phase_cut
    X_trend[:, 8*m:9*m] = X_trend[:, 7*m:8*m] * phase_cut
    return X_trend
 
@njit
def sele_global_trend(trend_order): 
    if trend_order == 0:
        return fun(global_trend_order0)
    elif trend_order == 1:
        return fun(global_trend_order1)
    elif trend_order == 2:
        return fun(global_trend_order2)
    elif trend_order == 3:
        return fun(global_trend_order3)
    elif trend_order == 4:
        return fun(global_trend_order4) 
    elif trend_order == 5:
        return fun(global_trend_order5)
    elif trend_order == 6:
        return fun(global_trend_order6)
    elif trend_order == 7:
        return fun(global_trend_order7)
    elif trend_order == 8:
        return fun(global_trend_order8)
    else:
        raise ValueError("The provided trend_order is out of range. Please provide an integer in [0, 1, 2, 3, 4, 5, 6, 7, 8].")    