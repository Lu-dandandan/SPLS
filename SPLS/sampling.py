import numpy as np
from SPLS.constants import Msun,Rsun,d_to_s,G,A_up,A_down 
from numpy import pi


def d_grid(P, d_sam):
    """
    This function restricts `d_sam` based on the bounds determined by the parameter `P`.

    The bounds are calculated using the constants `A_up` and `A_down`, and are applied to create a subset of 
    `d_sam` that lies within the range of the calculated lower and upper bounds.

    Parameters:
    -----------
    P : float or numpy.ndarray
        The sampled period(s) (in days).
    d_sam : numpy.ndarray
        The sampled transit duration in linear search.

    Returns:
    -----------
    numpy.ndarray: The subset of `d_sam` that lies within the calculated bounds.
    numpy.ndarray: The indices of the elements in `d_sam` that lie within the bounds.
    """
    d_up = A_up * P ** (1 / 3.0) 
    dmax2 = np.min(np.array([d_up, d_sam[-1]]))
    d_down = A_down * P ** (1 / 3.0)
    dmin2 = np.max(np.array([d_sam[0], d_down]))
    idx_d = np.where((d_sam >= dmin2) & (d_sam <= dmax2))[0]
    return d_sam[idx_d], idx_d



def dp_fun(Rs, Ms, S, OS_P, P):
    """
    Compute ΔP

    Parameters:
    -----------
    Rs : float
        The radius of the star in solar radii (Rsun).
    Ms : float
        The mass of the star in solar masses (Msun).
    S  : float
        Time span (in days).
    OS_P : float
        The oversampling parameter (range: 2-5) adjusts the resolution.
    P  : float or numpy.ndarray
        Orbital period (in days).

    Returns:
    -----------
    dp : float or numpy.ndarray
        The computed dp value after applying the formula.
    """

    Ms *= Msun 
    Rs *= Rsun  
    S = S * d_to_s
    P = P * d_to_s
    A = (2.0 * pi) ** (2.0 / 3.0) / pi * Rs / (G * Ms) ** (1.0 / 3.0) / S / OS_P
    dp = A * P ** (4.0/3.0)
    return dp / d_to_s

def P_grid(Rs, Ms, S, OS_P, Pmin, Pmax): 
    """
    Generate a grid of periods based on stellar radius, mass, and time span.

    The function performs cubic sampling of the frequency and returns periods within the specified minimum and maximum values.

    Parameters:
    ----------
    Rs : float
        Stellar radius in solar radii (R_sun).
    Ms : float
        Stellar mass in solar masses (M_sun).
    S : float
        Time span (in days).
    OS_P : float
        The oversampling parameter (range: 2-5) adjusts the resolution.
    Pmin : float
        Minimum period (in days) for the output.
    Pmax : float
        Maximum period (in days) for the output.

    Returns:
    -------
    Per : numpy.ndarray
        An array of orbital periods (in days) within the specified range [Pmin, Pmax].
    
    """

    Ms *= Msun 
    Rs *= Rsun  
    S = S * d_to_s

    fre_min = 2.0 / S
    fre_max = (G * Ms / (3 * Rs) ** 3) ** 0.5 / 2.0 / pi

    # Cubic sampling of frequency
    A = (2.0 * pi) ** (2 / 3.0) / pi * Rs / (G * Ms) ** (1 / 3.0) / S / OS_P
    C = fre_min ** (1 / 3.0) - A / 3.0
    N = (fre_max ** (1 / 3.0) - fre_min ** (1 / 3.0) + A / 3.0) * 3.0 / A
    x = np.arange(N) + 1
    fre = (A / 3.0 * x + C) ** 3
    Per = 1 / fre / d_to_s 
    return Per[(Per <= Pmax) & (Per >= Pmin)] 