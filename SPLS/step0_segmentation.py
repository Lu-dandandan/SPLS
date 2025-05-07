import numpy as np   
from numba import njit
import matplotlib.pyplot as plt       
from matplotlib import rcParams; rcParams["figure.dpi"] = 250   


def get_segs_time(t, f, df, gap_time_threshold, gap_min_num_in_one_seg):
    """
    Segment time-series data based on time gaps.

    Segments are defined where the time difference exceeds a threshold (`gap_time_threshold * cadence`). Segments shorter than 
    a minimum number of points are discarded.

    Parameters:
    -----------
    t : numpy.ndarray
        Time array in days (must be sorted in ascending order).
    f : numpy.ndarray
        Corresponding flux values.
    df : numpy.ndarray
        Corresponding flux uncertainties.
    gap_time_threshold : float
        Threshold multiplier for detecting time gaps. The gap between consecutive segments must exceed 
        `gap_time_threshold * cadence`.
    gap_min_num_in_one_seg : int
        Minimum number of data points required to keep a segment.

    Returns:
    --------
    t_segs : list of numpy.ndarray
        List of time segments (in days).
    f_segs : list of numpy.ndarray
        List of flux segments.
    df_segs : list of numpy.ndarray
        List of flux uncertainty segments.
    dt_median : float
        Median time interval used for gap detection.
    """

    # Calculate time differences and their median
    dt = np.diff(t)
    dt_median = np.median(dt)

    # Identify indices where time gaps exceed threshold
    split_indices = np.where(dt > dt_median * gap_time_threshold)[0] + 1
    
    # Generate segment index pairs (start, end)
    start_indices = np.insert(split_indices, 0, 0)
    end_indices = np.append(split_indices, len(t))
    segments = list(zip(start_indices, end_indices))
    
    # Keep only valid segments with enough data points
    valid_segments = [(s, e) for s, e in segments if (e - s) > gap_min_num_in_one_seg]
    
    # Extract time, flux, and uncertainty segments
    t_segs = [t[s:e] for s, e in valid_segments]
    f_segs = [f[s:e] for s, e in valid_segments]
    df_segs = [df[s:e] for s, e in valid_segments]

    print(f'The light curve is divided into {len(t_segs)} segments according to time gaps.')
    return t_segs, f_segs, df_segs, dt_median

@njit
def compute_outliers(delta_f, mad_threshold):
    """
    Identify outliers in delta_f based on a MAD-based (Median Absolute Deviation) threshold.

    Parameters:
    -----------
    delta_f : numpy.ndarray
        Array of flux differences (np.diff(flux)).
    mad_threshold : float
        Multiplier for the robust standard deviation to define the outlier threshold.

    Returns:
    --------
    outlier_indices : numpy.ndarray
        Indices of delta_f where the change exceeds the MAD-based threshold.
    threshold : float
        The absolute deviation threshold used for detecting outliers.
    median : float
        The median of delta_f, used as the central value.
    """
    
    # Compute the median of the delta_f array
    median = np.median(delta_f)

    # Compute the Median Absolute Deviation (MAD)
    mad = np.median(np.abs(delta_f - median))

    # Calculate the outlier threshold using a robust estimate of standard deviation
    threshold = mad_threshold * 1.4826 * mad # 1.4826 is the consistency constant for normal distributions
    
    # Identify indices where the absolute deviation exceeds the threshold
    return np.where(np.abs(delta_f - median) > threshold)[0], threshold, median 


def get_segs_time_flux(t_segs, f_segs, df_segs, gap_delta_flux_mad_threshold, gap_time_threshold, gap_min_num_in_one_seg, dt_median):    
    """
    Further clean and segment light curve data by removing points near flux gaps (Δflux outliers).
    After cleaning, segments are re-split based on time gaps and short segments are discarded.

    Parameters:
    ----------
    t_segs : list of numpy.ndarray
        Original list of time segments (in days).
    f_segs : list of numpy.ndarray
        Corresponding flux values for each time segment.
    df_segs : list of numpy.ndarray
        Corresponding flux uncertainties for each time segment.
    gap_delta_flux_mad_threshold : float
        Threshold for Δflux outlier detection (in units of MAD).
    gap_time_threshold : float
        Threshold multiplier for time gap-based segmentation.
    gap_min_num_in_one_seg : int
        Minimum number of points for a valid segment.
    dt_median : float
        Median of the time intervals.

    Returns:
    -------
    t_segs_final : list of numpy.ndarray
        Final cleaned and segmented time arrays (in days).
    f_segs_final : list of numpy.ndarray
        Final cleaned and segmented flux arrays.
    df_segs_final : list of numpy.ndarray
        Final cleaned and segmented flux uncertainty arrays.

    Notes: 
    -----
    If a flux gap (Δflux outlier) causes a subsequent time gap, the function will generate plots for visual inspection.

    """

    t_segs_final = [] 
    f_segs_final = []
    df_segs_final = []

    for i in range(len(t_segs)):
        t_seg = t_segs[i]
        f_seg = f_segs[i]
        df_seg = df_segs[i]

        # Step 1: Detect large flux gaps (Δflux outliers) in this segment
        delta_f = np.diff(f_seg)
        is_outlier, gap_flux_threshold, median_delta_f = compute_outliers(delta_f, gap_delta_flux_mad_threshold)
        
        # Compute the mid-point times of each flux gap (between points)        
        outlier_times = (t_seg[is_outlier] + t_seg[np.array(is_outlier + 1, dtype=int)])/2.0 

        # Step 2: Create a mask to remove all data within a window centered on outlier times
        keep_mask = np.ones_like(t_seg, dtype=bool) 
        for t_outlier in outlier_times: 
            in_window = (t_seg >= t_outlier - dt_median * gap_time_threshold / 2.0) & (t_seg <= t_outlier + dt_median * gap_time_threshold / 2.0)
            keep_mask &= ~in_window

        # Step 3: Clean the data by applying the mask
        t_clean = t_seg[keep_mask]
        f_clean = f_seg[keep_mask]
        df_clean = df_seg[keep_mask]  

        # Step 4: Further segment the cleaned data based on time gaps
        dt = np.diff(t_clean)
        split_indices = np.where(dt > dt_median * gap_time_threshold)[0] + 1
        
        if len(split_indices) > 0:
            # If there are large time gaps after cleaning which is caused by flux gaps, visualize and split
            fig, ax = plt.subplots(1, 2, figsize = (10, 4))
            
            # Plot Time-ΔFlux
            ax[1].plot((t_seg[:-1] + t_seg[1:]) / 2.0, delta_f, c = 'black')            
            ax[1].axhline(median_delta_f + gap_flux_threshold, c = '0.75', label = 'thresholds')
            ax[1].axhline(median_delta_f - gap_flux_threshold, c = '0.75')
            ax[1].set_xlabel('Time')
            ax[1].set_ylabel(r'$\Delta$ Flux')
            ax[1].set_title(r'Time - $\Delta$ Flux')
            ax[1].legend()
            
            # Split cleaned data into subsegments
            t_cleans = np.split(t_clean, split_indices)
            f_cleans = np.split(f_clean, split_indices)
            df_cleans = np.split(df_clean, split_indices) 

            # Plot Time-Flux
            ax[0].plot(t_seg, f_seg, c='r', label = 'deleted data near flux gaps') 
            for j in range(len(t_cleans)):
                t_clean0 = t_cleans[j]
                f_clean0 = f_cleans[j]
                df_clean0 = df_cleans[j] 

                # Only keep segments that have enough points
                if len(t_clean0) > gap_min_num_in_one_seg:
                    t_segs_final.append(t_clean0)
                    f_segs_final.append(f_clean0)
                    df_segs_final.append(df_clean0)  
                ax[0].plot(t_clean0, f_clean0, c = 'black')
            ax[0].set_xlabel('Time')
            ax[0].set_ylabel('Flux') 
            ax[0].set_title('Time - Flux')
            ax[0].legend() 
            fig.suptitle('The segment containing flux gaps') 
            fig.tight_layout()
            plt.show()       
        else:
            # If no further split needed, just add the cleaned segment
            t_segs_final.append(t_clean)
            f_segs_final.append(f_clean)
            df_segs_final.append(df_clean) 

    print(f'The light curve is further divided into {len(t_segs_final)} segments according to flux gaps.')
    return t_segs_final, f_segs_final, df_segs_final