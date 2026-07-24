# Segmented-Polynomial-fitting Least Squares (SPLS)

SPLS is a Python implementation of **Segmented-Polynomial-fitting Least
Squares**, a transit-search algorithm designed for weak, long-period signals in
light curves with stellar or instrumental background trends.

SPLS fits the transit signal and local background trends simultaneously with a
segmented double-polynomial model. This avoids the signal distortion that can
occur when detrending and transit detection are performed as separate steps.

- **Paper:** [Zheng, Feng & Rui (2026), *The Astronomical Journal*, 171,
  64](https://doi.org/10.3847/1538-3881/ae2679)
- **Open-access preprint:** [arXiv:2512.02356](https://arxiv.org/abs/2512.02356)
- **Archived software release used in the paper (v1.0.0):**
  [Zenodo](https://doi.org/10.5281/zenodo.15411397)

## Method overview

SPLS is intended for photometric time series. For each set
of trial transit parameters, the method uses one polynomial for a shared transit
shape and separate local polynomials for the background trends around individual
transits.

The search has three main stages:

1. **Segmentation and trend-model selection**
   - Divide the light curve at time gaps and, optionally, flux discontinuities.
   - Select a suitable background-polynomial order through Bayes-factor model
     comparison.
2. **Linear search**
   - Evaluate the log-likelihood difference on a two-dimensional grid of trial
     transit durations and mid-transit times.
3. **Period search and global fitting**
   - Combine compatible single-transit results over trial periods.
   - Globally refit the periodic model and baseline model.
   - Construct a periodogram from the log-likelihood difference and Signal
     Detection Efficiency (SDE).

This three-stage approximation retains a final global fit while reducing the
cost of an otherwise exhaustive search over period, duration, and epoch.

## Results reported in the paper

In the injection-recovery experiment described in the paper, SPLS was compared
with biweight-detrended BLS and TLS:

- For injected signals with periods of 10–480 days and SNR below 9, SPLS
  achieved a true-positive rate at least 22.6% higher than the comparison
  methods at the same 10% false-positive rate.
- At the ROC-derived threshold used in that experiment, SPLS recovered the most
  injected signals and produced the fewest false recoveries among the three
  tested methods.
- On the tested Kepler confirmed single-planet systems, SPLS reached a 97%
  recovery fraction at an SDE threshold of 9.89.

These values describe the datasets, preprocessing, parameter settings, and
decision rules used in the paper. In particular, SDE = 9.89 is not a universal
detection threshold. Calibrate a threshold for the survey, preprocessing
pipeline, search range, and desired false-positive rate in your own application.

## Installation

Clone the repository and install the package from its root directory:

```bash
git clone https://github.com/Lu-dandandan/SPLS.git
cd SPLS
python -m pip install .
```

## Quick start

The following example uses the bundled Kepler-572 light curve and mirrors the
workflow in `Example1_Kepler-572.ipynb`. Run it from the repository root.

```python
import numpy as np
import pandas as pd

from SPLS import SPLeastSquares


# The bundled CSV contains normalized time, flux, and flux-uncertainty columns.
light_curve = pd.read_csv("example_data/Kepler572.csv")
time = light_curve["t"].to_numpy(dtype=float)
flux = light_curve["f"].to_numpy(dtype=float)
flux_error = light_curve["df"].to_numpy(dtype=float)

# SPLS expects finite, equally sized arrays and strictly positive uncertainties.
valid = (
    np.isfinite(time)
    & np.isfinite(flux)
    & np.isfinite(flux_error)
    & (flux_error > 0)
)
search = SPLeastSquares(time[valid], flux[valid], flux_error[valid])

# Step 0: segment the light curve and select a background-trend order.
# Flux-gap segmentation is useful for suppressing discontinuity-driven false
# positives when searching for weak signals.
search.step0_segment(flux_gap=True)

maximum_duration = 0.9  # days
window_size = 1.8  # days; must be larger than maximum_duration
trend_order = search.step0_default_trend_order(
    dmax=maximum_duration,
    window=window_size,
)

# Step 1: construct the single-transit likelihood map.
# A fourth-order signal polynomial is recommended by the paper.
signal_order = 4
duration_samples = search.step1_pre_default_dsam1(
    sig_order=signal_order,
    dmax=maximum_duration,
    OS_d=10,
)
linear_result = search.step1_linear_search(
    trend_order=trend_order,
    sig_order=signal_order,
    d_sam=duration_samples,
    window=window_size,
    max_workers=1,
    Pmin_step1=3.0,
    OS_tm=3,
)
linear_result.plot()

# Step 2: generate the trial periods, perform the periodic search, and globally
# refit each trial. Use the same min_num_transit value in both calls.
minimum_transits = 2
period_samples = search.step2_pre_default_Psam(
    Pmin_step2=6.0,
    min_num_transit=minimum_transits,
)
periodogram_result = search.step2_periodogram(
    P_sam=period_samples,
    min_num_transit=minimum_transits,
    max_workers=1,
)

print(f"Best period:   {periodogram_result.P_best:.6f} days")
print(f"Best duration: {periodogram_result.d_best * 24:.4f} hours")
print(f"Best epoch:    {periodogram_result.tm0_best:.6f} days")
print(f"Peak SDE:      {periodogram_result.SDE_best:.3f}")
print(f"Depth SNR:     {periodogram_result.depth_snr:.3f}")

periodogram_result.plots()
```


## Input data

Create the search object with three one-dimensional arrays:

```python
from SPLS import SPLeastSquares

search = SPLeastSquares(time, flux, flux_error)
```

| Argument | Meaning | Practical requirements |
| --- | --- | --- |
| `time` | Observation times in days | One cadence within a search; no overlapping timestamps; gaps are allowed |
| `flux` | Photometric flux | Preferably normalized around unity |
| `flux_error` | Flux uncertainties | Same normalization as `flux`; finite and strictly positive |

The arrays must have equal lengths. The constructor sorts them by time. Before
creating the object, remove NaNs, infinities, invalid uncertainties.

## Search workflow and parameter guidance

### 1. Segment the data

```python
search.step0_segment(
    flux_gap=False,
    gap_time_threshold=4.5,
    gap_min_num_in_one_seg=10,
    gap_delta_flux_mad_threshold=5,
)
```

- `flux_gap=False` segments at time gaps only. This preserves more data and is a
  reasonable starting point for strong signals.
- `flux_gap=True` also splits at significant flux discontinuities. It can reduce
  false positives caused by discontinuities when targeting weak signals, but it
  may exclude additional data around the resulting boundaries.
- `gap_time_threshold` is measured in units of the data cadence.
- Segments with fewer than `gap_min_num_in_one_seg` points are discarded.

### 2. Select the trend order

```python
trend_order = search.step0_default_trend_order(
    dmax=maximum_duration,
    window=window_size,
    quantile_value=0.9,
    trend_order_max=3,
    OS_seg=12,
)
```

The method compares adjacent polynomial complexities with a Bayes-factor
criterion in sampled local segments, then uses the requested quantile of the
segment-level preferred orders. The paper uses the 90th percentile by default.
This is a pragmatic global choice, not a guarantee that every local trend is
described optimally.

### 3. Sample durations and run the linear search

```python
duration_samples = search.step1_pre_default_dsam1(
    sig_order=4,
    dmax=maximum_duration,
    dmin=None,
    OS_d=15,
)

linear_result = search.step1_linear_search(
    trend_order=trend_order,
    sig_order=4,
    d_sam=duration_samples,
    window=window_size,
    max_workers=1,
    Pmin_step1=10.0,
    OS_P=2,
    Rs=1.0,
    Ms=1.0,
    OS_tm=5,
    A_limit=True,
)
```

Important constraints and trade-offs:

- The implemented signal-polynomial orders are 2 and 4; order 4 is recommended
  for a more flexible transit shape.
- The default minimum duration is
  `(sig_order / 2 + 2) × minimum cadence`, ensuring enough in-transit points for
  a stable polynomial fit.
- `window` must be larger than the largest sampled duration.
- `Pmin_step1` must be at least as large as `window`. Smaller values are
  automatically raised to `window`.
- A smaller `Pmin_step1` creates a finer mid-transit-time grid and can greatly
  increase memory use and runtime, especially for long-baseline light curves.
- With default period sampling, set `Rs` and `Ms` in solar units when reliable
  stellar parameters are available. They influence the period-grid resolution.
- `A_limit=True` applies the transit-shape coefficient constraints described in
  the paper and is the recommended default.
- `max_workers > 1` parallelizes the linear search across light-curve segments.
  Start with `max_workers=1`, then increase it only after measuring memory use.

For a custom period grid, provide a compatible `dPmin` to
`step1_linear_search`; this value controls the mid-transit-time spacing used by
the folding-based period search.

### 4. Generate periods and build the periodogram

```python
minimum_transits = 3
period_samples = search.step2_pre_default_Psam(
    Pmin_step2=10.0,
    Pmax_step2=None,
    min_num_transit=minimum_transits,
)

periodogram_result = search.step2_periodogram(
    P_sam=period_samples,
    min_num_transit=minimum_transits,
    max_workers=1,
    d_limit=True,
)
```

- `Pmin_step2` must be no smaller than `Pmin_step1`.
- The default maximum period is
  `time_span / (min_num_transit - 1)`.
- Use the same `min_num_transit` value when generating periods and running the
  periodogram.
- `d_limit=True` restricts the duration range as a function of period using the
  physically motivated region described in the paper.
- Sequential execution (`max_workers=1`) is recommended for this stage. In the
  current implementation, multiprocessing can be slower because each worker has
  substantial memory and data-transfer overhead.

## Result objects

`step1_linear_search` returns a `Result_LinearSearch` object:

| Attribute or method | Description |
| --- | --- |
| `dlnL` | Two-dimensional log-likelihood-difference map |
| `tm_sam` | Sampled mid-transit times |
| `d_sam` | Sampled transit durations |
| `plot()` | Plot the one- and two-dimensional linear-search summaries |

`step2_periodogram` returns a `Result_Periodogram` object:

| Attribute or method | Description |
| --- | --- |
| `P_sam`, `d_sam` | Sampled periods and durations |
| `dlnL_arr`, `SDE_arr` | Periodogram log-likelihood differences and SDE values |
| `P_best` | Period at the highest SDE peak, in days |
| `d_best` | Best-fit transit duration, in days |
| `tm0_best` | Best-fit first mid-transit time, in the input time system |
| `SDE_best` | SDE of the highest periodogram peak |
| `depth_best`, `depth_sigma_best`, `depth_snr` | Fitted depth diagnostics |
| `number_segments_at_best_parameters` | Number of fitted transit windows |
| `plots()` | Plot the periodogram, fitted segments, and phase curves |

The plotted light curves contain only the windows used in the fit. One phase
plot shows the simultaneous transit-plus-trend model; the detrended phase plot
subtracts the fitted trend component for visualization.

## Examples

| Notebook | Purpose | Additional dependency or data |
| --- | --- | --- |
| [`Example1_Kepler-572.ipynb`](Example1_Kepler-572.ipynb) | End-to-end search of a partial Kepler long-cadence light curve | Downloads data with Lightkurve |
| [`Example2_KIC4458832.ipynb`](Example2_KIC4458832.ipynb) | Inject a synthetic transit and recover it with SPLS | Uses Astropy and `example_data/KIC4458832.csv` |
| [`Example3_segment_introduction.ipynb`](Example3_segment_introduction.ipynb) | Compare time-gap-only and time-plus-flux-gap segmentation | Uses `example_data/segment_example.csv` |


## Computational considerations

SPLS gains sensitivity by fitting the signal and background jointly, but the
approach is more expensive than a conventional detrend-then-search pipeline. In
the paper's injection-recovery setup, the reported single-core Apple M2 time per
target was approximately 193.2 minutes for SPLS, compared with 50.8 minutes for
TLS and 0.5 minutes for biweight+BLS. These are benchmark-specific values rather
than general runtime guarantees.

The main runtime drivers are:

- total light-curve time span;
- minimum searched period;
- number of sampled durations;
- mid-transit-time oversampling;
- period oversampling and search range;
- number and size of retained segments.

For initial tests, use a short light-curve interval or a restricted period range,
then expand the search after confirming the data preparation and parameter
choices.

## Scope and limitations

- SPLS is a detection method, not a complete validation pipeline. Candidate
  signals still require vetting for aliases, instrumental systematics, stellar
  variability, and astrophysical false positives.
- The local polynomial background model may be inadequate for some rapidly
  varying or highly structured light curves. Trend-order selection can affect
  both recovery and false-positive behavior.
- Data close to segmentation boundaries may not be used in a trial transit
  window, so some otherwise available transits can be excluded.
- The periodic model assumes a shared transit shape across epochs. Transit-timing
  variations, duration variations, or strong shape evolution can reduce
  sensitivity.
- The SDE distribution depends on the data and the searched parameter space.
  Estimate detection thresholds with non-injected controls or injection-recovery
  tests matched to the intended survey.

## Citation

If you use SPLS in scientific work, please cite the method paper.

```bibtex
@ARTICLE{2026AJ....171...64Z,
       author = {{Zheng}, Shuyue and {Feng}, Fabo and {Rui}, Yicheng},
        title = "{Segmented-Polynomial-fitting Least Squares (SPLS): An Optimized Algorithm to Find Earth Twins}",
      journal = {\aj},
     keywords = {Exoplanet detection methods, Transit photometry, Time series analysis, Period search, Astronomy data analysis, Light curves, 489, 1709, 1916, 1955, 1858, 918, Earth and Planetary Astrophysics, Instrumentation and Methods for Astrophysics},
         year = 2026,
        month = feb,
       volume = {171},
       number = {2},
          eid = {64},
        pages = {64},
          doi = {10.3847/1538-3881/ae2679},
archivePrefix = {arXiv},
       eprint = {2512.02356},
 primaryClass = {astro-ph.EP},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026AJ....171...64Z},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
