import os
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.stats import truncnorm, triang, norm, uniform
from sklearn.metrics import r2_score
import TableTerp.TableTerp as Terp
import atmo.atmo as atmo

# Toggle for reproducible/random results
use_fixed_seed = True  # Set to True for reproducibility, False for randomness
rng = np.random.default_rng(42) if use_fixed_seed else None

# Factor for consistent conversion
LBM_TO_KG = 0.45359237

# Flag to skip plotting distributions
MAKE_SETUP_PLOTS = True

# Amplify or mute incremental untrimmed DCD effect
DCD_FIXED = 0.0025

# Reference values of N1 model
N1C_VS_FFC_INTERCEPT = 70
N1C_VS_FFC_SLOPE = 0.003
N1C_VS_FFC_NOISE_STD = 0.05 # Noise term

SD_ENG_OFFSET_TAIL = 250
SD_ENG_OFFSET_POINT = 10

MACH_NOISE_STD = 0.0005
PRESS_ALT_FT_NOISE_STD = 5
AOA_NOISE_DEG_STD = 0.025
GW_LB_NOISE_STD = 20

FHV_NOM_BTUPERLB = 18580 # Expected nominal FHV
FHV_SD_BTUPERLB = 57.6 # Std dev for FHV

MIN_FUEL_RESERVE_LB = 3000   # Reasonable min quantity of fuel

# Define counts
n_tails = 35
n_days = 180
n_pts_per_day_per_tail_mean = 45 # 60 was observed other studies that averaged 1500 points per tail per month
n_pts_per_day_per_tail_std = 15
n_pts_per_day_per_tail_max = 90

payload_by_flt = np.empty((n_days, n_tails), dtype=object)

# Constants from airplane definition
s_ref_ft2 = 1340  # Wing reference area in square feet
b_span_ft = 112.58 # Wing span 
ar = b_span_ft**2 / s_ref_ft2
pi_ar = np.pi * ar
owe = 95500 # Operating weight empty
cl_zero_aoa = 0.15
k_lift_curve = 1

# Setup table interpolators
# Get list of table files in the target directory for the airplane/type
tabledata_dir = os.path.join(os.path.dirname(__file__), 'TableData')
file_list = glob.glob(os.path.join(tabledata_dir, '*.apdb'))

# Process the files and create the table interpolators required for the specific airplane
bound_err = False # If True, when interpolated values are requested outside of the domain of the input data, a ValueError is raised. If False, then fill_value is used. Default is True.
interpolators = Terp.process_files(file_list, bound_err)

# Model reference conditions
theta_ref = 0.751865348 # standard day at tropopause 
mach_ref = 0.78
sfc_nom_perhr = 0.6
n_nom = 0.5

# Set directory for collecting output plots
figdir = os.path.join(os.path.dirname(__file__), 'Runs')

# Define pressure altitude values and their observed frequencies,
# This will form a realistic sampling distribution
press_alts_ft = [32000, 33000, 34000, 35000, 36000, 37000, 38000, 39000, 40000, 41000]
frequencies   = [    0,   500,  1000,  6000, 13000, 25000, 21000,  7000,  4000, 1000 ]

# Operational effects
# Monthly operational weight variation (bias and std dev) to reflect seasonality
gross_wt_monthly_bias_lb = [50, 38, 13, 13, 25, 19, 19, 25, 25, 25, 28, 63]
gross_wt_monthly_std_lb =  [100, 100, 80, 70, 40, 50, 50, 50, 50, 70, 100, 110]

# Convert frequencies to probabilities
total_frequency = sum(frequencies)
press_alt_probabilities = [freq / total_frequency for freq in frequencies]

# Per-flight weak correlation (no altitude drift)
USE_PER_FLIGHT_WEAK_CORR = True  # flip to False to go i.i.d.

# AR(1) persistence for [Mach, ISA(C), Lat(deg), Track(deg)]
PHI_VEC_4 = np.array([0.95, 0.92, 0.92, 0.98])

# Per-step innovation stdevs (tiny wiggle per point)
STEP_SIGMAS_4 = np.array([
    0.00040,  # Mach
    0.20,     # ISA dev (C)
    0.015,    # Latitude (deg)
    0.25      # Track (deg)
])

# Weak cross-correlations among innovations
STEP_CORR_4 = np.array([
    [ 1.00, -0.10,  0.00,  0.00],  # Mach
    [-0.10,  1.00,  0.00,  0.00],  # ISA
    [ 0.00,  0.00,  1.00,  0.10],  # Lat
    [ 0.00,  0.00,  0.10,  1.00],  # Track
])

# Bounds for clipping (Mach, ISA, Lat, Track). Track wraps 0–360.
BOUNDS_4 = [
    (0.755, 0.790),   # Mach
    (-7.0,   17.0),    # ISA
    (8.0,    68.0),    # Lat
    (0.0,   360.0)     # Track
]
WRAP_IDX_4 = {3}  # index of Track in the 4-vector below

def _reflect(x, lo, hi):
    r = hi - lo
    y = (x - lo) % (2*r)
    y = np.where(y > r, 2*r - y, y)
    return lo + y

# Simple widening to hit target 95% widths for Latitude and ISA
TARGET_95_WIDTH_LAT = 10.0  # degrees
TARGET_95_WIDTH_ISA = 3.0   # deg C

def _sigma_step_for_target_width(width, phi):
    sigma_stat = width / 3.92          # target SD from desired 95% width
    return sigma_stat * np.sqrt(1.0 - phi**2)

# indices: [Mach, ISA, Lat, Track] = [0, 1, 2, 3]
STEP_SIGMAS_4[2] = _sigma_step_for_target_width(TARGET_95_WIDTH_LAT, PHI_VEC_4[2])  # Latitude
STEP_SIGMAS_4[1] = _sigma_step_for_target_width(TARGET_95_WIDTH_ISA, PHI_VEC_4[1])  # ISA

# ==== Helpers ====
def _safe_cholesky_from_corr(sigmas, corr):
    Sigma = np.outer(sigmas, sigmas) * corr
    for k in range(6):
        try:
            return np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError:
            Sigma += np.eye(Sigma.shape[0]) * (10.0 ** (-(8 - k)))
    raise RuntimeError("Covariance not SPD even after jitter.")

def _mv_ar1_flight_4(n, mu4, phi_vec, step_sigmas, step_corr, bounds, wrap_idx=None, rng=None):
    """AR(1) around baseline mu4 for [Mach, ISA, Lat, Track]."""
    L = _safe_cholesky_from_corr(step_sigmas, step_corr)
    x = np.zeros((n, 4))
    x[0] = mu4
    for t in range(1, n):
        z = rng.normal(size=4) if rng is not None else np.random.normal(size=4)
        innov = L @ z
        x[t] = mu4 + phi_vec * (x[t-1] - mu4) + innov
        # clip/wrap
        for j, (lo, hi) in enumerate(bounds):
            if wrap_idx and j in wrap_idx:
                period = hi - lo
                x[t, j] = (x[t, j] - lo) % period + lo
            else:
                x[t, j] = np.clip(x[t, j], lo, hi)
    return x

def _nearest_alt_stat(alt_ft, alts=(32000,33000,34000,35000,36000,37000,38000,39000,40000,41000), stats=None):
    if stats is None: return None
    nearest = min(alts, key=lambda a: abs(a - alt_ft))
    return stats[nearest]['mean'], stats[nearest]['std_dev']

def _sample_gw_from_alt_vector(alts, stats, rng):
    out = []
    for a in alts:
        mu, sd = _nearest_alt_stat(a, stats=stats)
        out.append(truncated_normal(mu, sd, 100000, 162000, 1, random_state=rng)[0])
    return out

# The gross weight and pressure altitude are strongly related
# The following table defines a realistic mapping of pressure altitude to mean and standard deviation for gross weight
gross_wt_lb_stats = {
    32000: {'mean': 139389.1, 'std_dev': 9148.1},
    33000: {'mean': 145470.4, 'std_dev': 7501.4},
    34000: {'mean': 144617.0, 'std_dev': 7472.3},
    35000: {'mean': 150169.4, 'std_dev': 7057.6},
    36000: {'mean': 146880.1, 'std_dev': 5076.6},
    37000: {'mean': 143929.0, 'std_dev': 4517.5},
    38000: {'mean': 139299.2, 'std_dev': 4520.7},
    39000: {'mean': 134395.2, 'std_dev': 5283.8},
    40000: {'mean': 126094.4, 'std_dev': 6158.5},
    41000: {'mean': 121100.0, 'std_dev': 6176.6}
}

# Function to select month based on day
def get_month(day):
    return (day // 30) % 12  # Assuming each month has 30 days for simplicity

# Save all open figures
def save_all_figures(output_dir=figdir, file_format='svg'):
    """
    Saves all open matplotlib figures to the specified directory with unique filenames.
    Parameters:
        output_dir (str): The directory where the figures will be saved.
        file_format (str): The file format for the saved figures (e.g., 'png', 'pdf').
    Example usage:
    save_all_figures(figdir)
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all figure numbers
    figures = [plt.figure(i) for i in plt.get_fignums()]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for i, fig in enumerate(figures):
        filename = os.path.join(output_dir, f"figure_{timestamp}_{i+1}.{file_format}")
        fig.savefig(filename)
        print(f"Saved: {filename}")

def plot_generated_data(values, title="Distribution of Values", xlabel="Value"):
    """
    Plot a histogram for a set of pre-generated values.
    Parameters:
    - values: array-like, the data values to plot
    - title: str, title for the plot
    - xlabel: str, label for the x-axis
    - bins: int, number of bins for the histogram
    """    
    edge_width=1.0
    title_fontsize=18 
    label_fontsize=14 
    tick_fontsize=14
    border_width=1.0
    bins=10
    ylabel = "Frequency"
    fontweight='bold'

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, edgecolor='black', color='darkcyan', linewidth=1)
    plt.title(title, fontsize=title_fontsize, fontweight=fontweight)
    plt.xlabel(xlabel, fontsize=label_fontsize, fontweight=fontweight)
    plt.ylabel(ylabel, fontsize=label_fontsize, fontweight=fontweight)
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    # Set tick parameters including font size and weight
    plt.tick_params(axis='both', which='major', labelsize=tick_fontsize, width=edge_width)
    # Set border (spine) widths
    for spine in plt.gca().spines.values():
        spine.set_linewidth(border_width)
    save_all_figures(figdir)
    plt.show()

def plot_histogram_by_tail_all_days(df, parameter, bin_width=0.05, display_option="both", legend_fs=12, tick_fs=12, label_fs=14):
    """
    Plot histograms and/or KDEs for a specified parameter by tail over all days.
    Parameters:
    - df (DataFrame): The data containing the parameters, including 'Tail' and the specified parameter.
    - parameter (str): The name of the parameter/column to plot.
    - bin_width (float): The width of the bins for the histogram.
    - display_option (str): Options for display - "both", "histogram", "kde".
    """
    tails = df["Tail"].unique()
    fig, ax = plt.subplots(figsize=(9, 6))

    for tail in tails:
        data = df[df["Tail"] == tail][parameter]
        range_min, range_max = data.min(), data.max()
        num_bins = max(1, int(np.ceil((range_max - range_min) / bin_width)))

        if display_option in ["both", "histogram"]:
            ax.hist(data, bins=num_bins, density=True, alpha=0.4, edgecolor="black",
                    linewidth=1.2, label=tail)

        if display_option in ["both", "kde"]:
            sns.kdeplot(data, fill=True, linewidth=1.5, label=f"{tail}", ax=ax)

    ax.set_xlabel(parameter, fontsize=label_fs, fontweight="bold")
    ax.set_ylabel("Density", fontsize=label_fs, fontweight="bold")

    # ⬇️ This is the key part
    ax.tick_params(axis="both", which="both", labelsize=tick_fs)  # tick label size
    leg = ax.legend(title="Tail", loc="upper right", ncol=2,
                    fontsize=legend_fs,        # legend entries
                    title_fontsize=legend_fs)  # legend title
    # Optional: bold legend title
    leg.get_title().set_fontweight("bold")

    fig.tight_layout()
    save_all_figures(figdir)
    plt.show()

def plot_bar_chart_by_tail(df, parameter, y_min=None, y_max=None):
    """
    Plot a bar chart for the specified parameter by tail.
    Parameters:
    - df (DataFrame): The data containing the parameters, including 'Tail' and the specified parameter.
    - parameter (str): The name of the parameter/column to plot.
    - y_min (float, optional): The minimum value for the y-axis. If None, the y-axis will auto-scale.
    """
    # Calculate mean value of the parameter for each tail
    means_by_tail = df.groupby("Tail")[parameter].mean()
    
    plt.figure(figsize=(12, 8))
    means_by_tail.plot(kind="bar", color="darkcyan", edgecolor="black")

    # Set y-axis minimum value if provided
    if y_min is not None:
        plt.ylim(bottom=y_min)

    # Set y-axis maximum value if provided
    if y_max is not None:
        plt.ylim(top=y_max)
    
    # Set labels and title
    plt.title(f"Average {parameter} by Tail", fontsize=16, fontweight="bold")
    plt.xlabel("Tail", fontsize=14, fontweight="bold")
    plt.ylabel(f"Average {parameter}", fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    save_all_figures(figdir)
    plt.show()

def plot_bar_chart_from_array(values, tails, title="Parameter by Tail", xlabel="Tail", ylabel="Value", y_min=None, y_max=None):
    """
    Plot a bar chart for an array of parameter values by tail.    
    Parameters:
    - values (array-like): Array of parameter values corresponding to each tail.
    - tails (array-like): Array of tail names.
    - title (str): Title for the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    - y_min (float, optional): Minimum value for the y-axis. If None, the y-axis will auto-scale.
    - y_max (float, optional): Maximum value for the y-axis. If None, the y-axis will auto-scale.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(tails, values, color="darkcyan", edgecolor="black")
    
    # Set y-axis limits if provided
    if y_min is not None:
        plt.ylim(bottom=y_min)
    if y_max is not None:
        plt.ylim(top=y_max)

    # Set labels and title
    plt.title(title, fontsize=16, fontweight="bold")
    plt.xlabel(xlabel, fontsize=14, fontweight="bold")
    plt.ylabel(ylabel, fontsize=14, fontweight="bold")
    
    plt.tight_layout()
    save_all_figures(figdir)
    plt.show()

def plot_generated_dist_and_bar(val_array, tail_array, axis_label):
    if not MAKE_SETUP_PLOTS:
        return
    plot_generated_data(val_array, title="", xlabel=axis_label)
    plot_bar_chart_from_array(val_array, tail_array, title="", xlabel="Tail", ylabel=axis_label)

def plot_scatter_by_tail(df, x_param, y_param, xlim=None, ylim=None, grid=True):
    """
    Plot a scatter plot of one parameter versus another, subset by tail.    
    Parameters:
    - df (DataFrame): The data containing parameters by tail and day.
    - x_param (str): The name of the parameter for the x-axis.
    - y_param (str): The name of the parameter for the y-axis.
    - xlim (tuple): x-axis limits as (xmin, xmax).
    - ylim (tuple): y-axis limits as (ymin, ymax).
    - grid (bool): Whether to show major gridlines (default is True).
    """
    unique_tails = df['Tail'].unique()
    markers = ['o', 's', 'D', 'v', '^', '<', '>', 'p', '*', 'h']  # List of marker shapes

    plt.figure(figsize=(9, 6))
    #plt.title(f"Scatter Plot of {y_param} vs {x_param} by Tail", fontsize=16, fontweight='bold')
    plt.xlabel(x_param, fontsize=14, fontweight='bold')
    plt.ylabel(y_param, fontsize=14, fontweight='bold')

    for i, tail in enumerate(unique_tails):
        tail_data = df[df['Tail'] == tail]
        marker_shape = markers[i % len(markers)]  # Cycle through marker shapes
        plt.scatter(tail_data[x_param], tail_data[y_param], label=f'Tail {tail}', 
                    s=14,  # Marker size
                    alpha=0.7, marker=marker_shape)

    # Apply x and y limits if provided
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    # Show gridlines if requested
    if grid:
        plt.grid(visible=True, which='major', linestyle='-', linewidth=0.5)

    plt.legend(title="Tail", fontsize=12, title_fontsize=12, ncol=2)
    save_all_figures(figdir)
    plt.show()

# Define truncated normal helper function with random state
def truncated_normal(mean, std, lower, upper, size, random_state=None):
    """
    Generate samples from a truncated normal distribution.
    Parameters:
    - mean: Mean of the distribution
    - std: Standard deviation of the distribution
    - lower: Lower bound for truncation
    - upper: Upper bound for truncation
    - size: Number of samples to generate
    - random_state: An optional random generator (if None, will be non-deterministic)
    """
    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm(a, b, loc=mean, scale=std).rvs(size=size, random_state=random_state)

def gravitational_acceleration(latitude, altitude, true_speed, true_track_angle):
    # Constants
    gcoeff = 32.1724418256819 # Coefficient in Lambert's equation. Note: This is NOT the standard value g. This would not yield the standard value at the associated latitude (45.5425 deg)
    ωe = 7.292115e-5  # Earth's rotation rate in radians per second
    re = 20855531.496063 # From ICAO 7488, r=6356766 m, converted using 0.3048 m/ft, nominal radius earth, feet

    # Convert degrees to radians for trigonometric functions
    latitude_rad = np.radians(latitude)
    true_track_angle_rad = np.radians(true_track_angle)

    # Calculate gravitational acceleration at sea level for given latitudes
    cos2phi = np.cos(2 * latitude_rad)
    g_phi_SL = gcoeff * (1 - 2.6373e-3 * cos2phi + 5.9e-6 * cos2phi**2)

    # Calculate gravitational acceleration at given altitudes
    cos_phi = np.cos(latitude_rad)
    g_phi_z = (g_phi_SL + ωe**2 * re * cos_phi**2) * (re / (re + altitude))**2 - ωe**2 * (re + altitude) * cos_phi**2

    # Calculate centrifugal and Coriolis corrections
    sin_phi = np.sin(latitude_rad)
    sin_chi = np.sin(true_track_angle_rad)
    delta_g_centrifugal = -(true_speed**2 / (re + altitude) + 2 * ωe * true_speed * cos_phi * sin_chi)

    # Total gravitational acceleration
    g_total = g_phi_z + delta_g_centrifugal

    return g_total
    
# Sample Tails for Study
tails = [i for i in range(1, n_tails + 1)]

# Time of Study
n_pts_per_day_per_tail = truncated_normal(n_pts_per_day_per_tail_mean, n_pts_per_day_per_tail_std, 0, n_pts_per_day_per_tail_max, (n_days, n_tails), random_state=rng).astype(int)

# Operating conditions
# Initialize empty lists to store daily data
mach = np.empty((n_days, n_tails), dtype=object)
gross_wt_lb = np.empty((n_days, n_tails), dtype=object)
isa_dev_degc = np.empty((n_days, n_tails), dtype=object)
press_alt_ft = np.empty((n_days, n_tails), dtype=object)
track_true_deg = np.empty((n_days, n_tails), dtype=object)
latitude_deg = np.empty((n_days, n_tails), dtype=object)

# Generate number‑of‑points, flight structure, and per‑flight latent factors
flight_ids      = np.empty((n_days, n_tails), dtype=object)
n_points_by_flt = np.empty((n_days, n_tails), dtype=object)
fhv_by_flt      = np.empty((n_days, n_tails), dtype=object)
mass_bias_by_flt= np.empty((n_days, n_tails), dtype=object)

for day in range(n_days):
    month = get_month(day)
    for tail in range(n_tails):
        # Decide total points and the flight partition
        n_points_day = n_pts_per_day_per_tail[day, tail]
        n_flights = rng.integers(1, 6) # 1–5 flights
        sizes = rng.multinomial(n_points_day, np.ones(n_flights) / n_flights)
        flight_ids[day, tail] = np.repeat(np.arange(n_flights), sizes)
        n_points_by_flt[day, tail] = sizes

        # Per‑flight latent factors
        fhv_by_flt[day, tail] = truncated_normal(FHV_NOM_BTUPERLB, FHV_SD_BTUPERLB, FHV_NOM_BTUPERLB-3*FHV_SD_BTUPERLB, FHV_NOM_BTUPERLB+3*FHV_SD_BTUPERLB, n_flights, random_state=rng)
        mass_bias_by_flt[day, tail] = truncated_normal(gross_wt_monthly_bias_lb[month], gross_wt_monthly_std_lb[month],gross_wt_monthly_bias_lb[month]-3*gross_wt_monthly_std_lb[month],
                                                        gross_wt_monthly_bias_lb[month]+3*gross_wt_monthly_std_lb[month], n_flights, random_state=rng)

        # Per-flight operating conditions (altitude constant per flight)
        mach_day_list, isa_day_list, lat_day_list, track_day_list, alt_day_list = [], [], [], [], []

        for flt_idx, m_f in enumerate(n_points_by_flt[day, tail]):
            if m_f == 0:
                continue

            # Choose ONE flight level for the whole flight (discrete, realistic)
            alt0 = rng.choice(press_alts_ft, p=press_alt_probabilities)
            alt_series = np.full(m_f, alt0, dtype=float) + triang.rvs(0.5, loc=-25, scale=50, size=m_f, random_state=rng)

            # Baselines for this flight (μ4) for [Mach, ISA, Lat, Track]
            mach0 = truncated_normal(0.773, 0.004, 0.755, 0.790, 1, random_state=rng)[0]
            # When drawing baselines for a flight, keep a buffer inside those bounds
            isa0 = truncated_normal(2.0, 2.5, BOUNDS_4[1][0]+1.0, BOUNDS_4[1][1]-1.0, 1, random_state=rng)[0]
            lat0 = truncated_normal(32.5, 6.0, BOUNDS_4[2][0]+2.0, BOUNDS_4[2][1]-2.0, 1, random_state=rng)[0]
            trk0  = rng.uniform(0.0, 360.0)
            mu4   = np.array([mach0, isa0, lat0, trk0])

            if USE_PER_FLIGHT_WEAK_CORR:
                series4 = _mv_ar1_flight_4(n=m_f,mu4=mu4,phi_vec=PHI_VEC_4,step_sigmas=STEP_SIGMAS_4,step_corr=STEP_CORR_4,bounds=BOUNDS_4,wrap_idx=WRAP_IDX_4,rng=rng)
            else:
                series4 = np.column_stack([
                    truncated_normal(0.773, 0.011, 0.755, 0.790, m_f, random_state=rng),  # Mach
                    truncated_normal(2.0,   5.0,  -5.0,  15.0,  m_f, random_state=rng),   # ISA
                    truncated_normal(32.5, 15.0,  10.0,  66.0,  m_f, random_state=rng),   # Lat
                    rng.uniform(0.0, 360.0, m_f)                                         # Track
                ])

            # Optional low-frequency drift per flight (adds span, but not fixed size)
            lat_A = rng.normal(0.0, 3.0)   # typical total drift amplitude (deg). 95% ~ ±6°
            isa_A = rng.normal(0.0, 1.0)   # typical total drift amplitude (°C). 95% ~ ±2°C

            if m_f > 1:
                t = np.linspace(-0.5, 0.5, m_f)  # centered ramp
                series4[:, 2] += lat_A * t       # latitude drift
                series4[:, 1] += isa_A * t       # ISA drift

            # Re-clip to bounds after the drift
            series4[:, 2] = _reflect(series4[:, 2], *BOUNDS_4[2])  # Lat
            series4[:, 1] = _reflect(series4[:, 1], *BOUNDS_4[1])  # ISA

            mach_day_list.append(series4[:, 0])
            isa_day_list.append(series4[:, 1])
            lat_day_list.append(series4[:, 2])
            track_day_list.append(series4[:, 3])
            alt_day_list.append(alt_series)

        # Stitch back to per-day arrays
        mach[day, tail]          = np.concatenate(mach_day_list)   if mach_day_list else np.array([])
        isa_dev_degc[day, tail]  = np.concatenate(isa_day_list)    if isa_day_list else np.array([])
        latitude_deg[day, tail]  = np.concatenate(lat_day_list)    if lat_day_list else np.array([])
        track_true_deg[day, tail]= np.concatenate(track_day_list)  if track_day_list else np.array([])
        press_alt_ft[day, tail]  = np.concatenate(alt_day_list)    if alt_day_list else np.array([])

        # Use existing GW-vs-altitude mapping
        gross_wt_lb[day, tail] = _sample_gw_from_alt_vector(press_alt_ft[day, tail], gross_wt_lb_stats, rng)

    # Enforce 50/50 north/south by day; GW monotone with latitude
    # Collect all flights for this day across all tails
    flight_refs = []
    for tail in range(n_tails):
        sizes = n_points_by_flt[day, tail]
        if sizes is None:
            continue
        for flt_idx, m_f in enumerate(sizes):
            if m_f is None or m_f <= 1:
                continue
            flight_refs.append((tail, flt_idx))

    if flight_refs:
        # Shuffle flight list deterministically if rng is fixed
        if rng is not None:
            order = rng.permutation(len(flight_refs))
        else:
            order = np.random.permutation(len(flight_refs))
        shuffled = [flight_refs[i] for i in order]

        # Half become northbound (latitude increasing); the rest southbound (latitude decreasing)
        half = len(shuffled) // 2
        inc_set = set(shuffled[:half])  # (tail, flt_idx) pairs that will be increasing

        # Apply monotone latitude and monotone-with-latitude gross weight to each flight
        for (tail, flt_idx) in shuffled:
            fid = flight_ids[day, tail]
            idx = np.where(fid == flt_idx)[0]
            if idx.size <= 1:
                continue

            # Work on numpy views
            lat_arr = np.asarray(latitude_deg[day, tail], dtype=float)
            gw_arr  = np.asarray(gross_wt_lb[day, tail], dtype=float)

            # Monotone latitude: increasing for "northbound", decreasing otherwise
            lat_sorted = np.sort(lat_arr[idx])
            if (tail, flt_idx) in inc_set:
                lat_arr[idx] = lat_sorted           # northbound
            else:
                lat_arr[idx] = lat_sorted[::-1]     # southbound

            # Monotone gross weight *with* latitude:
            # always sort GW descending in point order (decreasing along flight).
            # This yields a negative slope vs latitude for northbound flights
            # and a positive slope vs latitude for southbound flights (both monotone).
            gw_sorted = np.sort(gw_arr[idx])
            gw_arr[idx] = gw_sorted[::-1]

            # Write back
            latitude_deg[day, tail] = lat_arr
            gross_wt_lb[day, tail]  = gw_arr.tolist()

        # Choose one payload per flight (feasible w.r.t. min GW)
        for tail in range(n_tails):
            sizes = n_points_by_flt[day, tail]
            if sizes is None:
                payload_by_flt[day, tail] = np.array([], dtype=float)
                continue

            fid = flight_ids[day, tail]
            gw_day = np.asarray(gross_wt_lb[day, tail], dtype=float)

            # payload_list has one value per flight for this tail
            payload_list = np.zeros_like(sizes, dtype=float)

            for flt_idx, m_f in enumerate(sizes):
                if m_f <= 0 or fid is None:
                    payload_list[flt_idx] = 0.0
                    continue

                idx = np.where(fid == flt_idx)[0]
                if idx.size == 0:
                    payload_list[flt_idx] = 0.0
                    continue

                # Ffeasible max payload at the lightest point in this flight
                min_gw = np.min(gw_day[idx])
                max_payload = max(0.0, min_gw - owe - MIN_FUEL_RESERVE_LB)

                base_frac = rng.uniform(0.25, 0.65)  # or any distribution you prefer
                payload_list[flt_idx] = base_frac * max_payload

            payload_by_flt[day, tail] = payload_list

# Aircraft deterioration effects setup
# NOTE on triangular distributions: first number is shape param, which must be between 0 and 1; loc = left; right = loc + scale
empty_wt_bias_lb = triang.rvs(0.5, loc=-250, scale=500, size=n_tails, random_state=rng)
plot_generated_dist_and_bar(empty_wt_bias_lb, tails, "Fixed Empty Weight Bias (lb)")
plot_generated_dist_and_bar(empty_wt_bias_lb*LBM_TO_KG, tails, "Fixed Empty Weight Error (kg)")

delta_cd_aircraft = triang.rvs(0.5, loc=0.0000, scale=DCD_FIXED, size=n_tails, random_state=rng)
plot_generated_dist_and_bar(delta_cd_aircraft, tails, "Fixed Aircraft \u0394${C_D}$")

thrust_split_left_right = uniform.rvs(loc=-0.25, scale=0.5, size=n_tails, random_state=rng)
plot_generated_dist_and_bar(thrust_split_left_right, tails, "Fixed Thrust Split, Left vs Right, % above 50% left")

engine_offset_1 = truncated_normal(0, SD_ENG_OFFSET_TAIL, -3*SD_ENG_OFFSET_TAIL, 3*SD_ENG_OFFSET_TAIL, size=n_tails, random_state=rng)
plot_generated_dist_and_bar(engine_offset_1, tails, "Fixed Engine 1 WFOC Offset")
plot_generated_dist_and_bar(engine_offset_1*LBM_TO_KG, tails, "Fixed Engine #1 Fuel Flow Offset (kg/hr)")

engine_offset_2 = truncated_normal(0, SD_ENG_OFFSET_TAIL, -3*SD_ENG_OFFSET_TAIL, 3*SD_ENG_OFFSET_TAIL, size=n_tails, random_state=rng)
plot_generated_dist_and_bar(engine_offset_2, tails, "Fixed Engine 2 WFOC Offset")
plot_generated_dist_and_bar(engine_offset_2*LBM_TO_KG, tails, "Fixed Engine #2 Fuel Flow Offset (kg/hr)")

press_alt_bias_ft = truncated_normal(0, 5, -20, 20, size=n_tails, random_state=rng)
plot_generated_dist_and_bar(press_alt_bias_ft, tails, "Fixed Altitude Bias - feet")

mach_bias = truncated_normal(0, 0.00005, -0.0002, 0.0002, size=n_tails, random_state=rng)
plot_generated_dist_and_bar(mach_bias, tails, "Fixed Mach Bias")

aoa_bias_deg = truncated_normal(0, 0.025, -0.1, 0.1, size=n_tails, random_state=rng)
plot_generated_dist_and_bar(aoa_bias_deg, tails, "Fixed AoA Bias - deg")

# Create Tail DataFrame
df_tail = pd.DataFrame({
    "empty_wt_bias_lb": empty_wt_bias_lb,
    "delta_cd_aircraft": delta_cd_aircraft,
    "thrust_split_left_right": thrust_split_left_right,
    "engine_offset_1": engine_offset_1,
    "engine_offset_2": engine_offset_2,
    "press_alt_bias_ft": press_alt_bias_ft,
    "mach_bias": mach_bias,
    "aoa_bias_deg":aoa_bias_deg
})

# Print the DataFrame as a formatted table
print(df_tail.to_string(index=True))

# Derived quantities function
def compute_derived_quantities_with_bias(tail_idx, day, mach_array, gross_wt_lb_array, isa_dev_degc_array, press_alt_ft_array, track_true_deg_array, latitude_deg_array, flight_id_array, fhv_array, mass_bias_array, payload_array):
    results = []
    month = get_month(day)
    for i, (mach, gross_wt_lb, isa_dev_degc, press_alt_ft, track_true_deg, latitude_deg, flt_id) in enumerate(zip(mach_array, gross_wt_lb_array, isa_dev_degc_array, press_alt_ft_array, track_true_deg_array, latitude_deg_array, flight_id_array)):
        # Differentiate between true and indicated Mach and altitude
        mach_indicated = mach
        mach_true = mach_indicated + mach_bias[tail_idx] + rng.normal(0, MACH_NOISE_STD)
        press_alt_indicated_ft = press_alt_ft
        press_alt_true_ft = press_alt_indicated_ft + press_alt_bias_ft[tail_idx] + rng.normal(0, PRESS_ALT_FT_NOISE_STD)
            
        # Fuel Heating Value (FHV) for each cruise point
        fhv_btuperlb = fhv_array[flt_id]

        # Rate of climb (ft/min) and rate of acceleration (knots/min) for each cruise point
        roc_ftpermin = norm.rvs(0, 10, random_state=rng)
        roc_ftpersec = roc_ftpermin / 60 # Convert units to ft/sec
        accel_ktpermin = norm.rvs(0, 0.5, random_state=rng)
        accel_ft_per_sec2 = accel_ktpermin * 0.028130164 # Convert units to ft/sec^2

        # Calculate derived quantities with bias adjustments
        t_std_k = atmo.OatStdDay_Kelvin_fHp(press_alt_true_ft)
        stat_air_temp_k = isa_dev_degc + t_std_k
        theta = stat_air_temp_k / 288.15
        theta_total = theta * (1 + 0.2 * mach_true ** 2)
        ktas = mach_true * (4325.735929 * stat_air_temp_k) ** 0.5 * 0.592483801
        v_ftpersec = atmo.ConvKtsToFtPerSec(ktas)
        delta = atmo.Delta_fHp(press_alt_true_ft)
        delta_total = delta * (1 + 0.2 * mach_true ** 2) ** 3.5
        q_psf = 1481.351637 * delta * mach_true ** 2

        # Compute effective gravitational acceleration
        g_effective_ftpers2 = gravitational_acceleration(latitude_deg, press_alt_true_ft, v_ftpersec, track_true_deg)

        # Operational weight bias
        gross_wt_true_lb = gross_wt_lb

        gw_operational_bias_lb = mass_bias_array[flt_id] + rng.normal(0, GW_LB_NOISE_STD)
        gross_wt_indicated_lb  = (gross_wt_true_lb + empty_wt_bias_lb[tail_idx] + gw_operational_bias_lb)

        gross_wt_effective_lb = gross_wt_true_lb * g_effective_ftpers2 / atmo.constGo
        w_over_delta_effective_lb = gross_wt_effective_lb / delta

        # Fixed payload for this flight
        payload_wt_lb = payload_array[flt_id]

        # Fuel at this point is whatever is left
        gross_wt_true_less_owe_lb = gross_wt_true_lb - owe
        fuel_wt_lb = gross_wt_true_less_owe_lb - payload_wt_lb
        if fuel_wt_lb < 0:
            # This shouldn't happen because of the feasibility clip, but guard anyway
            fuel_wt_lb = 0.0

        cl = gross_wt_effective_lb / (q_psf * s_ref_ft2)

        lift_curve_slope_perdeg = 2 * pi_ar / (2 + (ar**2*(1-mach_true**2)/k_lift_curve**2 + 4)**0.5) * (np.pi / 180)
        aoa_true_deg = (cl - cl_zero_aoa) / lift_curve_slope_perdeg
        aoa_indicated_deg = aoa_true_deg + aoa_bias_deg[tail_idx] + rng.normal(0, AOA_NOISE_DEG_STD)
        
        # Interpolate real drag polars
        cd_polar = Terp.interpolate_values(interpolators, '737polar', mach_true, cl)
        dcdre = Terp.interpolate_values(interpolators, '737dcdre', press_alt_true_ft, mach_true)
        cd = cd_polar + dcdre + delta_cd_aircraft[tail_idx]
        l_over_d = cl / cd
        mach_l_over_d = mach_true * l_over_d

        # Excess thrust due to climb
        fex_climb_lb = gross_wt_effective_lb * roc_ftpersec * stat_air_temp_k / t_std_k / v_ftpersec

        # Excess thrust due to acceleration
        fex_accel_lb = gross_wt_effective_lb * accel_ft_per_sec2 / atmo.constGo

        # Total excess thrust and adjusted net thrust
        fex_total_lb = fex_climb_lb + fex_accel_lb
        fnet_total_lb = (cd * q_psf * s_ref_ft2) + fex_total_lb

        fnet_eng1_lb = fnet_total_lb / 2 * (1 + thrust_split_left_right[tail_idx] / 100) 
        fnet_eng2_lb = fnet_total_lb / 2 * (1 - thrust_split_left_right[tail_idx] / 100)

        fnet_eng1_cor_lb = fnet_eng1_lb / delta
        fnet_eng2_cor_lb = fnet_eng2_lb / delta

        fnet_total_cor_lb = fnet_eng1_cor_lb + fnet_eng2_cor_lb

        # Establish fuel flow adjustment based on FHV
        fhv_ratio = FHV_NOM_BTUPERLB / fhv_btuperlb

        # Baseline SFC
        sfc_base_perhr = sfc_nom_perhr * (theta / theta_ref) ** 0.5 * (mach_true / mach_ref) ** n_nom

        wfdot_base_eng1_lbperhr = sfc_base_perhr * fnet_eng1_lb
        wfdot_base_eng2_lbperhr = sfc_base_perhr * fnet_eng2_lb
        
        eng1_offset_point = rng.normal(0, SD_ENG_OFFSET_POINT)
        eng2_offset_point = rng.normal(0, SD_ENG_OFFSET_POINT)

        kwfx1 = (engine_offset_1[tail_idx]+eng1_offset_point)*delta_total*theta_total**0.5 / wfdot_base_eng1_lbperhr + 1
        kwfx2 = (engine_offset_2[tail_idx]+eng2_offset_point)*delta_total*theta_total**0.5 / wfdot_base_eng2_lbperhr + 1

        wfdot_eng1_lbperhr = wfdot_base_eng1_lbperhr * kwfx1 * fhv_ratio
        wfdot_eng2_lbperhr = wfdot_base_eng2_lbperhr * kwfx2 * fhv_ratio
        wfdot_total_lbperhr = wfdot_eng1_lbperhr + wfdot_eng2_lbperhr

        wfdot_cor_eng1_lbperhr = wfdot_eng1_lbperhr / (delta_total * theta_total**0.5)
        wfdot_cor_eng2_lbperhr = wfdot_eng2_lbperhr / (delta_total * theta_total**0.5)
        wfdot_cor_total_lbperhr = wfdot_cor_eng1_lbperhr + wfdot_cor_eng2_lbperhr

        wfdot_base_cor_eng1_lbperhr = wfdot_base_eng1_lbperhr / (delta_total * theta_total**0.5)
        wfdot_base_cor_eng2_lbperhr = wfdot_base_eng2_lbperhr / (delta_total * theta_total**0.5)

        # Engine health indicators
        n1c_eng1_pct = N1C_VS_FFC_SLOPE * wfdot_base_cor_eng1_lbperhr + N1C_VS_FFC_INTERCEPT + rng.normal(loc=0, scale=N1C_VS_FFC_NOISE_STD)
        n1c_eng2_pct = N1C_VS_FFC_SLOPE * wfdot_base_cor_eng2_lbperhr + N1C_VS_FFC_INTERCEPT + rng.normal(loc=0, scale=N1C_VS_FFC_NOISE_STD)
        
        n1_eng1_pct = n1c_eng1_pct * theta_total**0.5
        n1_eng2_pct = n1c_eng2_pct * theta_total**0.5

        # Specific range
        sr_nmperlb = ktas / wfdot_total_lbperhr

        # Add each computed value to the result dictionary for easy storage
        result = {
            "Month": month,
            "Flight ID": int(flt_id),
            "Mach Indicated": mach_indicated,
            "Mach True": mach_true,
            "Pressure Altitude Indicated (ft)": press_alt_indicated_ft,
            "Pressure Altitude True (ft)": press_alt_true_ft,
            "Gross Weight Indicated (lb)": gross_wt_indicated_lb,
            "Gross Weight Indicated (kg)": gross_wt_indicated_lb * LBM_TO_KG,
            "Gross Weight True (lb)": gross_wt_true_lb,
            "Gross Weight Effective (lb)": gross_wt_effective_lb,
            "W/delta Effective": w_over_delta_effective_lb,
            "Payload Weight (lb)": payload_wt_lb,
            "Fuel Weight (lb)": fuel_wt_lb,
            "AoA True (deg)": aoa_true_deg,
            "AoA Indicated (deg)": aoa_indicated_deg,
            "True Track (deg)": track_true_deg,
            "Latitude (deg)": latitude_deg,
            "OAT (K)": stat_air_temp_k,
            "T Standard Day (K)": t_std_k,
            "ISA Deviation (deg C)": isa_dev_degc,
            "theta": theta,
            "theta total": theta_total,
            "delta": delta,
            "delta total": delta_total,
            "KTAS": ktas,
            "q (psf)": q_psf,
            "g Effective (ft/sec^2)": g_effective_ftpers2,
            "CL": cl,
            "CD Polar": cd_polar,
            "Delta CD Reynolds": dcdre,
            "CD": cd,
            "L/D": l_over_d,
            "Mach(L/D)": mach_l_over_d,
            "FF Factor Eng 1": kwfx1,
            "FF Factor Eng 2": kwfx2,
            "Climb Rate (ft/min)": roc_ftpermin,
            "Accel (knots/min)": accel_ktpermin,
            "Excess Thrust Climb (lb)": fex_climb_lb,
            "Excess Thrust Accel (lb)": fex_accel_lb,
            "Excess Thrust Total (lb)": fex_total_lb,
            "Fnet Total (lb)": fnet_total_lb,
            "Fnet Eng 1 (lb)": fnet_eng1_lb,
            "Fnet Eng 2 (lb)": fnet_eng2_lb,
            "Fnet Cor Eng 1 (lb)": fnet_eng1_cor_lb,
            "Fnet Cor Eng 2 (lb)": fnet_eng2_cor_lb,
            "Fnet Cor Total (lb)": fnet_total_cor_lb,
            "FHV ratio": fhv_ratio,
            "FHV (BTU/lb)": fhv_btuperlb,
            "Wfdot Base Eng 1 (lb/hr)": wfdot_base_eng1_lbperhr,
            "Wfdot Base Eng 2 (lb/hr)": wfdot_base_eng2_lbperhr,
            "Fuel Flow Engine #1 (kg/hr)": wfdot_base_eng1_lbperhr * LBM_TO_KG,
            "Fuel Flow Engine #2 (kg/hr)": wfdot_base_eng2_lbperhr * LBM_TO_KG,
            "Wfdot Total (lb/hr)": wfdot_total_lbperhr,
            "Wfdot Eng 1 (lb/hr)": wfdot_eng1_lbperhr,
            "Wfdot Eng 2 (lb/hr)": wfdot_eng2_lbperhr,
            "Wfdot Cor Total (lb/hr)": wfdot_cor_total_lbperhr,
            "Corrected Fuel Flow Engine #1 (lb/hr)": wfdot_cor_eng1_lbperhr,
            "Corrected Fuel Flow Engine #2 (lb/hr)": wfdot_cor_eng2_lbperhr,
            "Corrected Fuel Flow Engine #1 (kg/hr)": wfdot_cor_eng1_lbperhr * LBM_TO_KG,
            "Corrected Fuel Flow Engine #2 (kg/hr)": wfdot_cor_eng2_lbperhr * LBM_TO_KG,
            "Corrected N1 Engine #1 (%)": n1c_eng1_pct,
            "Corrected N1 Engine #2 (%)": n1c_eng2_pct,
            "N1 Eng 1 (%)": n1_eng1_pct,
            "N1 Eng 2 (%)": n1_eng2_pct,
            "SFC Baseline (1/hr)": sfc_base_perhr,
            "Specific Range (nm/lb)": sr_nmperlb,
            "Specific Range (nm/kg)": sr_nmperlb / LBM_TO_KG
        }
        results.append(result)
    return results

# Collect data
data_with_bias = {
    'Tail': [],
    'Day': []
}

# Initialize other columns based on the keys of the first result to ensure all columns are present
for key in compute_derived_quantities_with_bias(0, 0, mach[0, 0], gross_wt_lb[0, 0], isa_dev_degc[0, 0], press_alt_ft[0, 0], track_true_deg[0, 0], latitude_deg[0, 0],
                                                flight_ids[0, 0],
                                                fhv_by_flt[0, 0],
                                                mass_bias_by_flt[0, 0],
                                                payload_by_flt[0, 0]
                                                )[0].keys():
    data_with_bias[key] = []

# Populate DataFrame
for tail_idx, tail in enumerate(tails):
    for day in range(n_days):
        derived_results = compute_derived_quantities_with_bias(
            tail_idx, day, mach[day, tail_idx], gross_wt_lb[day, tail_idx], isa_dev_degc[day, tail_idx],
            press_alt_ft[day, tail_idx], track_true_deg[day, tail_idx], latitude_deg[day, tail_idx],
            flight_ids[day, tail_idx],
            fhv_by_flt[day, tail_idx],
            mass_bias_by_flt[day, tail_idx],
            payload_by_flt[day, tail_idx])
        for result in derived_results:
            data_with_bias['Tail'].append(tail)
            data_with_bias['Day'].append(day)
            for key, value in result.items():
                data_with_bias[key].append(value)

# Create DataFrame
df = pd.DataFrame(data_with_bias)
print(df.head())

# Save as CSV
df.to_csv("synthetic_databaseXXX.csv", index=False)

plot_histogram_by_tail_all_days(df, "Mach Indicated", display_option="kde")
plot_histogram_by_tail_all_days(df, "Mach True", display_option="kde")
plot_histogram_by_tail_all_days(df, "Pressure Altitude Indicated (ft)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Pressure Altitude True (ft)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Gross Weight Indicated (lb)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Gross Weight Indicated (kg)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Gross Weight True (lb)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Gross Weight Effective (lb)", display_option="kde")
plot_histogram_by_tail_all_days(df, "W/delta Effective", display_option="kde")
plot_histogram_by_tail_all_days(df, "CL", display_option="kde")
plot_histogram_by_tail_all_days(df, "AoA True (deg)", display_option="kde")
plot_histogram_by_tail_all_days(df, "AoA Indicated (deg)", display_option="kde")
plot_histogram_by_tail_all_days(df, "CD Polar", display_option="kde")
plot_histogram_by_tail_all_days(df, "Delta CD Reynolds", display_option="kde")
plot_histogram_by_tail_all_days(df, "CD", display_option="kde")
plot_histogram_by_tail_all_days(df, "L/D", display_option="kde")
plot_histogram_by_tail_all_days(df, "Mach(L/D)", display_option="kde")
plot_histogram_by_tail_all_days(df, 'FF Factor Eng 1', bin_width=0.002, display_option="kde")
plot_histogram_by_tail_all_days(df, 'FF Factor Eng 2', bin_width=0.002, display_option="kde")
plot_histogram_by_tail_all_days(df, "Latitude (deg)", display_option="kde")
plot_histogram_by_tail_all_days(df, "True Track (deg)", display_option="kde")
plot_histogram_by_tail_all_days(df, "g Effective (ft/sec^2)", display_option="kde")
plot_histogram_by_tail_all_days(df, "ISA Deviation (deg C)", display_option="kde")
plot_histogram_by_tail_all_days(df, "FHV (BTU/lb)", display_option="kde")
plot_histogram_by_tail_all_days(df, "FHV ratio", display_option="kde")
plot_histogram_by_tail_all_days(df, "Climb Rate (ft/min)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Accel (knots/min)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Excess Thrust Climb (lb)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Excess Thrust Accel (lb)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Corrected N1 Engine #1 (%)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Corrected N1 Engine #2 (%)", display_option="kde")
plot_histogram_by_tail_all_days(df, "N1 Eng 1 (%)", display_option="kde")
plot_histogram_by_tail_all_days(df, "N1 Eng 2 (%)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Fnet Total (lb)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Fnet Eng 1 (lb)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Fnet Eng 2 (lb)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Fnet Cor Total (lb)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Fnet Cor Eng 1 (lb)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Fnet Cor Eng 2 (lb)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Wfdot Base Eng 1 (lb/hr)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Wfdot Base Eng 2 (lb/hr)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Fuel Flow Engine #1 (kg/hr)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Fuel Flow Engine #2 (kg/hr)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Wfdot Total (lb/hr)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Wfdot Eng 1 (lb/hr)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Wfdot Eng 2 (lb/hr)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Wfdot Cor Total (lb/hr)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Corrected Fuel Flow Engine #1 (lb/hr)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Corrected Fuel Flow Engine #2 (lb/hr)", display_option="kde")
plot_histogram_by_tail_all_days(df, "SFC Baseline (1/hr)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Specific Range (nm/lb)", display_option="kde")
plot_histogram_by_tail_all_days(df, "Specific Range (nm/kg)", display_option="kde")

# Plot the mean FF Factor Combined by tail
plot_bar_chart_by_tail(df, "FF Factor Eng 1", y_min=0.9, y_max=1.1)
plot_bar_chart_by_tail(df, "FF Factor Eng 2", y_min=0.9, y_max=1.1)
plot_bar_chart_by_tail(df, "Corrected N1 Engine #1 (%)", y_min=90, y_max=105)
plot_bar_chart_by_tail(df, "Corrected N1 Engine #2 (%)", y_min=90, y_max=105)
plot_bar_chart_by_tail(df, "CD Polar", y_min=0.0310, y_max=0.0340)
plot_scatter_by_tail(df, 'CD Polar', 'CL', xlim=(0.0220,0.0460), ylim=(0.30,0.75))
plot_scatter_by_tail(df, 'CD', 'CL', xlim=(0.0220,0.0460), ylim=(0.30,0.75))
plot_scatter_by_tail(df, "Mach True", "Pressure Altitude True (ft)", xlim=(0.75,0.81))
plot_scatter_by_tail(df, "Pressure Altitude True (ft)", "L/D")
plot_scatter_by_tail(df, "Pressure Altitude True (ft)", "Mach(L/D)")
plot_scatter_by_tail(df, "Pressure Altitude True (ft)", "W/delta Effective")
plot_scatter_by_tail(df, "Pressure Altitude True (ft)", "Delta CD Reynolds")
plot_scatter_by_tail(df, "Mach True", "Delta CD Reynolds", xlim=(0.75,0.81))
plot_scatter_by_tail(df, "Mach True", "CL", xlim=(0.75,0.81))
plot_scatter_by_tail(df, "Pressure Altitude True (ft)", "CL")
plot_scatter_by_tail(df, "Corrected N1 Engine #1 (%)", "N1 Eng 1 (%)")
plot_scatter_by_tail(df, "Corrected N1 Engine #2 (%)", "N1 Eng 2 (%)")
plot_scatter_by_tail(df, "Mach True", "Corrected N1 Engine #1 (%)")
plot_scatter_by_tail(df, "Mach True", "Corrected N1 Engine #2 (%)")
plot_scatter_by_tail(df, "Pressure Altitude True (ft)", "Corrected N1 Engine #1 (%)")
plot_scatter_by_tail(df, "Pressure Altitude True (ft)", "Corrected N1 Engine #2 (%)")
plot_scatter_by_tail(df, "Corrected N1 Engine #1 (%)", "Fnet Cor Eng 1 (lb)")
plot_scatter_by_tail(df, "Corrected N1 Engine #2 (%)", "Fnet Cor Eng 2 (lb)")
plot_scatter_by_tail(df, "Corrected N1 Engine #1 (%)", "Fnet Eng 1 (lb)")
plot_scatter_by_tail(df, "Corrected N1 Engine #2 (%)", "Fnet Eng 2 (lb)")
plot_scatter_by_tail(df, "Corrected N1 Engine #1 (%)", "Corrected Fuel Flow Engine #1 (lb/hr)", xlim=(90,112))
plot_scatter_by_tail(df, "Corrected N1 Engine #2 (%)", "Corrected Fuel Flow Engine #2 (lb/hr)", xlim=(90,112))
plot_scatter_by_tail(df, "Corrected N1 Engine #1 (%)", "Corrected Fuel Flow Engine #1 (kg/hr)", xlim=(90,112), ylim=(2500,5500))
plot_scatter_by_tail(df, "Corrected N1 Engine #2 (%)", "Corrected Fuel Flow Engine #2 (kg/hr)", xlim=(90,112), ylim=(2500,5500))
plot_scatter_by_tail(df, "Corrected N1 Engine #1 (%)", "Wfdot Eng 1 (lb/hr)")
plot_scatter_by_tail(df, "Corrected N1 Engine #2 (%)", "Wfdot Eng 2 (lb/hr)")
plot_scatter_by_tail(df, "Corrected N1 Engine #1 (%)", 'FF Factor Eng 1')
plot_scatter_by_tail(df, "Corrected N1 Engine #2 (%)", 'FF Factor Eng 2')
plot_scatter_by_tail(df, "Gross Weight True (lb)", "AoA True (deg)")
plot_scatter_by_tail(df, "AoA True (deg)", "AoA Indicated (deg)")
plot_scatter_by_tail(df, "AoA True (deg)", "CL")
plot_scatter_by_tail(df, "AoA Indicated (deg)", "CL")
plot_scatter_by_tail(df, "Corrected N1 Engine #1 (%)", "Fnet Cor Eng 1 (lb)")
plot_scatter_by_tail(df, "Corrected N1 Engine #2 (%)", "Fnet Cor Eng 2 (lb)")

# Report nominal characteristics of fits to linear engine relationships
coeffs = np.polyfit(df["Corrected N1 Engine #1 (%)"], df["Corrected Fuel Flow Engine #1 (lb/hr)"], deg=1) # Fit a linear regression (1st degree polynomial)
slope, intercept = coeffs
fit_line = np.poly1d(coeffs)(df["Corrected N1 Engine #1 (%)"]) # Predicted values from the linear model
r2 = r2_score(df["Corrected Fuel Flow Engine #1 (lb/hr)"], fit_line) # Calculate R^2
print(f"Linear fit, Wfdot Cor Eng 1 (lb/hr) vs N1 Cor Eng 1 (%): y = {slope:.5f}x + {intercept:.5f}; R2 = {r2:.5f}")

coeffs = np.polyfit(df["Corrected N1 Engine #2 (%)"], df["Corrected Fuel Flow Engine #2 (lb/hr)"], deg=1) # Fit a linear regression (1st degree polynomial)
slope, intercept = coeffs
fit_line = np.poly1d(coeffs)(df["Corrected N1 Engine #2 (%)"]) # Predicted values from the linear model
r2 = r2_score(df["Corrected Fuel Flow Engine #2 (lb/hr)"], fit_line) # Calculate R^2
print(f"Linear fit, Wfdot Cor Eng 2 (lb/hr) vs N1 Cor Eng 2 (%): y = {slope:.5f}x + {intercept:.5f}; R2 = {r2:.5f}")

coeffs = np.polyfit(df["Fnet Cor Eng 1 (lb)"], df["Corrected N1 Engine #1 (%)"], deg=1) # Fit a linear regression (1st degree polynomial)
slope, intercept = coeffs
fit_line = np.poly1d(coeffs)(df["Fnet Cor Eng 1 (lb)"]) # Predicted values from the linear model
r2 = r2_score(df["Corrected N1 Engine #1 (%)"], fit_line) # Calculate R^2
print(f"Linear fit, N1 Cor Eng 1 (%) VS Fnet Cor Eng 1 (lb): y = {slope:.7f}x + {intercept:.5f}; R2 = {r2:.5f}")

coeffs = np.polyfit(df["Fnet Cor Eng 2 (lb)"], df["Corrected N1 Engine #2 (%)"], deg=1) # Fit a linear regression (1st degree polynomial)
slope, intercept = coeffs
fit_line = np.poly1d(coeffs)(df["Fnet Cor Eng 2 (lb)"]) # Predicted values from the linear model
r2 = r2_score(df["Corrected N1 Engine #2 (%)"], fit_line) # Calculate R^2
print(f"Linear fit, N1 Cor Eng 2 (%) VS Fnet Cor Eng 2 (lb): y = {slope:.7f}x + {intercept:.5f}; R2 = {r2:.5f}")

# Report nominal characteristics of fits to linear engine relationships
coeffs = np.polyfit(df["AoA Indicated (deg)"], df["CL"], deg=1) # Fit a linear regression (1st degree polynomial)
slope, intercept = coeffs
fit_line = np.poly1d(coeffs)(df["AoA Indicated (deg)"]) # Predicted values from the linear model
r2 = r2_score(df["CL"], fit_line) # Calculate R^2
print(f"Linear fit, CL vs AoA Indicated (deg): y = {slope:.5f}x + {intercept:.5f}; R2 = {r2:.5f}")

means_by_tail = df.groupby("Tail")["FF Factor Eng 1"].mean()
print(means_by_tail)

means_by_tail = df.groupby("Tail")["FF Factor Eng 2"].mean()
print(means_by_tail)