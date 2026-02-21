import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import pickle
import datetime
import platform
import gala
import astropy
from astropy.coordinates import CartesianRepresentation, CartesianDifferential
from sklearn.decomposition import PCA
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import r2_score
import pandas as pd
from scipy.stats import f_oneway
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from scipy.stats import pearsonr

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.lines import Line2D
import seaborn as sns

from gala.units import galactic
from gala.potential import Hamiltonian
from gala.potential import LogarithmicPotential
from gala.dynamics import PhaseSpacePosition
from gala.dynamics.actionangle import find_actions_o2gf
from gala.dynamics.mockstream import (
    MockStreamGenerator,
    FardalStreamDF
)
from gala.integrate import LeapfrogIntegrator


from tqdm.notebook import tqdm
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib.animation import FFMpegWriter



# -----------------------------
# LEVEL 1: Stream–Plane Angle
# -----------------------------
def compute_theta_plane(stream_orbits, orbit):

    nt = stream_orbits.pos.x.shape[0]
    theta_plane = np.zeros(nt)

    prev_axis = None

    for i in range(nt):

        # Stream positions
        xyz = np.vstack([
            stream_orbits.pos.x[i].value,
            stream_orbits.pos.y[i].value,
            stream_orbits.pos.z[i].value
        ]).T

        # ---- Center before PCA (critical!) ----
        xyz -= xyz.mean(axis=0)

        # PCA
        pca = PCA(n_components=1)
        pca.fit(xyz)
        axis = pca.components_[0]
        axis /= np.linalg.norm(axis)

        # ---- Stabilize PCA sign across time ----
        if prev_axis is not None:
            if np.dot(axis, prev_axis) < 0:
                axis = -axis
        prev_axis = axis.copy()

        # Orbital angular momentum
        r = np.array([
            orbit.pos.x[i].value,
            orbit.pos.y[i].value,
            orbit.pos.z[i].value
        ])
        v = np.array([
            orbit.vel.d_x[i].value,
            orbit.vel.d_y[i].value,
            orbit.vel.d_z[i].value
        ])

        L = np.cross(r, v)
        normL = np.linalg.norm(L)

        if normL < 1e-12:
            theta_plane[i] = theta_plane[i-1] if i > 0 else 0
            continue

        Lhat = L / normL

        cosang = np.clip(np.abs(np.dot(axis, Lhat)), 0, 1)
        theta_plane[i] = np.degrees(np.arccos(cosang))

    return theta_plane


# -----------------------------
# LEVEL 2: Orbital Precession
# -----------------------------
def compute_precession_curve(orbit):

    nt = orbit.pos.x.shape[0]
    Lhat = np.zeros((nt, 3))

    prev_L = None

    for i in range(nt):

        r = np.array([
            orbit.pos.x[i].value,
            orbit.pos.y[i].value,
            orbit.pos.z[i].value
        ])
        v = np.array([
            orbit.vel.d_x[i].value,
            orbit.vel.d_y[i].value,
            orbit.vel.d_z[i].value
        ])

        L = np.cross(r, v)
        normL = np.linalg.norm(L)

        if normL < 1e-12:
            Lhat[i] = prev_L if prev_L is not None else np.array([0,0,1])
            continue

        L_unit = L / normL

        # ---- Stabilize orientation (prevent artificial flips) ----
        if prev_L is not None:
            if np.dot(L_unit, prev_L) < 0:
                L_unit = -L_unit

        Lhat[i] = L_unit
        prev_L = L_unit.copy()

    L0 = Lhat[0]

    cosang = np.clip(np.sum(Lhat * L0, axis=1), -1, 1)
    delta_L = np.degrees(np.arccos(cosang))

    return delta_L


# -----------------------------
# LEVEL 3: Thickness
# -----------------------------
def compute_thickness_curve(stream_orbits):

    nt = stream_orbits.pos.x.shape[0]
    sigma_perp = np.zeros(nt)
    sigma_parallel = np.zeros(nt)

    prev_axis = None

    for i in range(nt):

        xyz = np.vstack([
            stream_orbits.pos.x[i].value,
            stream_orbits.pos.y[i].value,
            stream_orbits.pos.z[i].value
        ]).T

        # Center first (critical!)
        xyz -= xyz.mean(axis=0)

        pca = PCA(n_components=1)
        pca.fit(xyz)
        axis = pca.components_[0]
        axis /= np.linalg.norm(axis)

        # Stabilize sign over time
        if prev_axis is not None:
            if np.dot(axis, prev_axis) < 0:
                axis = -axis
        prev_axis = axis.copy()

        # Parallel projection
        proj = xyz @ axis
        recon = np.outer(proj, axis)

        # Perpendicular residual
        residual = xyz - recon

        sigma_parallel[i] = np.std(proj)
        sigma_perp[i] = np.sqrt(np.mean(np.sum(residual**2, axis=1)))

    return sigma_perp, sigma_parallel


def compute_thickness_growth(t_Gyr, sigma_perp):

    # Use only late-time region for growth
    mask = t_Gyr > -2.0

    slope = np.polyfit(
        t_Gyr[mask],
        sigma_perp[mask],
        1
    )[0]

    return slope


# -----------------------------
# LEVEL 4: Oscillation Amplitude
# -----------------------------
def compute_fft_diagnostics(theta_curve, t_Gyr):

    # Remove linear trend (important!)
    theta_detrended = detrend(theta_curve)

    dt = np.mean(np.diff(t_Gyr))

    yf = rfft(theta_detrended)
    xf = rfftfreq(len(theta_detrended), dt)

    power = np.abs(yf)

    # Ignore zero-frequency
    power[0] = 0

    dominant_idx = np.argmax(power)

    dominant_freq = xf[dominant_idx]
    dominant_amp = power[dominant_idx]

    total_power = np.sum(power)

    return dominant_freq, dominant_amp, total_power


def cohens_d(df, metric, halo1, halo2, return_direction=False):

    """
    Compute Cohen's d effect size between two halo groups.
    Robust to small sample size and zero variance.
    """

    g1 = df[df.halo == halo1][metric].dropna().values
    g2 = df[df.halo == halo2][metric].dropna().values

    n1, n2 = len(g1), len(g2)

    mu1, mu2 = np.mean(g1), np.mean(g2)
    s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)

    # pooled std
    denom = (n1 + n2 - 2)

    if denom <= 0:
        return np.nan

    s_pooled = np.sqrt(
        ((n1 - 1)*s1**2 + (n2 - 1)*s2**2) / denom
    )

    if s_pooled == 0:
        return np.inf if abs(mu1 - mu2) > 0 else 0

    d = (mu1 - mu2) / s_pooled

    return d if return_direction else abs(d)


def interpret_effect(d):

    d = abs(d)

    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "moderate"
    elif d < 1.5:
        return "large"
    else:
        return "very large"

def variance_ratio(df, metric):
    """
    Compute between-halo / within-halo variance ratio.
    Equivalent to ANOVA F-statistic (for balanced samples).
    """

    groups = df.groupby("halo")[metric]

    # Number of groups
    k = groups.ngroups
    N = len(df)

    # Grand mean
    grand_mean = df[metric].mean()

    # Between-group sum of squares
    ss_between = sum(
        len(group) * (group.mean() - grand_mean)**2
        for _, group in groups
    )

    # Within-group sum of squares
    ss_within = sum(
        ((group - group.mean())**2).sum()
        for _, group in groups
    )

    # Mean squares
    ms_between = ss_between / (k - 1)
    ms_within  = ss_within / (N - k)

    if ms_within == 0:
        return np.inf

    return ms_between / ms_within

def compute_temporal_variance_ratio(metric_name):

    # Collect curves
    curves = {halo: [] for halo in halos}

    for _, row in df.iterrows():
        curves[row["halo"]].append(row[metric_name])

    # Convert to arrays
    for halo in halos:
        curves[halo] = np.vstack(curves[halo])  # shape: (n_mass, nt)

    R_t = np.zeros(nt)

    for i in range(nt):

        # Means per halo at time i
        halo_means = []
        halo_vars = []

        for halo in halos:
            vals = curves[halo][:, i]
            halo_means.append(np.mean(vals))
            halo_vars.append(np.var(vals))

        grand_mean = np.mean(halo_means)

        between = np.sum((np.array(halo_means) - grand_mean)**2)
        within = np.mean(halo_vars)

        R_t[i] = between / (within + 1e-8)

    return R_t

def compute_actions_frequencies(orbit):
    """
    Compute actions and fundamental frequencies
    using O2GF for a single integrated orbit.
    """

    result = find_actions_o2gf(
        orbit,
        N_max=8
    )

    # Actions (JR, Jphi, Jz)
    J = result["actions"].to_value()

    # Frequencies (rad / time)
    Omega = result["freqs"].to(u.rad/u.Gyr).value

    # Ensure 1D vector
    Omega = np.array(Omega).flatten()

    return J, Omega

def compute_frequency_metrics(Omega):

    Omega_R, Omega_phi, Omega_z = Omega

    A = Omega_z / Omega_R
    B = Omega_z / Omega_phi

    Omega_norm = np.linalg.norm(Omega)
    fz = Omega_z / Omega_norm

    return {
        "A_omega": A,
        "B_omega": B,
        "fz_mean": fz,
        "Omega_norm": Omega_norm
    }

