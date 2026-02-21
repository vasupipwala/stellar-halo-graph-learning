"""
stream_analysis.py

Dynamical diagnostics and statistical analysis tools
for stellar stream simulations.

Author: Vasu Pipwala
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import astropy.units as u
from typing import Dict, Tuple

from astropy.coordinates import CartesianRepresentation
from sklearn.decomposition import PCA
from scipy.signal import detrend
from scipy.fft import rfft, rfftfreq
from scipy.stats import spearmanr, pearsonr
from gala.dynamics.actionangle import find_actions_o2gf


# =====================================================================
# Core Diagnostics Class
# =====================================================================

class StreamDiagnostics:
    """
    High-level dynamical diagnostics for stellar streams.
    """

    # ------------------------------------------------------------
    # LEVEL 1: Stream–Plane Angle
    # ------------------------------------------------------------

    @staticmethod
    def compute_theta_plane(stream_orbits, orbit) -> np.ndarray:
        nt = stream_orbits.pos.x.shape[0]
        theta_plane = np.zeros(nt)

        prev_axis = None

        for i in range(nt):

            xyz = np.vstack([
                stream_orbits.pos.x[i].value,
                stream_orbits.pos.y[i].value,
                stream_orbits.pos.z[i].value
            ]).T

            xyz -= xyz.mean(axis=0)

            pca = PCA(n_components=1)
            pca.fit(xyz)

            axis = pca.components_[0]
            axis /= np.linalg.norm(axis)

            if prev_axis is not None and np.dot(axis, prev_axis) < 0:
                axis = -axis

            prev_axis = axis.copy()

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

    # ------------------------------------------------------------
    # LEVEL 2: Orbital Precession
    # ------------------------------------------------------------

    @staticmethod
    def compute_precession_curve(orbit) -> np.ndarray:

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

            if prev_L is not None and np.dot(L_unit, prev_L) < 0:
                L_unit = -L_unit

            Lhat[i] = L_unit
            prev_L = L_unit.copy()

        L0 = Lhat[0]
        cosang = np.clip(np.sum(Lhat * L0, axis=1), -1, 1)

        return np.degrees(np.arccos(cosang))

    # ------------------------------------------------------------
    # LEVEL 3: Thickness Evolution
    # ------------------------------------------------------------

    @staticmethod
    def compute_thickness_curve(stream_orbits) -> Tuple[np.ndarray, np.ndarray]:

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

            xyz -= xyz.mean(axis=0)

            pca = PCA(n_components=1)
            pca.fit(xyz)

            axis = pca.components_[0]
            axis /= np.linalg.norm(axis)

            if prev_axis is not None and np.dot(axis, prev_axis) < 0:
                axis = -axis

            prev_axis = axis.copy()

            proj = xyz @ axis
            recon = np.outer(proj, axis)
            residual = xyz - recon

            sigma_parallel[i] = np.std(proj)
            sigma_perp[i] = np.sqrt(np.mean(np.sum(residual**2, axis=1)))

        return sigma_perp, sigma_parallel

    @staticmethod
    def compute_thickness_growth(t_Gyr, sigma_perp, late_time_cut=-2.0):

        mask = t_Gyr > late_time_cut

        slope = np.polyfit(
            t_Gyr[mask],
            sigma_perp[mask],
            1
        )[0]

        return slope

    # ------------------------------------------------------------
    # LEVEL 4: Oscillation Diagnostics
    # ------------------------------------------------------------

    @staticmethod
    def compute_fft_diagnostics(theta_curve, t_Gyr):

        theta_detrended = detrend(theta_curve)
        dt = np.mean(np.diff(t_Gyr))

        yf = rfft(theta_detrended)
        xf = rfftfreq(len(theta_detrended), dt)

        power = np.abs(yf)
        power[0] = 0

        idx = np.argmax(power)

        return {
            "dominant_frequency": xf[idx],
            "dominant_amplitude": power[idx],
            "total_power": np.sum(power)
        }

    # ------------------------------------------------------------
    # LEVEL 5: Action–Angle Diagnostics
    # ------------------------------------------------------------

    @staticmethod
    def compute_actions_frequencies(orbit):

        result = find_actions_o2gf(orbit, N_max=8)

        J = result["actions"].to_value()
        Omega = result["freqs"].to(u.rad/u.Gyr).value
        Omega = np.array(Omega).flatten()

        return J, Omega

    @staticmethod
    def compute_frequency_metrics(Omega):

        Omega_R, Omega_phi, Omega_z = Omega

        Omega_norm = np.linalg.norm(Omega)

        return {
            "A_omega": Omega_z / Omega_R,
            "B_omega": Omega_z / Omega_phi,
            "fz_mean": Omega_z / Omega_norm,
            "Omega_norm": Omega_norm
        }


# =====================================================================
# Statistical Utilities
# =====================================================================

class StatisticalDiagnostics:

    @staticmethod
    def cohens_d(df, metric, halo1, halo2, return_direction=False):

        g1 = df[df.halo == halo1][metric].dropna().values
        g2 = df[df.halo == halo2][metric].dropna().values

        n1, n2 = len(g1), len(g2)

        mu1, mu2 = np.mean(g1), np.mean(g2)
        s1, s2 = np.std(g1, ddof=1), np.std(g2, ddof=1)

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

    @staticmethod
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

    @staticmethod
    def variance_ratio(df, metric):

        groups = df.groupby("halo")[metric]

        k = groups.ngroups
        N = len(df)
        grand_mean = df[metric].mean()

        ss_between = sum(
            len(group) * (group.mean() - grand_mean)**2
            for _, group in groups
        )

        ss_within = sum(
            ((group - group.mean())**2).sum()
            for _, group in groups
        )

        ms_between = ss_between / (k - 1)
        ms_within  = ss_within / (N - k)

        if ms_within == 0:
            return np.inf

        return ms_between / ms_within