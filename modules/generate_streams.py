"""
generate_streams.py

Module for generating mock stellar streams
in axisymmetric logarithmic halo potentials.

Author: Vasu Pipwala
"""

from __future__ import annotations
import numpy as np
import astropy.units as u
import datetime
import platform
import pickle
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import gala
import astropy
from gala.units import galactic
from gala.potential import Hamiltonian, LogarithmicPotential
from gala.dynamics import PhaseSpacePosition
from gala.dynamics.mockstream import MockStreamGenerator, FardalStreamDF
from astropy.coordinates import (
    CartesianRepresentation,
    CartesianDifferential
)

# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Configuration Dataclasses
# ---------------------------------------------------------------------

@dataclass
class TimeConfig:
    t_backward: u.Quantity = 4 * u.Gyr
    dt: u.Quantity = 1 * u.Myr


@dataclass
class ProgenitorConfig:
    mass: u.Quantity
    initial_position: u.Quantity
    initial_velocity: u.Quantity


@dataclass
class HaloConfig:
    name: str
    q: float


@dataclass
class ExperimentConfig:
    time: TimeConfig
    progenitor_masses: List[u.Quantity]
    halo_shapes: Dict[str, float]
    n_particles: int = 1500


# ---------------------------------------------------------------------
# Core Simulation Class
# ---------------------------------------------------------------------

class StellarStreamSimulator:
    """
    Main simulation engine for generating stellar streams.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results: List[Dict[str, Any]] = []

        logger.info("StellarStreamSimulator initialized.")

    # ---------------------------------------------------------------
    # Internal Builders
    # ---------------------------------------------------------------

    @staticmethod
    def _make_hamiltonian(q: float) -> Hamiltonian:
        pot = LogarithmicPotential(
            v_c=220 * u.km / u.s,
            r_h=12 * u.kpc,
            q1=1.0,
            q2=1.0,
            q3=q,
            units=galactic
        )
        return Hamiltonian(pot)

    def _make_time_array(self) -> u.Quantity:
        cfg = self.config.time
        t = np.arange(
            0,
            -cfg.t_backward.to_value(u.Myr),
            -cfg.dt.to_value(u.Myr)
        ) * u.Myr
        return t

    @staticmethod
    def _initial_phase_space(pos, vel) -> PhaseSpacePosition:
        return PhaseSpacePosition(pos=pos, vel=vel)

    # ---------------------------------------------------------------
    # Snapshot Extraction Utility
    # ---------------------------------------------------------------

    @staticmethod
    def extract_stream_snapshot(
        stream,
        t_array: u.Quantity,
        time_index: int = 0,
        reverse_time_axis: bool = True
    ) -> PhaseSpacePosition:
        """
        Extract a single time snapshot from a gala mock stream.

        Parameters
        ----------
        stream : gala.dynamics.PhaseSpacePosition or tuple
            Stream object returned by MockStreamGenerator.
        t_array : Quantity
            Time array used during integration.
        time_index : int
            Index of the time snapshot to extract.
        reverse_time_axis : bool
            Whether to reverse the stored time axis
            (gala often stores backward integration reversed).

        Returns
        -------
        PhaseSpacePosition
            Phase-space snapshot at requested time.
        """

        # Some gala versions return (stream, prog)
        if isinstance(stream, tuple):
            mock = stream[0]
        else:
            mock = stream

        nt = len(t_array)
        total_points = mock.pos.x.shape[0]

        if total_points % nt != 0:
            raise ValueError(
                "Total number of stored points is not divisible "
                "by number of time steps."
            )

        npart = total_points // nt

        # Reshape to (nt, npart)
        x_all = mock.pos.x.reshape(nt, npart)
        y_all = mock.pos.y.reshape(nt, npart)
        z_all = mock.pos.z.reshape(nt, npart)

        vx_all = mock.vel.d_x.reshape(nt, npart)
        vy_all = mock.vel.d_y.reshape(nt, npart)
        vz_all = mock.vel.d_z.reshape(nt, npart)

        if reverse_time_axis:
            x_all = x_all[::-1]
            y_all = y_all[::-1]
            z_all = z_all[::-1]

            vx_all = vx_all[::-1]
            vy_all = vy_all[::-1]
            vz_all = vz_all[::-1]

        if time_index >= nt:
            raise IndexError(
                f"time_index={time_index} exceeds available "
                f"time steps (nt={nt})."
            )

        # Extract epoch
        x = x_all[time_index]
        y = y_all[time_index]
        z = z_all[time_index]

        vx = vx_all[time_index]
        vy = vy_all[time_index]
        vz = vz_all[time_index]

        pos = CartesianRepresentation(x, y, z)
        vel = CartesianDifferential(vx, vy, vz)
        pos = pos.with_differentials(vel)

        return PhaseSpacePosition(pos)

    # ---------------------------------------------------------------
    # Main Simulation Logic
    # ---------------------------------------------------------------

    def run(self) -> List[Dict[str, Any]]:
        logger.info("Starting stream generation experiment.")

        t = self._make_time_array()

        # Default initial conditions
        w0 = self._initial_phase_space(
            pos=[8.5, 0.0, 5.0] * u.kpc,
            vel=[0.0, 180.0, 60.0] * u.km / u.s
        )

        df = FardalStreamDF(gala_modified=True)

        for halo_name, q in self.config.halo_shapes.items():
            logger.info(f"Creating halo: {halo_name} (q={q})")

            H = self._make_hamiltonian(q)
            orbit = H.integrate_orbit(w0, t=t)
            generator = MockStreamGenerator(df, H)

            for i, mass in enumerate(self.config.progenitor_masses):

                logger.info(
                    f"Generating stream | halo={halo_name} | "
                    f"M={mass.value:.1e} Msun"
                )

                stream = generator.run(
                    w0,
                    prog_mass=mass,
                    n_particles=self.config.n_particles,
                    t=t
                )

                self.results.append({
                    "halo": halo_name,
                    "q": q,
                    "progenitor_id": i,
                    "mass": mass,
                    "stream": stream,
                    "orbit": orbit,
                    "t": t
                })

        logger.info("Experiment completed.")
        return self.results

    # ---------------------------------------------------------------
    # Saving / Metadata
    # ---------------------------------------------------------------

    def save(self, filename: str = "streams.pkl") -> None:
        payload = {
            "streams": self.results,
            "experiment_config": asdict(self.config),
            "metadata": {
                "created": datetime.datetime.now().isoformat(),
                "python_version": platform.python_version(),
                "gala_version": gala.__version__,
                "astropy_version": astropy.__version__,
                "description": "GC streams in axisymmetric halo potentials"
            }
        }

        with open(filename, "wb") as f:
            pickle.dump(payload, f)

        logger.info(f"Saved {len(self.results)} streams to {filename}")

    