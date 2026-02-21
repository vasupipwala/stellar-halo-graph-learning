import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import pickle
import datetime
import platform
import gala
import astropy

from gala.units import galactic
from gala.potential import Hamiltonian
from gala.potential import LogarithmicPotential
from gala.dynamics import PhaseSpacePosition
from gala.dynamics.mockstream import (
    MockStreamGenerator,
    FardalStreamDF
)

from tqdm.notebook import tqdm
import time
from mpl_toolkits.mplot3d import Axes3D




def make_galactic_hamiltonian(q=1.0):
    pot = LogarithmicPotential(
        v_c=220 * u.km/u.s,
        r_h=12 * u.kpc,
        q1=1.0,
        q2=1.0,
        q3=q,
        units=galactic
    )
    return Hamiltonian(pot)

def make_time_array(t_backward=4 * u.Gyr, dt=1 * u.Myr):
    """
    Create a backward time array for orbit and stream integration.
    """
    t = np.arange(
        0,
        -t_backward.to_value(u.Myr),
        -dt.to_value(u.Myr)
    ) * u.Myr
    return t

def initial_progenitor_phase_space():
    """
    Initial phase-space position of the progenitor globular cluster.
    """
    pos = [8.5, 0.0, 5.0] * u.kpc
    vel = [0.0, 180.0, 60.0] * u.km/u.s
    return PhaseSpacePosition(pos=pos, vel=vel)

def integrate_progenitor_orbit(hamiltonian, w0, t):
    """
    Integrate the progenitor orbit over the given time array.
    """
    orbit = hamiltonian.integrate_orbit(w0, t=t)
    return orbit

def generate_stream(
    hamiltonian,
    w0,
    t,
    progenitor_mass=1e4 * u.Msun,
    n_particles=1500,
):
    """
    Generate a stellar stream using a particle-spray model.
    
    Progenitor physics:
        - encoded via progenitor mass (tidal stripping)
    Galactic dynamics:
        - encoded via Hamiltonian
    """

    # Fardal (2015) DF, gala-modified parameters
    df = FardalStreamDF(gala_modified=True)

    generator = MockStreamGenerator(df, hamiltonian)

    stream = generator.run(
        w0,
        prog_mass=progenitor_mass,
        n_particles=n_particles,
        t=t
    )

    return stream

def run_experiment(
    t_backward=4 * u.Gyr,
    dt=1 * u.Myr,
    n_particles=1500,
):
    halo_shapes = {
        "spherical": 1.0,
        "oblate": 0.8,
        "prolate": 1.2
    }

    progenitor_masses = [
        8e3 * u.Msun,
        1e4 * u.Msun,
        2e4 * u.Msun,
    ]

    # Shared initial conditions
    w0 = initial_progenitor_phase_space()
    t = make_time_array(t_backward=t_backward, dt=dt)

    total_streams = len(halo_shapes) * len(progenitor_masses)
    streams = []

    with tqdm(
        total=total_streams,
        desc="Generating stellar streams",
        mininterval=0.5
    ) as pbar:

        for halo_name, q in halo_shapes.items():
            H = make_galactic_hamiltonian(q=q)

            # Integrate progenitor orbit ONCE per halo
            orbit = integrate_progenitor_orbit(H, w0, t)

            for i, mass in enumerate(progenitor_masses):

                print(
                    f"Starting stream: halo={halo_name}, "
                    f"M={mass.value:.1e} Msun",
                    flush=True
                )

                start = time.time()

                stream = generate_stream(
                    H,
                    w0=w0,
                    t=t,
                    progenitor_mass=mass,
                    n_particles=n_particles
                )

                elapsed = time.time() - start

                streams.append({
                    "halo": halo_name,
                    "q": q,
                    "progenitor_id": i,
                    "mass": mass,
                    "stream": stream,
                    "orbit": orbit,
                    "t": t
                })

                pbar.update(1)
                pbar.set_postfix({
                    "halo": halo_name,
                    "M_prog": f"{mass.value:.1e}",
                    "sec": f"{elapsed:.1f}"
                })

    return streams

def plot_stream(stream_w, title=""):
    x, y, z = stream_w.pos.xyz.to_value(u.kpc)

    plt.figure(figsize=(6, 5))
    plt.scatter(x, y, s=1, alpha=0.6)
    plt.xlabel("X [kpc]")
    plt.ylabel("Y [kpc]")
    plt.title(title)
    plt.axis("equal")
    plt.show()

def save_streams(streams, filename="streams.pkl"):
    """
    Save stream ensemble with metadata for reproducibility.
    """
    payload = {
        "streams": streams,
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

    print(f"Saved {len(streams)} streams to {filename}")