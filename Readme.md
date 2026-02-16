Title:- Physics-Driven Stellar Stream Ensembles and Graph-Based Representation Learning

1. Scientific Motivation

Stellar streams are extended, coherent structures formed when globular clusters or dwarf galaxies are tidally disrupted in a host galaxy. Since the pioneering work of Johnston et al. (1999) and Helmi & White (1999), streams have been recognized as sensitive probes of the Galactic potential. More recent studies have demonstrated that streams do not generally trace single orbits in non-spherical potentials, but instead exhibit stream–orbit misalignment driven by halo geometry and orbital precession (Fardal et al. 2015; Pearson et al. 2015).

Despite this sensitivity, extracting robust information from streams remains challenging because their observable properties depend on both:
    •    Galactic dynamics (halo shape, symmetry, precession), and
    •    progenitor physics (mass, internal motions, stripping history).

The central question motivating this project is therefore not whether streams encode halo information — that is well established — but:

Which aspects of stream phase-space structure are robust to progenitor variation and therefore suitable as dynamical diagnostics of the halo?

This project addresses that question using controlled numerical experiments and physics-guided machine learning.

⸻

2. Why an Idealized Setup Is Necessary

A common instinct is to begin with the most realistic possible Milky Way model: an NFW halo, baryonic disk, rotating bar, time evolution, and substructure. However, such realism obscures causal interpretation. When many physical effects are present simultaneously, it becomes impossible to determine why a given feature appears or whether it is robust.

This project therefore follows a long-standing tradition in galactic dynamics:

Dynamical insight is discovered in idealized systems and tested in realistic ones — not the other way around.

This philosophy underlies the development of:
    •    action–angle variables (Binney & Tremaine 2008),
    •    stream–orbit plane misalignment theory (Sanders & Binney 2013; Pearson et al. 2015),
    •    and invariant-based analyses of stellar systems.

Accordingly, the present setup is intentionally simplified to isolate first-order effects.

⸻

3. Galactic Potential Model

Current Choice

The Galactic potential is modeled as a single-component logarithmic halo with adjustable symmetry:
    •    spherical,
    •    oblate,
    •    prolate.

This choice provides:
    •    analytic smoothness,
    •    explicit control over halo geometry,
    •    a time-independent Hamiltonian,
    •    and computational efficiency.

Logarithmic potentials have been widely used as controlled testbeds in stream dynamics (e.g. Law & Majewski 2010; Pearson et al. 2015).

What Is Excluded (and Why)

The following components are not included at this stage:
    •    NFW radial profiles,
    •    baryonic disks (e.g. Miyamoto–Nagai),
    •    rotating bars,
    •    substructure,
    •    time dependence.

These effects are known to perturb streams (e.g. Dehnen 2000; Banik et al. 2019), but including them prematurely would entangle multiple physical mechanisms and defeat the project’s purpose. They are reserved for future stress tests, not baseline discovery.

⸻

4. Progenitor Modeling Philosophy

Progenitor Orbit

The globular cluster progenitor is treated as a test particle whose orbit is explicitly integrated in the Galactic Hamiltonian. This orbit is independent of progenitor mass, which is exact in collisionless dynamics and appropriate for globular-cluster masses where dynamical friction is negligible (Binney & Tremaine 2008).

This separation is essential: halo-driven orbital dynamics must not be contaminated by progenitor-specific physics.

Progenitor Physics Included

Progenitor physics enters only through finite progenitor mass, which controls:
    •    the tidal radius,
    •    the stripping rate,
    •    the first-order width of the stream.

This minimal inclusion is sufficient to generate realistic thin streams while preserving interpretability.

Progenitor Physics Excluded

The following are deliberately excluded:
    •    explicit internal velocity distributions (e.g. Maxwellian sampling),
    •    relaxation and mass segregation,
    •    binaries,
    •    self-gravity of the stream.

Such effects are either subdominant for thin globular-cluster streams or actively harmful to isolating halo-driven structure at this stage. This choice is consistent with particle-spray models used in the literature (Fardal et al. 2015; Küpper et al. 2012).

⸻

5. Stream Generation Method

Streams are generated using particle-spray methods following Fardal et al. (2015), as implemented in the gala dynamics framework. In this approach:
    •    stars escape near the Lagrange points,
    •    stripping occurs continuously along the orbit,
    •    stream–orbit misalignment emerges naturally in non-spherical potentials.

This method occupies a well-motivated middle ground:
    •    more physical than orbit-fitting,
    •    vastly more interpretable and efficient than full N-body simulations.

Each particle represents a phase-space tracer, not an individual star.

⸻

6. Experimental Design: Why 9 Streams Are Enough

The stream ensemble consists of:
    •    3 halo geometries (spherical, oblate, prolate),
    •    3 progenitor masses,
    •    9 total streams.

This is a factorial design intended to answer a causal question, not to perform statistical inference. The goal is to test existence and structure, not to map a full parameter space.

Adding more realizations at this stage would reduce interpretability without strengthening the central claim.

⸻

7. Why Use a Graph Neural Network?

Stellar streams are:
    •    extended,
    •    sparse,
    •    six-dimensional,
    •    and locally coherent.

Graphs provide a natural representation:
    •    nodes: stream particles,
    •    edges: local phase-space neighborhoods,
    •    node features: positions and velocities.

A GNN is therefore used as a representation learner, not as a predictive model.

⸻

8. What the GNN Is Testing (and What It Is Not)

The Representation Hypothesis

The central hypothesis is:

If halo geometry is a natural organizing principle of stream phase-space structure, then a GNN trained on raw phase-space graphs should cluster streams by halo geometry rather than progenitor mass.

This is not trivial. Machine-learning models often latch onto the easiest signals — in this case, progenitor-driven differences such as local dispersion — rather than global dynamical structure.

Falsifiability

The hypothesis can fail in meaningful ways:
    •    embeddings may cluster by progenitor mass,
    •    or show no clear organization at all.

Either outcome is scientifically informative.

Explicit Non-Claims

The GNN:
    •    does not infer halo parameters,
    •    does not model the Milky Way,
    •    does not claim direct applicability to Gaia data.

It is a diagnostic instrument, not a black box.

⸻

9. Relation to Observational Data

This project is not trained on Gaia data. Instead, it establishes a controlled representation space that can later be:
    1.    stress-tested with more realistic simulations,
    2.    and only then compared to observational streams.

This ordering follows best practices in both physics and ML.

⸻

10. Summary and Scope

This project constructs a minimal, interpretable, physics-driven framework for studying stellar streams and uses machine learning as a microscope rather than a predictor. By deliberately suppressing realism in favor of control, it asks a precise and falsifiable question about representation and invariance in stream dynamics. The result is not a final model of the Galaxy, but a principled foundation for future work.
