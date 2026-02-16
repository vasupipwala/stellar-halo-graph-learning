Project Execution Plan

Physics-Driven Stellar Streams and Graph Neural Networks

⸻

0. Project Objective (Anchor)

Core scientific question

Does halo geometry emerge as a natural organizing principle of stellar stream phase-space structure when progenitor physics is varied, and can this be detected using graph-based representation learning?

This is not a parameter-inference project.
This is a representation and invariance discovery experiment.

⸻

1. Conceptual Setup & Guardrails (Pre-coding)

1.1 Explicit assumptions (must be written before coding)
    •    Collisionless dynamics
    •    Fixed, time-independent potential
    •    Globular-cluster mass regime (no dynamical friction)
    •    Streams modeled as phase-space tracers
    •    ML used as diagnostic, not predictor

Deliverable
    •    A short “Assumptions & Scope” section (already drafted in README)

Failure mode
    •    If assumptions are unclear, later interpretation becomes invalid

⸻

1.2 Hypotheses (formalized)
    •    H₀ (null): Stream embeddings cluster primarily by progenitor mass
    •    H₁ (alternative): Stream embeddings cluster primarily by halo geometry

This makes the project falsifiable.

⸻

2. Physics Engine Construction (Streams + Orbits)

2.1 Define Galactic potentials

Task
    •    Implement three halo geometries:
    •    spherical
    •    oblate
    •    prolate

Why
    •    These are the minimal symmetries needed to induce or suppress stream–orbit misalignment (Pearson et al. 2015)

Output
    •    make_galactic_hamiltonian(q)

Validation check
    •    Circular orbits remain planar in spherical case
    •    Precession appears in flattened cases

⸻

2.2 Define progenitor initial conditions

Task
    •    Fix a single progenitor phase-space point

Why
    •    Changing initial conditions introduces unnecessary degeneracy
    •    Geometry, not orbit family, is the focus

Output
    •    initial_progenitor_phase_space()

Failure mode
    •    Changing ICs without reason invalidates comparisons

⸻

2.3 Define common time grid

Task
    •    Create backward time array used by both orbit and stream

Why
    •    Orbit–stream comparisons require identical temporal support

Output
    •    make_time_array()

Validation check
    •    Orbit length ≫ stream particle count

⸻

2.4 Integrate progenitor orbit

Task
    •    Integrate orbit as a test particle

Why
    •    Orbit is purely halo-driven
    •    Mass must not enter here

Output
    •    integrate_progenitor_orbit()

Validation check
    •    Orbit is smooth and continuous
    •    Shape changes with halo flattening

⸻

2.5 Generate stellar streams

Task
    •    Use particle-spray (Fardal 2015 DF)
    •    Vary progenitor mass only

Why
    •    Minimal progenitor physics
    •    Allows robustness testing

Output
    •    generate_stream()

Validation checks
    •    Stream thickness increases with progenitor mass
    •    Stream follows orbit closely only in spherical case


⸻

3. Stream Ensemble Design

3.1 Factorial experiment

Parameter   ------  Values
Halo geometry   ------  3
Progenitor mass ------  3
Total streams   ------  9

Why
    •    Minimal set that allows causal disentanglement
    •    Interpretation-friendly

Deliverable
    •    List of 9 labeled stream objects

Failure mode
    •    Adding more parameters too early reduces clarity

⸻

3.2 Sanity-check plots (mandatory)

For each stream:
    •    3D position space
    •    3D velocity space
    •    Stream vs orbit overlay

Why
    •    ML without physical sanity checks is meaningless

Checkpoint
    •    Misalignment appears only in non-spherical halos

⸻

4. Graph Construction (Physics → ML Interface)

4.1 Node definition

Nodes
    •    Stream particles

Node features
    •    (x, y, z, v_x, v_y, v_z)

Why
    •    Full phase space is where invariants live

⸻

4.2 Edge construction

Options
    •    k-nearest neighbors in 6D
    •    radius-based neighborhoods

Why
    •    Streams are locally coherent structures
    •    Graph captures continuity without grid bias

Deliverable
    •    One graph per stream

Failure mode
    •    Over-connecting destroys locality
    •    Under-connecting fragments stream

⸻

4.3 Graph-level representation

Goal
    •    One embedding per stream

Why
    •    The question is stream-level organization, not particle classification

⸻

5. GNN Architecture & Training

5.1 Architecture choice

Recommended
    •    Message-passing GNN (e.g. GraphConv / GIN)

Why
    •    Proven to capture relational structure
    •    Interpretable aggregation

Explicit non-requirements
    •    No deep, over-parameterized model
    •    No hyperparameter sweep

⸻

5.2 Training objective

Option A: supervised contrastive
    •    Positive pairs: same halo
    •    Negatives: different halo

Option B: weakly supervised
    •    Train embeddings
    •    Analyze clustering post hoc

Why
    •    Focus on representation geometry, not accuracy

⸻

5.3 Training constraints
    •    Small dataset (9 graphs)
    •    Heavy regularization
    •    Multiple random seeds

Why
    •    Overfitting is a feature here — interpretation matters more than generalization

⸻

6. Embedding Analysis (Core Scientific Step)

6.1 Visualization
    •    2D projection (PCA / UMAP)
    •    Color by:
    •    halo geometry
    •    progenitor mass

Primary diagnostic
    •    Do same-halo streams cluster?

⸻

6.2 Quantitative checks
    •    Intra-halo vs inter-halo distances
    •    Sensitivity to progenitor mass

Success criterion
    •    Halo signal dominates progenitor signal

⸻

6.3 Failure analysis (mandatory)

If embeddings cluster by:
    •    progenitor mass → ML not suitable
    •    random noise → representation insufficient

Both outcomes are scientifically valid.

⸻

7. Interpretation & Physics Mapping

7.1 Connect embeddings to dynamics

Ask:
    •    Which phase-space directions dominate?
    •    Is velocity space more informative than position space?
    •    Are misalignment-related features emphasized?

⸻

7.2 Explicit limitations

State clearly:
    •    No claim of Milky Way inference
    •    No claim of observational readiness

This protects scientific credibility.

⸻

8. Final Deliverables

Mandatory outputs
    1.    Stream ensemble (saved objects)
    2.    Orbit–stream diagnostic plots
    3.    Graph construction code
    4.    GNN training notebook/script
    5.    Embedding visualization
    6.    README + Abstract (already drafted)

⸻

9. Time Budget (Realistic 3–5 Days)

Day ------  Focus
Day 1   ------  Stream generation + orbit diagnostics
Day 2   ------  Graph construction + sanity checks
Day 3   ------  GNN training + embeddings
Day 4   ------  Interpretation + robustness
Day 5   ------  Write-up + figures


