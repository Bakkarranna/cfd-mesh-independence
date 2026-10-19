# cfd-mesh-independence

A mesh-independence study on a NACA 0012 airfoil with y+ monitoring.

## What this project is

A Python lab that builds a family of structured C-grid / O-mesh refinements
around a NACA 0012 airfoil, computes lift and drag with a teaching surrogate,
estimates y+ for each mesh, and decides whether the drag result has converged
with respect to mesh refinement.

It is a meshing study, not a production CFD run. The aerodynamics come from a
transparent teaching model (thin-airfoil theory for lift, turbulent flat-plate
skin friction plus a small pressure-drag placeholder for drag). That is
intentional: the point of the repo is the workflow around mesh design and y+,
not a production-dictionary drag value.

## Why this project exists

Boundary-layer meshing controls how well a CFD solution resolves the near-wall
velocity gradient. If the first cell is too far from the wall, y+ is too large
for the wall treatment you chose, and the skin-friction and separation
predictions degrade.

This project demonstrates the loop a CFD engineer runs in practice:

1. Pick a target y+ for the wall treatment. Here the target is 30, which is in
   the log-layer band for a wall-function treatment.
2. Build a family of meshes with different cell counts but the same near-wall
   treatment.
3. Compute the loads and the y+ for each mesh.
4. Decide whether the result is mesh-independent, meaning further refinement
   does not change the answer within tolerance.

The project is written so an interviewer can see three concrete things:

- how a structured airfoil mesh is described and counted,
- how y+ is estimated from wall shear and first-cell height,
- and how you decide that a mesh is independent of further refinement.

## What is inside

```
cfd-mesh-independence/
├── pyproject.toml          # project metadata, dependencies, test config
├── LICENSE                  # MIT
├── .gitignore
├── README.md                # this file
├── src/
│   ├── __init__.py
│   ├── cli.py               # entrypoint: run-study
│   ├── geometry.py          # NACA 4-digit and NACA 0012 coordinate generation
│   ├── mesh.py              # structured C-grid / O-mesh builder and cell-count helpers
│   ├── flow.py              # teaching surrogate aerodynamics (Cl, Cd)
│   ├── yplus.py             # y+ and first-cell-height estimation
│   └── study.py             # study runner: sweeps meshes, writes CSV + plot + RESULTS.md
├── tests/
│   ├── __init__.py
│   ├── test_mesh.py         # geometry, mesh, and y+ tests
│   └── test_study.py        # study runner, convergence logic, and output tests
├── outputs/
│   ├── mesh_convergence.csv
│   ├── mesh_convergence.png
│   └── RESULTS.md           # auto-generated results report
└── docs/
    └── (this project’s narrative lives in the README and RESULTS.md)
```

## How to run

Install the project in the shared virtual environment and run the study:

```bash
cd f1-data-projects/cfd-mesh-independence
.venv/Scripts/pip install -e ".[dev]"
.venv/Scripts/pytest tests/ -v
.venv/Scripts/python -m cfd_mesh_independence.study
```

Or use the entrypoint:

```bash
.venv/Scripts/run-study
```

The study runs a small grid by default: angles of attack 0° and 5°, six surface
resolutions, and six matched radial counts. It writes:

- `outputs/mesh_convergence.csv` — one row per mesh
- `outputs/mesh_convergence.png` — Cd and y+ versus cell count
- `outputs/RESULTS.md` — a short results report with the actual numbers

The CSV and PNG are regenerated each time the study runs; they are not committed
to git.

## How the teaching model works

### Geometry

The airfoil shape is generated from the standard NACA 4-digit thickness and
camber formulas. The NACA 0012 is the 4-digit shape with 12% thickness and zero
camber. The coordinates are closed, symmetric about the chord line, and span
from the leading edge (x = 0) to the trailing edge (x = chord).

### Mesh

The structured C-grid / O-mesh is built in geometric space:

- The airfoil surface is resampled to a fixed number of points using arc-length
  spacing.
- A geometric progression is used for the wall-normal spacing so the first cell
  height can be set explicitly and the cells expand smoothly outward.
- The far-field boundary is placed many chords away so it does not pollute the
  near-wall solution.

The mesh report records cell count, first-cell height, clustering ratio,
surface points, and total points. Cell count is the product of the
circumferential divisions and the radial divisions.

### Aerodynamics

- Lift uses thin-airfoil theory: Cl = 2π α (α in radians). This is a legitimate
  idealisation for a symmetric airfoil at small angles, not a random number.
- Drag uses a turbulent flat-plate skin-friction correlation (c_f = 0.027 /
  Re^(1/7)) plus a small constant pressure-drag placeholder. The placeholder is
  intentionally small and constant across the mesh study so the convergence story
  is driven by the mesh, not by a changing drag model.

### y+

y+ is estimated from the flat-plate wall shear and the first-cell height:

- u_tau = sqrt(tau_w / rho)
- y+ = (u_tau * y) / nu

The first-cell height is chosen to put the cell centre at the target y+. Because
the first-cell height is held constant across the mesh series, y+ is the same on
every mesh in this run. The study keeps that fixed on purpose: it shows that the
mesh can be refined in the outer layers without changing the near-wall treatment,
which is exactly the question a mesh-independence study answers.

## How convergence is decided

For each angle of attack, the study compares the drag coefficient on one mesh to
the drag coefficient on the previous coarser mesh. If the difference is below
1e-3, the result is flagged as mesh-independent for that angle. Because the
teaching Cd is constant across meshes, the flag becomes True once a second mesh
result exists for the angle.

In a production study you would vary the first-cell height as the mesh is
refined, and you would compute y+ from a converged wall shear rather than from a
planning estimate. That is deliberately left to the verification and validation
companion project, which compares computed results with experiment and discusses
numerical uncertainty.

## Key CFD ideas you can talk about from this repo

- **Target y+ and wall treatment.** Why wall-function treatments want the first
  cell in the log layer (y+ ~ 30–300) and why resolved boundary layers want
  y+ < 1.
- **Structured near-wall meshing.** C-grid / O-mesh topology, geometric
  clustering, and the number of cells on the surface and through the boundary
  layer.
- **Mesh independence.** What it means, how you test for it, and why you usually
  refine in the far field first and the near-wall region last.
- **Cost of refinement.** How cell count grows with surface resolution, and why
  near-wall clustering is the expensive part of an airfoil mesh.
- **Teaching models vs. production solvers.** Why a transparent surrogate is
  useful for a portfolio project, and what a production study would do
  differently.

## What a reviewer or interviewer might ask

- How do you choose a target y+?
- What changes when you move from a wall-function treatment to a low-Re resolved
  boundary layer?
- How do you know a mesh is independent of further refinement?
- What is the cost of reducing first-cell height?
- What is the difference between this study and a real RANS run? What would you
  add to make it production-grade?

## License

MIT. See `LICENSE`.
