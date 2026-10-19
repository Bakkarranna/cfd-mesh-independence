"""Mesh-independence study for the NACA 0012 airfoil.

The study builds a family of structured C-grid meshes at a fixed angle of
attack and Reynolds number, computes the aerodynamic coefficients with the
teaching surrogate in :mod:`cfd_mesh_independence.flow`, estimates y+ for each
mesh from the flat-plate wall shear, and reports whether the drag coefficient
has converged with respect to cell count.

The output is a CSV and a two-panel plot (Cd vs. cell count, y+ vs. cell count).
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .flow import aero_loads
from .mesh import build_c_grid
from .yplus import estimate_yplus_for_airfoil, y_plus_from_height, tau_w_from_c_f


@dataclass(frozen=True)
class StudySpec:
    """Fixed parameters for one mesh-independence study."""

    chord: float = 1.0
    u_inf: float = 50.0
    rho: float = 1.225
    mu: float = 1.81e-5
    yplus_target: float = 30.0
    pressure_drag_fraction: float = 0.15
    far_field_factor: float = 20.0


@dataclass(frozen=True)
class StudyRow:
    """One row of the mesh-independence study."""

    angle_of_attack: float
    n_surface: int
    n_radial: int
    first_cell_height: float
    far_field_radius: float
    cell_count: int
    cl: float
    cd: float
    yplus: float
    c_f: float
    converged: bool


def run_study(
    spec: StudySpec,
    angles: Sequence[float] = (0.0, 5.0),
    surface_counts: Sequence[int] = (200, 300, 400, 600, 800, 1000),
    radial_counts: Sequence[int] = (20, 30, 40, 60, 80, 100),
    *,
    out_dir: pathlib.Path | str = "outputs",
    plot: bool = True,
    csv: bool = True,
    seed: int = 0,
) -> pd.DataFrame:
    """Run the mesh-independence study and return the results as a DataFrame.

    Parameters
    ----------
    spec:
        Fixed flow and geometry parameters for the study.
    angles:
        Angles of attack to study, in degrees.
    surface_counts:
        Candidate airfoil surface point counts. Each is used as the surface
        resolution for one mesh.
    radial_counts:
        Candidate radial cell counts. Matched 1:1 with `surface_counts`.
    out_dir:
        Directory for the generated CSV and plot.
    plot:
        If True, write a PNG of the convergence results.
    csv:
        If True, write a CSV of the results.
    seed:
        Unused, kept for reproducibility hooks in later projects.
    """
    rng = np.random.default_rng(seed)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[StudyRow] = []
    base_yplus = estimate_yplus_for_airfoil(
        spec.chord, spec.u_inf, spec.rho, spec.mu, spec.yplus_target
    )
    tau_w_ref = tau_w_from_c_f(base_yplus["c_f"], spec.rho, spec.u_inf)
    far_field = spec.chord * spec.far_field_factor

    for alpha in angles:
        cl_ref = aero_loads(alpha, spec.chord, spec.u_inf, spec.rho, spec.mu)["Cl"]
        cd_ref = aero_loads(alpha, spec.chord, spec.u_inf, spec.rho, spec.mu)["Cd"]

        for n_surface, n_radial in zip(surface_counts, radial_counts):
            mesh = build_c_grid(
                airfoil_coordinates=np.zeros((0, 2)),  # unused placeholder
                chord=spec.chord,
                n_surface=n_surface,
                n_radial=n_radial,
                first_cell_height=base_yplus["first_cell_height"],
                max_radial_distance=far_field,
            )
            # The teaching surrogate computes Cl and Cd from the flow, not from
            # the mesh, so the only mesh-dependent quantity in the estimates is
            # the y+ computed from the first-cell height. We keep the first-cell
            # height fixed across the mesh series so the y+ story is about the
            # near-wall treatment, not about varying the wall spacing by hand.
            ycell = base_yplus["first_cell_height"]
            yplus = y_plus_from_height(ycell, tau_w_ref, spec.rho, spec.mu)

            # Convergence decision: compare Cd on this mesh to Cd on the
            # previous larger mesh for the same angle of attack. Once the
            # change is below a tolerance, the drag result is mesh-independent
            # by this criterion.
            converged = _is_converged(rows, alpha, cd_ref, tolerance=1e-3)

            rows.append(
                StudyRow(
                    angle_of_attack=alpha,
                    n_surface=n_surface,
                    n_radial=n_radial,
                    first_cell_height=ycell,
                    far_field_radius=far_field,
                    cell_count=mesh.cell_count,
                    cl=cl_ref,
                    cd=cd_ref,
                    yplus=yplus,
                    c_f=base_yplus["c_f"],
                    converged=converged,
                )
            )

    df = pd.DataFrame(
        [
            {
                "angle_of_attack": r.angle_of_attack,
                "n_surface": r.n_surface,
                "n_radial": r.n_radial,
                "first_cell_height": r.first_cell_height,
                "far_field_radius": r.far_field_radius,
                "cell_count": r.cell_count,
                "Cl": r.cl,
                "Cd": r.cd,
                "yplus": r.yplus,
                "c_f": r.c_f,
                "mesh_independent": r.converged,
            }
            for r in rows
        ]
    )

    if csv:
        df.to_csv(out / "mesh_convergence.csv", index=False)

    if plot:
        _write_convergence_plot(df, out, angles)

    return df


def _is_converged(
    existing: list[StudyRow], angle: float, cd_new: float, *, tolerance: float
) -> bool:
    """Return True if the drag coefficient has stopped changing for the given
    angle of attack.

    Convergence is assessed against the most recent coarser mesh result for the
    same angle. If there is no previous result, the mesh is not yet converged.
    """
    previous = [r for r in existing if r.angle_of_attack == angle]
    if len(previous) < 1:
        return False
    cd_old = previous[-1].cd
    # Coerce to a plain Python bool: numpy scalars compare equal to True/False
    # but are not the same object, so `is False` in the tests would fail.
    return bool(abs(cd_new - cd_old) <= tolerance)


def _write_convergence_plot(df: pd.DataFrame, out: pathlib.Path, angles: Sequence[float]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, y, ylabel in [
        (axes[0], "Cd", "Drag coefficient Cd"),
        (axes[1], "yplus", "y+ at the wall"),
    ]:
        for alpha in angles:
            sub = df[df["angle_of_attack"] == alpha]
            marker = "o" if alpha == 0.0 else "s"
            linestyle = "-" if alpha == 0.0 else "--"
            ax.plot(
                sub["cell_count"],
                sub[y],
                marker=marker,
                linestyle=linestyle,
                label=f"alpha = {alpha:.1f}°",
            )
        ax.set_xlabel("Mesh cell count")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", linestyle=":", linewidth=0.5)
        ax.legend()

    axes[0].set_title("Drag coefficient vs. mesh refinement")
    axes[1].set_title("y+ at the wall vs. mesh refinement")
    fig.suptitle("NACA 0012 mesh-independence study", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out / "mesh_convergence.png", dpi=150)
    plt.close(fig)


def write_results_report(df: pd.DataFrame, spec: StudySpec, path: pathlib.Path) -> None:
    """Write a short RESULTS-style markdown report with the actual numbers.

    The report lists the study parameters, the convergence decision for each
    angle of attack, and the final Cd and y+ on the finest mesh.
    """
    lines: list[str] = []
    lines.append("# Mesh-independence study — results")
    lines.append("")
    lines.append("## Study parameters")
    lines.append("")
    lines.append(f"- Chord: {spec.chord} m")
    lines.append(f"- Freestream speed: {spec.u_inf} m/s")
    lines.append(f"- Density: {spec.rho} kg/m³")
    lines.append(f"- Dynamic viscosity: {spec.mu} Pa·s")
    lines.append(f"- Reynolds number (based on chord): {spec.rho * spec.u_inf * spec.chord / spec.mu:.2e}")
    lines.append(f"- Target y+: {spec.yplus_target}")
    lines.append(f"- Pressure-drag placeholder fraction: {spec.pressure_drag_fraction}")
    lines.append(f"- Far-field radius: {spec.chord * spec.far_field_factor} m")
    lines.append(f"- Convergence tolerance on Cd: 1e-3")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| alpha (°) | cells | Cd | y+ | mesh-independent |")
    lines.append("|----------:|------:|------:|--------:|:----------------:|")
    for _, row in df.iterrows():
        lines.append(
            f"| {row['angle_of_attack']:.1f} | {int(row['cell_count']):,} "
            f"| {row['Cd']:.5f} | {row['yplus']:.2f} | "
            f"{'yes' if row['mesh_independent'] else 'no'} |"
        )
    lines.append("")
    lines.append("## Discussion")
    lines.append("")
    lines.append(
        "Cd is computed from a turbulent flat-plate skin-friction correlation plus a "
        "small constant pressure-drag placeholder. It does not depend on the mesh in "
        "this teaching model, so the convergence flag is set only once a second mesh "
        "result exists for the same angle of attack and the difference is within the "
        "tolerance."
    )
    lines.append("")
    lines.append(
        "The y+ column is the actual y+ at the wall for the first cell height used in "
        "the study. It is the same on every mesh in this run because the first-cell "
        "height is held constant; the point of the study is to show that the mesh can "
        "be refined in the outer layers without changing the near-wall treatment."
    )
    lines.append("")
    lines.append(
        "A production study would vary the first-cell height as the mesh is refined and "
        "would compute y+ from a converged wall shear rather than from a planning "
        "estimate. That variation is intentionally left to Project F, which compares "
        "computed results with experiment and discusses numerical uncertainty."
    )
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append("- `mesh_convergence.csv` — raw results behind this report")
    lines.append("- `mesh_convergence.png` — convergence plot")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    spec = StudySpec()
    angles = (0.0, 5.0)
    surface_counts = (200, 300, 400, 600, 800, 1000)
    radial_counts = (20, 30, 40, 60, 80, 100)

    print("Running NACA 0012 mesh-independence study")
    print(f"  chord         = {spec.chord} m")
    print(f"  u_inf         = {spec.u_inf} m/s")
    print(f"  Re_c          = {spec.rho * spec.u_inf * spec.chord / spec.mu:.2e}")
    print(f"  target y+     = {spec.yplus_target}")
    print(f"  angles        = {angles}")
    print(f"  surface counts= {surface_counts}")
    print(f"  radial counts = {radial_counts}")

    df = run_study(
        spec,
        angles=angles,
        surface_counts=surface_counts,
        radial_counts=radial_counts,
    )

    out = pathlib.Path("outputs")
    print(f"\nResults written to {out}/")
    print(f"  rows            = {len(df)}")
    print(f"  final Cd (0°)  = {df[df['angle_of_attack'] == 0.0]['Cd'].iloc[-1]:.5f}")
    print(f"  final y+       = {df['yplus'].iloc[-1]:.2f}")
    print(f"  converged rows  = {int(df['mesh_independent'].sum())}")

    write_results_report(df, spec, out / "RESULTS.md")
    print(f"  report          = {out / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
