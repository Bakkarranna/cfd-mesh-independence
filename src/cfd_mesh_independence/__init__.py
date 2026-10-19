'''cfd_mesh_independence — NACA 0012 mesh-independence study with y+ monitoring.

Public API
---------

Geometry
    ``cfd_mesh_independence.geometry.naca0012_coordinates``
    ``cfd_mesh_independence.geometry.naca4_coordinates``
    ``cfd_mesh_independence.geometry.chordwise_spacing``

Mesh
    ``cfd_mesh_independence.mesh.build_c_grid``
    ``cfd_mesh_independence.mesh.cell_count_estimate``
    ``cfd_mesh_independence.mesh.MeshReport``

y+
    ``cfd_mesh_independence.yplus.estimate_yplus_for_airfoil``
    ``cfd_mesh_independence.yplus.y_plus_from_height``
    ``cfd_mesh_independence.yplus.first_cell_height_for_yplus``

Flow
    ``cfd_mesh_independence.flow.aero_loads``
    ``cfd_mesh_independence.flow.cl_thin_airfoil``
    ``cfd_mesh_independence.flow.cd_skin_friction``

Study
    ``cfd_mesh_independence.study.run_study``
    ``cfd_mesh_independence.study.write_results_report``
    ``cfd_mesh_independence.study.StudySpec``
    ``cfd_mesh_independence.study.StudyRow``

CLI
    ``cfd_mesh_independence.cli.main``
'''


class _OptionalImportError(ImportError):
    """Raised when an optional dependency is missing for one public submodule."""


def _import_optional(name: str):
    try:
        return __import__(name, fromlist=[''])
    except Exception as e:
        raise _OptionalImportError(
            f"Optional dependency for {name} is not installed: {e!r}"
        ) from e


# Core submodules that should always be importable.
from .flow import aero_loads, cd_skin_friction, cl_thin_airfoil
from .geometry import chordwise_spacing, naca0012_coordinates, naca4_coordinates
from .mesh import MeshReport, build_c_grid, cell_count_estimate
from .yplus import (
    c_f_flat_plate_turbulent,
    estimate_yplus_for_airfoil,
    first_cell_height_for_yplus,
    tau_w_from_c_f,
    u_tau_from_tau_w,
    y_plus_from_height,
)

# Optional submodules.
try:
    from .study import StudyRow, StudySpec, run_study, write_results_report
except _OptionalImportError:
    run_study = write_results_report = StudySpec = StudyRow = None  # type: ignore[assignment]

try:
    from .cli import main as cli_main
except _OptionalImportError:
    cli_main = None  # type: ignore[assignment]

__all__ = [
    "MeshReport",
    "StudyRow",
    "StudySpec",
    "aero_loads",
    "build_c_grid",
    "c_f_flat_plate_turbulent",
    "cd_skin_friction",
    "cell_count_estimate",
    "chordwise_spacing",
    "cl_thin_airfoil",
    "cli_main",
    "estimate_yplus_for_airfoil",
    "first_cell_height_for_yplus",
    "naca0012_coordinates",
    "naca4_coordinates",
    "run_study",
    "tau_w_from_c_f",
    "u_tau_from_tau_w",
    "write_results_report",
    "y_plus_from_height",
]
