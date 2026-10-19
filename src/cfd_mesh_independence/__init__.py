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
from .geometry import naca0012_coordinates, naca4_coordinates, chordwise_spacing  # noqa: F401
from .mesh import build_c_grid, cell_count_estimate, MeshReport  # noqa: F401
from .yplus import (  # noqa: F401
    estimate_yplus_for_airfoil,
    y_plus_from_height,
    first_cell_height_for_yplus,
    u_tau_from_tau_w,
    c_f_flat_plate_turbulent,
    tau_w_from_c_f,
)
from .flow import aero_loads, cl_thin_airfoil, cd_skin_friction  # noqa: F401

# Optional submodules.
try:
    from .study import run_study, write_results_report, StudySpec, StudyRow  # noqa: F401
except _OptionalImportError:
    run_study = write_results_report = StudySpec = StudyRow = None  # type: ignore[assignment]

try:
    from .cli import main as cli_main  # noqa: F401
except _OptionalImportError:
    cli_main = None  # type: ignore[assignment]

__all__ = [
    "naca0012_coordinates",
    "naca4_coordinates",
    "chordwise_spacing",
    "build_c_grid",
    "cell_count_estimate",
    "MeshReport",
    "estimate_yplus_for_airfoil",
    "y_plus_from_height",
    "first_cell_height_for_yplus",
    "u_tau_from_tau_w",
    "c_f_flat_plate_turbulent",
    "tau_w_from_c_f",
    "aero_loads",
    "cl_thin_airfoil",
    "cd_skin_friction",
    "run_study",
    "write_results_report",
    "StudySpec",
    "StudyRow",
    "cli_main",
]
