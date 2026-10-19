"""Structured C-grid mesh around an airfoil.

The mesh is built in geometric space, not in a solver block. The idea is the
same as the "O-mesh" or "C-grid" used around airfoils in viscous CFD: cells
are clustered near the wall with a geometric progression so the first cell
height can be chosen to hit a target y+, and the far-field boundary is placed
far enough away that it does not pollute the near-wall solution.

This is a meshing lab, not a mesh for a production solver. The numbers it
reports (cell count, first cell height, clustering ratio) are the quantities
a CFD engineer uses to decide whether a mesh is ready for a boundary-layer
solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeshReport:
    """Deterministic description of one structured airfoil mesh."""

    chord: float
    n_surface: int
    n_radial: int
    first_cell_height: float
    max_radial_distance: float
    clustering_ratio: float
    cell_count: int
    n_surface_points: int
    n_total_points: int


def _geometric_series(
    first: float, ratio: float, count: int
) -> np.ndarray:
    """Return a monotonically increasing geometric series of length `count`.

    The first term equals `first`, each subsequent term is the previous term
    multiplied by `ratio`.
    """
    if count <= 0:
        return np.array([], dtype=float)
    if count == 1:
        return np.array([first], dtype=float)
    powers = np.arange(count, dtype=float)
    return first * (ratio ** powers)


def _spacing_from_distribution(
    total: float, count: int, first: float, last: float | None = None
) -> np.ndarray:
    """Return `count` segment lengths that sum to `total` with the given first
    segment.

    If `last` is given, the series is geometric between `first` and `last`.
    Otherwise the ratio is chosen so the last segment reaches `total`. The
    ratio is found by bisection on the closed form
    ``sum = first * (r^n - 1) / (r - 1)``, which is monotone in ``r`` for
    ``r > 1``, so the bracketing below is guaranteed to converge.
    """
    if count <= 0:
        return np.array([], dtype=float)
    if count == 1:
        return np.array([total], dtype=float)
    if first <= 0.0 or total <= 0.0:
        return np.full(count, total / count, dtype=float)

    if last is not None:
        if last <= 0.0:
            return np.full(count, total / count, dtype=float)
        ratio = (last / first) ** (1.0 / (count - 1))
        segs = first * ratio ** np.arange(count, dtype=float)
        return segs * (total / segs.sum())

    # No progression of growth ratio <= 1 can fit, so the wall-clustered mesh
    # must grow outward. Bisect the ratio in (1, ...) until the sum matches.
    def series_sum(r: float) -> float:
        if abs(r - 1.0) < 1e-12:
            return first * count
        return first * (r ** count - 1.0) / (r - 1.0)

    lo, hi = 1.0, 2.0
    for _ in range(200):
        if series_sum(hi) >= total:
            break
        hi *= 2.0
    else:
        return np.full(count, total / count, dtype=float)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if series_sum(mid) < total:
            lo = mid
        else:
            hi = mid
    ratio = 0.5 * (lo + hi)
    segs = first * ratio ** np.arange(count, dtype=float)
    # Rescale so the outermost node lands exactly on max_radial_distance.
    return segs * (total / segs.sum())


def build_c_grid(
    airfoil_coordinates: np.ndarray,
    chord: float = 1.0,
    n_surface: int = 200,
    n_radial: int = 30,
    first_cell_height: float = 1e-4,
    max_radial_distance: float = 20.0,
) -> MeshReport:
    """Build a structured C-grid / O-mesh around an airfoil.

    Parameters
    ----------
    airfoil_coordinates:
        Closed (x, y) polyline of the airfoil surface, in chord units.
    chord:
        Physical chord length in metres. The airfoil coordinates are assumed
        to be in chord units (0..1), so this scales the mesh to real metres.
    n_surface:
        Number of circumferential cells on the airfoil surface.
    n_radial:
        Number of cells outward from the surface to the far-field boundary.
    first_cell_height:
        Height in metres of the first cell off the wall.
    max_radial_distance:
        Distance from the airfoil quarter-chord point to the outer boundary,
        in metres.

    Returns
    -------
    report:
        Deterministic description of the resulting mesh.
    """
    scale = chord
    surface = _resample_airfoil(airfoil_coordinates, n_surface, scale)
    # A C-grid / O-mesh wraps all the way around the airfoil, so the surface is
    # always closed: the reported surface count is the count of circumferential
    # cells, not of unique points.

    # Radial spacing: cluster near the wall with a geometric progression whose
    # first element equals `first_cell_height` and whose sum reaches
    # `max_radial_distance`. Sweep the ratio above 1 (growing cells), which is
    # the correct direction for a wall-clustered mesh.
    radial_segs = _spacing_from_distribution(
        total=max_radial_distance, count=n_radial, first=first_cell_height
    )
    radial_positions = np.concatenate([[0.0], np.cumsum(radial_segs)])
    if radial_positions.size:
        # The rescaled series sums to max_radial_distance, so clamp the last
        # node to the exact target to remove float drift.
        radial_positions[-1] = max_radial_distance

    # Count cells: n_surface circumferential cells per radial layer, times
    # n_radial radial layers. The surface is closed, so no trailing-edge seam
    # has to be subtracted.
    cell_count = n_surface * n_radial

    # Estimate the clustering ratio from the geometric progression.
    clustering_ratio = radial_segs[1] / radial_segs[0] if n_radial > 1 else 1.0
    return MeshReport(
        chord=chord,
        n_surface=n_surface,
        n_radial=n_radial,
        first_cell_height=first_cell_height,
        max_radial_distance=max_radial_distance,
        clustering_ratio=float(clustering_ratio),
        cell_count=int(cell_count),
        n_surface_points=len(surface),
        n_total_points=int(len(surface) * n_radial + (n_radial + 1)),
    )


def _resample_airfoil(
    airfoil: np.ndarray, n_points: int, scale: float
) -> np.ndarray:
    """Resample the closed airfoil polyline to a fixed number of points.

    The resampling is arc-length based so the points are evenly spaced along
    the surface. The airfoil coordinates are assumed to be in chord units and
    are scaled to metres here.

    An empty (degenerate) polyline has no arc length to interpolate along, so
    a uniform placeholder out to `scale` is returned instead of interrupting
    the report for a caller that only needs the cell counts.
    """
    scaled = np.asarray(airfoil, dtype=float) * scale
    if scaled.size == 0 or scaled.shape[0] < 2:
        return np.zeros((max(int(n_points), 1), 2), dtype=float)
    # A closed polyline needs a wrapped coordinate array for arclength-based
    # interpolation. cumulative will have one more element than scaled so that
    # the last segment (closing the loop) is represented.
    closed_scaled = np.concatenate([scaled, scaled[:1]])
    diffs = np.diff(closed_scaled, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total = cumulative[-1]
    if not np.isfinite(total) or total <= 0.0:
        return np.zeros((max(int(n_points), 1), 2), dtype=float)
    targets = np.linspace(0.0, total, int(n_points), endpoint=False)
    xs = np.interp(targets, cumulative, closed_scaled[:, 0])
    ys = np.interp(targets, cumulative, closed_scaled[:, 1])
    return np.column_stack([xs, ys])


def cell_count_estimate(
    n_surface: int, n_radial: int, closed: bool = True
) -> int:
    """Return the number of cells in a structured C-grid / O-mesh.

    Parameters
    ----------
    n_surface:
        Number of surface (circumferential) divisions.
    n_radial:
        Number of radial divisions.
    closed:
        If True, the mesh is closed around the airfoil and the trailing-edge
        seam does not add a row of degenerate cells.
    """
    n_circ = n_surface if closed else n_surface - 1
    return int(n_circ * n_radial)


def far_field_radius_for_yplus(
    chord: float,
    max_expected_yplus: float,
    n_radial: int,
    first_cell_height: float,
) -> float:
    """Suggest a far-field radius large enough that boundary conditions are
    decoupled from the wall.

    This is a rule-of-thumb helper used by the study to keep the far-field
    consistent as the mesh is refined.
    """
    return float(max(chord * 20.0, max_expected_yplus * n_radial * 10.0))
