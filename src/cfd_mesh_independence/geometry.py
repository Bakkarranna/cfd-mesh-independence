"""NACA 4-digit airfoil geometry.

This module generates closed polylines for NACA 4-digit airfoils from the
standard thickness and camber formulas. For this project the only shape in
use is the NACA 0012, but the generator is written generally so the same
code is reusable for the angle-of-attack sweep in the companion project.
"""

from __future__ import annotations

import numpy as np


def _thickness_distribution(x: np.ndarray, max_thickness_percent: float) -> np.ndarray:
    """Return the half-thickness distribution t(x) for a NACA 4-digit airfoil.

    Parameters
    ----------
    x:
        Chordwise coordinate from 0 (leading edge) to 1 (trailing edge).
    max_thickness_percent:
        Maximum thickness as a percentage of the chord (12 for NACA 0012).

    Returns
    -------
    yt:
        Half-thickness at each x, in the same units as x.
    """
    t = max_thickness_percent / 100.0
    yt = 5.0 * t * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x ** 2
        + 0.2843 * x ** 3
        - 0.1015 * x ** 4
    )
    return yt


def _camber_line(
    x: np.ndarray, max_camber_percent: float, camber_position_percent: float
) -> np.ndarray:
    """Return the mean camber line z(x) for a NACA 4-digit airfoil."""
    m = max_camber_percent / 100.0
    p = camber_position_percent / 100.0
    zc = np.zeros_like(x)
    mask = x <= p
    if np.any(mask):
        zc[mask] = (m / (p ** 2)) * (2.0 * p * x[mask] - x[mask] ** 2)
    if np.any(~mask):
        z = x[~mask]
        zc[~mask] = (m / ((1.0 - p) ** 2)) * (
            (1.0 - 2.0 * p) + 2.0 * p * z - z ** 2
        )
    return zc


def _surface_slope(x: np.ndarray, zc: np.ndarray) -> np.ndarray:
    """Approximate the local camber-line slope dz/dx on a monotonic x array.

    The input must be a single surface whose x is strictly monotonic, because
    the second-order finite difference divides by the local spacing. Passing
    the concatenated upper+lower array instead would reverse direction at the
    trailing edge, where the two spacings sum to zero and the gradient emits
    NaNs.
    """
    x = np.asarray(x, dtype=float)
    if x.size < 3:
        raise ValueError("at least 3 points are needed for the finite difference")
    dx = np.diff(x)
    if np.any(dx == 0.0):
        raise ValueError("x must be strictly monotonic")
    if np.any(dx[:-1] * dx[1:] <= 0.0):
        raise ValueError("x must not reverse direction")
    return np.gradient(zc, x, edge_order=2)


def naca4_coordinates(
    n_points: int = 240,
    max_thickness_percent: float = 12.0,
    max_camber_percent: float = 0.0,
    camber_position_percent: float = 40.0,
) -> np.ndarray:
    """Return a closed (x, y) polyline for a NACA 4-digit airfoil.

    The polyline starts at the leading edge on the upper surface, traces the
    upper surface to the trailing edge, then the lower surface back to the
    leading edge. The trailing edge (x = 1) is shared by the two surfaces and
    is not duplicated, so the last point lies on the lower surface at x = 0.

    Parameters
    ----------
    n_points:
        Total number of points on the closed polyline. Half go on the upper
        surface, half on the lower surface, with one overlapping point at the
        leading edge.
    max_thickness_percent:
        Maximum thickness as a percentage of chord.
    max_camber_percent:
        Maximum camber as a percentage of chord.
    camber_position_percent:
        Position of maximum camber as a percentage of chord.

    Returns
    -------
    coords:
        Array of shape (n_points, 2) in chord units.
    """
    upper_x, lower_x = _split_x(n_points)

    # Each surface is differentiated on its own monotonic x array; the slope of
    # the camber line cannot be taken across the concatenated array.
    yt_u = _thickness_distribution(upper_x, max_thickness_percent)
    yt_l = _thickness_distribution(lower_x, max_thickness_percent)
    zc_u = _camber_line(upper_x, max_camber_percent, camber_position_percent)
    zc_l = _camber_line(lower_x, max_camber_percent, camber_position_percent)
    theta_u = np.arctan2(_surface_slope(upper_x, zc_u), np.ones_like(upper_x))
    theta_l = np.arctan2(_surface_slope(lower_x, zc_l), np.ones_like(lower_x))

    # The first half of the loop runs the upper surface from the leading edge to
    # the trailing edge; the second half runs the lower surface back to the
    # leading edge. Both ends therefore land on the trailing edge at x = 1.
    x = np.concatenate([upper_x, lower_x])
    y = np.concatenate(
        [
            zc_u + yt_u * np.cos(theta_u),
            zc_l - yt_l * np.cos(theta_l),
        ]
    )
    coords = np.column_stack([x, y])
    if not np.isfinite(coords).all():
        raise ValueError("airfoil coordinate generation produced non-finite values")
    return coords


def _closed_airfoil_xy(
    upper_x: np.ndarray,
    lower_x: np.ndarray,
    yu: np.ndarray,
    yl: np.ndarray,
) -> np.ndarray:
    """Return a closed (x, y) polyline from upper and lower surface arrays."""
    return np.column_stack(
        [np.concatenate([upper_x, lower_x]), np.concatenate([yu, yl])]
    )



def _split_x(n_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the closed polyline into upper and lower x-coordinates.

    The upper surface is traced from LE to TE on a quadratically spaced array,
    the lower surface from TE back to LE on its own array, so the two halves
    mirror each other and both end exactly on the trailing-edge line x = 1:
    the leading edge (x = 0) is shared and is not duplicated.

    Parameters
    ----------
    n_points:
        Total number of points in the closed polyline.

    Returns
    -------
    upper_x:
        Chordwise coordinates of the upper surface, increasing from 0 to 1.
    lower_x:
        Chordwise coordinates of the lower surface, decreasing from 1 to 0.

    Notes
    -----
    The x values are squared (``x = (i / (n - 1)) ** 2``) so points cluster at
    the leading edge, where the surface curvature is highest. `_split_x` must
    stay a pure function of `n_points` so the same split drives the surface and
    the derivative without any shared mutable state.
    """
    n_upper = n_points // 2
    n_lower = n_points - n_upper
    upper_x = _quadratic_spacing(n_upper)
    lower_x = _quadratic_spacing(n_lower)[::-1]
    return upper_x, lower_x


def _quadratic_spacing(n: int) -> np.ndarray:
    """Return `n` points from 0 to 1 clustered at the leading edge."""
    if n < 2:
        return np.array([0.0, 1.0][:n])
    return (np.arange(n, dtype=float) / (n - 1)) ** 2


def naca0012_coordinates(n_points: int = 240) -> np.ndarray:
    """Return a closed (x, y) polyline for a NACA 0012.

    This is a thin wrapper around :func:`naca4_coordinates` with the 0012
    parameters fixed, provided because this project uses that shape in every
    figure.
    """
    return naca4_coordinates(
        n_points=n_points,
        max_thickness_percent=12.0,
        max_camber_percent=0.0,
        camber_position_percent=40.0,
    )


def chordwise_spacing(n_points: int = 240) -> np.ndarray:
    """Return the x-coordinate of each point on the closed airfoil polyline.

    Useful when you need to map a scalar field (pressure, skin friction) onto
    the airfoil surface and report it versus chordwise position.
    """
    coords = naca0012_coordinates(n_points)
    return coords[:, 0]
