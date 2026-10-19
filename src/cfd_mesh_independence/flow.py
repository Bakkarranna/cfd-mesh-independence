"""Teaching surrogate aerodynamics for the mesh-independence study.

This module provides a *deliberately* simplified model of the airfoil loads so
the study is runnable without an external CFD solver and the convergence story
is driven by the mesh, not by a black box. The model has two parts:

1. Lift is from thin-airfoil theory, Cl = 2 pi alpha (linear, inviscid, small
   angle). That is a legitimate idealisation, not a random number, and the
   project documents it as such.
2. Drag is from a turbulent flat-plate skin-friction correlation plus a small
   constant pressure-drag placeholder, so Cd changes with Reynolds number and
   is sensitive to the near-wall spacing through the wall shear used for y+.

This is a teaching engine, not a RANS solver. The README says so in plain
language so a reviewer does not mistake the numbers for a production result.
"""

from __future__ import annotations

import numpy as np

from .yplus import c_f_flat_plate_turbulent


def cl_thin_airfoil(alpha_deg: float) -> float:
    """Return the lift coefficient from thin-airfoil theory.

    Cl = 2 * pi * alpha, with alpha in radians. Valid for small angles on a
    symmetric airfoil at low Mach and moderate Re.

    Parameters
    ----------
    alpha_deg:
        Angle of attack in degrees.
    """
    alpha_rad = np.radians(alpha_deg)
    return float(2.0 * np.pi * alpha_rad)


def cd_skin_friction(
    chord: float,
    u_inf: float,
    rho: float,
    mu: float,
    *,
    pressure_drag_fraction: float = 0.15,
) -> float:
    """Return an estimate of the drag coefficient for the airfoil.

    Cd = c_f * (1 + pressure_drag_fraction).

    The skin-friction coefficient c_f comes from the turbulent flat-plate
    correlation. The pressure-drag fraction is a small constant placeholder to
    represent form drag, profile drag, and the fact that a real airfoil is not
    a flat plate. It is intentionally small and constant across the mesh study
    so the convergence story is driven by the mesh refinement, not by a
    changing drag model.

    Parameters
    ----------
    chord:
        Chord length in m.
    u_inf:
        Freestream speed in m/s.
    rho:
        Fluid density in kg/m^3.
    mu:
        Dynamic viscosity in Pa.s.
    pressure_drag_fraction:
        Fraction of the flat-plate skin friction added as a placeholder for
        pressure drag.
    """
    Re_c = rho * u_inf * chord / mu
    c_f = c_f_flat_plate_turbulent(Re_c)
    return float(c_f * (1.0 + pressure_drag_fraction))


def aero_loads(
    alpha_deg: float,
    chord: float,
    u_inf: float,
    rho: float,
    mu: float,
) -> dict:
    """Return lift and drag coefficients for one operating point.

    Returns
    -------
    result:
        Dictionary with keys: alpha_deg, Cl, Cd, Re_c, c_f.
    """
    Re_c = rho * u_inf * chord / mu
    c_f = c_f_flat_plate_turbulent(Re_c)
    return {
        "alpha_deg": float(alpha_deg),
        "Cl": cl_thin_airfoil(alpha_deg),
        "Cd": cd_skin_friction(chord, u_inf, rho, mu),
        "Re_c": float(Re_c),
        "c_f": float(c_f),
    }
