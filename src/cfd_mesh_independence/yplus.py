"""y+ estimation for wall-bounded CFD.

y+ is the dimensionless wall distance used to decide whether a near-wall mesh
is consistent with the wall treatment you intend to use:

- For a wall-function treatment (log-law bridging), you usually want the
  first cell centre to sit in the log layer, e.g. 30 < y+ < 300.
- For a low-Reynolds-number (resolved) treatment, you want the wall shear to
  be resolved, e.g. y+ < 1.

The formulas here are the standard definitions used in most CFD textbooks and
pre-processors. They operate on a representative wall shear stress, so they
are a planning tool for mesh design rather than a post-processing diagnostic
from a converged solution. That distinction matters in an interview, and the
README calls it out explicitly.
"""

from __future__ import annotations

import numpy as np


def u_tau_from_tau_w(tau_w: float, rho: float) -> float:
    """Return the friction velocity u_tau = sqrt(tau_w / rho).

    Parameters
    ----------
    tau_w:
        Wall shear stress in Pa.
    rho:
        Fluid density in kg/m^3.
    """
    if tau_w <= 0.0 or rho <= 0.0:
        raise ValueError("tau_w and rho must be positive")
    return float(np.sqrt(tau_w / rho))


def y_plus_from_height(y: float, tau_w: float, rho: float, mu: float) -> float:
    """Return y+ for a wall-normal distance y given a wall shear stress.

    y+ = (u_tau * y) / nu, with nu = mu / rho.

    Parameters
    ----------
    y:
        Wall-normal distance of the cell centre, in m.
    tau_w:
        Wall shear stress in Pa.
    rho:
        Fluid density in kg/m^3.
    mu:
        Dynamic viscosity in Pa.s.
    """
    u_tau = u_tau_from_tau_w(tau_w, rho)
    nu = mu / rho
    if nu <= 0.0:
        raise ValueError("mu and rho must be positive")
    return float(u_tau * y / nu)


def first_cell_height_for_yplus(
    yplus_target: float,
    tau_w: float,
    rho: float,
    mu: float,
) -> float:
    """Return the wall-normal distance that puts the cell centre at yplus_target.

    y = (yplus_target * nu) / u_tau.

    Parameters
    ----------
    yplus_target:
        Desired y+ value for the cell centre (e.g. 1.0 or 30.0).
    tau_w:
        Expected wall shear stress in Pa.
    rho:
        Fluid density in kg/m^3.
    mu:
        Dynamic viscosity in Pa.s.
    """
    u_tau = u_tau_from_tau_w(tau_w, rho)
    nu = mu / rho
    return float(yplus_target * nu / u_tau)


def c_f_flat_plate_turbulent(Re: float) -> float:
    """Return a representative skin-friction coefficient for a turbulent flat
    plate using the 1/7th-power law correlation c_f = 0.027 / Re^{1/7}.

    This is the correlation used by the study as a planning estimate for the
    airfoil wall shear. It is not a substitute for an actual boundary-layer
    solve.
    """
    if Re <= 0.0:
        raise ValueError("Re must be positive")
    return 0.027 / (Re ** (1.0 / 7.0))


def tau_w_from_c_f(c_f: float, rho: float, u_inf: float) -> float:
    """Return wall shear stress from a skin-friction coefficient.

    tau_w = 0.5 * rho * u_inf^2 * c_f.
    """
    return 0.5 * rho * u_inf ** 2 * c_f


def estimate_yplus_for_airfoil(
    chord: float,
    u_inf: float,
    rho: float,
    mu: float,
    yplus_target: float = 30.0,
) -> dict:
    """Estimate the first-cell height and resulting y+ for an airfoil at the
    design condition.

    Uses the turbulent flat-plate correlation as a representative skin friction.
    Returns a dict with the key quantities so the study can report them
    alongside each mesh.

    Returns
    -------
    result:
        Dictionary with keys: Re_c, c_f, tau_w, u_tau, nu, first_cell_height,
        yplus_at_target, yplus_from_first.
    """
    Re_c = rho * u_inf * chord / mu
    c_f = c_f_flat_plate_turbulent(Re_c)
    tau_w = tau_w_from_c_f(c_f, rho, u_inf)
    u_tau = u_tau_from_tau_w(tau_w, rho)
    nu = mu / rho
    first_cell_height = first_cell_height_for_yplus(yplus_target, tau_w, rho, mu)
    return {
        "Re_c": float(Re_c),
        "c_f": float(c_f),
        "tau_w": float(tau_w),
        "u_tau": float(u_tau),
        "nu": float(nu),
        "first_cell_height": float(first_cell_height),
        "yplus_at_target": float(yplus_target),
        "yplus_from_first": float(yplus_target),
    }
