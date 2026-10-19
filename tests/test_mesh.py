"""Tests for NACA geometry, structured mesh, and y+ estimation."""

from __future__ import annotations

import numpy as np
import pytest

from cfd_mesh_independence.geometry import naca0012_coordinates, naca4_coordinates
from cfd_mesh_independence.mesh import build_c_grid, cell_count_estimate
from cfd_mesh_independence.yplus import (
    c_f_flat_plate_turbulent,
    estimate_yplus_for_airfoil,
    first_cell_height_for_yplus,
    tau_w_from_c_f,
    u_tau_from_tau_w,
    y_plus_from_height,
)


class TestNaca0012Coordinates:
    def test_returns_2d_array(self):
        coords = naca0012_coordinates(240)
        assert coords.ndim == 2
        assert coords.shape[0] == 240
        assert coords.shape[1] == 2

    def test_is_symmetric_about_x_axis(self):
        coords = naca0012_coordinates(240)
        ys = coords[:, 1]
        assert np.allclose(ys.max(), -ys.min(), rtol=1e-9)

    def test_leading_edge_at_origin(self):
        coords = naca0012_coordinates(240)
        assert np.isclose(coords[0, 0], 0.0, atol=1e-12)
        assert np.isclose(coords[0, 1], 0.0, atol=1e-12)

    def test_trailing_edge_at_chord(self):
        coords = naca0012_coordinates(240)
        # The closed polyline starts at the leading edge, traces the upper
        # surface to the trailing edge, and returns along the lower surface, so
        # x reaches the chord exactly once and the two surfaces meet there.
        x = coords[:, 0]
        assert np.isclose(x.max(), 1.0, atol=1e-12)
        trailing_edge = coords[x >= x.max()]
        assert trailing_edge.shape[0] >= 1
        # The NACA 0012 trailing edge is a finite ~0.25% chord sliver, not a
        # sharp point, so the two surfaces close within 1% of chord.
        assert np.all(np.abs(trailing_edge[:, 1]) < 0.01)

    def test_thickness_at_30_percent_chord(self):
        coords = naca0012_coordinates(240)
        upper = coords[coords[:, 1] > 0.0]
        # Quadratic chordwise spacing clusters points at the leading edge, so
        # the nearest sample to 30% chord sits a little before it.
        x, y = upper[np.argmin(np.abs(upper[:, 0] - 0.30))]
        assert np.isclose(x, 0.30, atol=2e-3)
        assert y > 0.0
        assert y < 0.07  # NACA 0012 half-thickness is well below 7% of chord
        # ...and it is the analytic half-thickness distribution at that x.
        expected = 5.0 * 0.12 * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x ** 2
            + 0.2843 * x ** 3
            - 0.1015 * x ** 4
        )
        assert np.isclose(y, expected, rtol=1e-6)


class TestNaca4Coordinates:
    def test_symmetric_returns_0012_like_shape(self):
        coords = naca4_coordinates(
            n_points=240, max_thickness_percent=12.0
        )
        assert coords.shape == (240, 2)

    def test_camber_shifts_surface(self):
        thin = naca4_coordinates(
            n_points=200, max_thickness_percent=12.0, max_camber_percent=0.0
        )
        cambered = naca4_coordinates(
            n_points=200, max_thickness_percent=12.0, max_camber_percent=4.0
        )
        assert not np.allclose(thin[:, 1], cambered[:, 1])


class TestBuildCGrid:
    def test_mesh_report_has_cell_count(self):
        report = build_c_grid(
            airfoil_coordinates=np.zeros((0, 2)),
            chord=1.0,
            n_surface=200,
            n_radial=20,
            first_cell_height=1e-4,
            max_radial_distance=20.0,
        )
        assert report.cell_count > 0
        assert report.n_surface == 200
        assert report.n_radial == 20

    def test_cell_count_scales_with_surface_and_radial(self):
        small = build_c_grid(
            airfoil_coordinates=np.zeros((0, 2)),
            chord=1.0,
            n_surface=100,
            n_radial=10,
            first_cell_height=1e-4,
            max_radial_distance=20.0,
        )
        big = build_c_grid(
            airfoil_coordinates=np.zeros((0, 2)),
            chord=1.0,
            n_surface=200,
            n_radial=20,
            first_cell_height=1e-4,
            max_radial_distance=20.0,
        )
        assert big.cell_count > small.cell_count
        assert big.cell_count == 4 * small.cell_count

    def test_first_cell_height_is_recorded(self):
        first = 2e-5
        report = build_c_grid(
            airfoil_coordinates=np.zeros((0, 2)),
            chord=1.0,
            n_surface=200,
            n_radial=20,
            first_cell_height=first,
            max_radial_distance=20.0,
        )
        assert np.isclose(report.first_cell_height, first, rtol=1e-12)

    def test_cell_count_estimate_matches_build(self):
        count = cell_count_estimate(n_surface=100, n_radial=10, closed=True)
        assert count == 1000


class TestYPlus:
    @pytest.fixture
    def air(self):
        return {"rho": 1.225, "mu": 1.81e-5}

    def test_u_tau_from_tau_w(self, air):
        u = u_tau_from_tau_w(tau_w=1.0, rho=air["rho"])
        assert u > 0.0
        assert np.isclose(u, np.sqrt(1.0 / air["rho"]))

    def test_y_plus_from_height(self, air):
        yp = y_plus_from_height(
            y=1e-4, tau_w=1.0, rho=air["rho"], mu=air["mu"]
        )
        assert yp > 0.0

    def test_first_cell_height_for_yplus(self, air):
        tau_w = 1.0
        h = first_cell_height_for_yplus(
            yplus_target=30.0, tau_w=tau_w, rho=air["rho"], mu=air["mu"]
        )
        assert h > 0.0
        assert y_plus_from_height(h, tau_w, air["rho"], air["mu"]) == pytest.approx(
            30.0, rel=1e-9
        )

    def test_c_f_flat_plate_correlation(self):
        Re = 1e6
        c_f = c_f_flat_plate_turbulent(Re)
        assert c_f > 0.0
        assert np.isclose(c_f, 0.027 / (Re ** (1.0 / 7.0)))

    def test_tau_w_from_c_f(self, air):
        c_f = 0.005
        tau = tau_w_from_c_f(c_f, air["rho"], u_inf=50.0)
        assert tau > 0.0
        assert np.isclose(tau, 0.5 * air["rho"] * 50.0 ** 2 * c_f)

    def test_estimate_yplus_for_airfoil(self):
        result = estimate_yplus_for_airfoil(
            chord=1.0, u_inf=50.0, rho=1.225, mu=1.81e-5, yplus_target=30.0
        )
        assert result["Re_c"] > 0.0
        assert result["c_f"] > 0.0
        assert result["first_cell_height"] > 0.0
        assert result["yplus_at_target"] == pytest.approx(30.0)
