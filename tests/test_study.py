"""Tests for the mesh-independence study."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from cfd_mesh_independence.study import StudySpec, run_study, write_results_report


@pytest.fixture
def spec() -> StudySpec:
    return StudySpec()


def test_run_study_returns_dataframe(spec: StudySpec):
    df = run_study(spec, angles=[0.0], surface_counts=[200], radial_counts=[20])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1


def test_run_study_has_required_columns(spec: StudySpec):
    df = run_study(spec, angles=[0.0], surface_counts=[200, 300], radial_counts=[20, 30])
    expected = {
        "angle_of_attack",
        "n_surface",
        "n_radial",
        "first_cell_height",
        "far_field_radius",
        "cell_count",
        "Cl",
        "Cd",
        "yplus",
        "c_f",
        "mesh_independent",
    }
    assert set(df.columns) == expected


def test_cd_is_finite(spec: StudySpec):
    df = run_study(spec, angles=[0.0, 5.0], surface_counts=[200, 400], radial_counts=[20, 40])
    assert np.isfinite(df["Cd"]).all()
    assert np.isfinite(df["Cl"]).all()
    assert np.isfinite(df["yplus"]).all()


def test_cd_matches_teaching_model(spec: StudySpec):
    df = run_study(spec, angles=[0.0, 5.0], surface_counts=[200], radial_counts=[20])
    row0 = df[df["angle_of_attack"] == 0.0].iloc[0]
    row5 = df[df["angle_of_attack"] == 5.0].iloc[0]
    assert row0["Cd"] > 0.0
    assert row5["Cl"] > 0.0
    # Thin-airfoil Cl at 5 degrees should be positive and roughly 2 pi alpha.
    expected_cl = 2.0 * np.pi * np.radians(5.0)
    assert np.isclose(row5["Cl"], expected_cl, rtol=1e-9)


def test_mesh_independent_flags_reasonable(spec: StudySpec):
    df = run_study(
        spec,
        angles=[0.0],
        surface_counts=[200, 300, 400],
        radial_counts=[20, 30, 40],
    )
    assert df["mesh_independent"].dtype == bool
    # The first row is never converged (no previous result). Subsequent rows
    # should be converged because the teaching Cd is constant.
    # The series accessors yield numpy booleans, so compare through bool().
    assert bool(df["mesh_independent"].iloc[0]) is False
    assert bool(df["mesh_independent"].iloc[1]) is True
    assert bool(df["mesh_independent"].iloc[2]) is True


def test_yplus_is_constant_when_first_cell_height_fixed(spec: StudySpec):
    df = run_study(
        spec,
        angles=[0.0],
        surface_counts=[200, 400, 600],
        radial_counts=[20, 40, 60],
    )
    assert df["yplus"].std() == pytest.approx(0.0, abs=1e-12)
    assert df["first_cell_height"].std() == pytest.approx(0.0, abs=1e-12)


def test_outputs_directory_is_created(spec: StudySpec, tmp_path: pathlib.Path):
    out = tmp_path / "out"
    df = run_study(
        spec,
        angles=[0.0],
        surface_counts=[200],
        radial_counts=[20],
        out_dir=str(out),
        csv=True,
        plot=True,
    )
    assert (out / "mesh_convergence.csv").exists()
    assert (out / "mesh_convergence.png").exists()
    loaded = pd.read_csv(out / "mesh_convergence.csv")
    assert len(loaded) == len(df)


def test_write_results_report_writes_file(spec: StudySpec, tmp_path: pathlib.Path):
    df = run_study(
        spec,
        angles=[0.0],
        surface_counts=[200],
        radial_counts=[20],
        out_dir=str(tmp_path / "study_out"),
        csv=False,
        plot=False,
    )
    report_path = tmp_path / "RESULTS.md"
    write_results_report(df, spec, report_path)
    assert report_path.exists()
    text = report_path.read_text(encoding="utf-8")
    assert "# Mesh-independence study — results" in text
    assert "| alpha (°) |" in text
