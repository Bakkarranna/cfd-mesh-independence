"""Command-line entrypoint for the mesh-independence study."""

from __future__ import annotations

from .study import StudySpec, run_study


def main() -> None:
    spec = StudySpec()
    df = run_study(
        spec,
        angles=(0.0, 5.0),
        surface_counts=(200, 300, 400, 600, 800, 1000),
        radial_counts=(20, 30, 40, 60, 80, 100),
    )
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
