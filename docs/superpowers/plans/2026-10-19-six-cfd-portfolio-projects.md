Status: implementation notes
=================

The src-layout package directory (`src/cfd_mesh_independence/`) is the
importable package; the flat `src/*.py` layout that preceded it could not be
imported at all, and the `ModuleNotFoundError: No module named '__future__'`
seen while debugging was an artifact of scratch scripts that overwrote
`sys.path` with only the `src` directory, removing the stdlib from the path.

Geometry layout (this is the contract the tests encode): the closed polyline
starts at the leading edge, runs the upper surface to the trailing edge, then
the lower surface back to x = 0, so both the first and last points lie on the
trailing-edge line at x = 1. Chordwise points use quadratic spacing
(`x = i / (n - 1)` squared) which clusters points at the leading edge. The
upper half runs x = 0 -> 1 and the lower half runs x = 1 -> 0 back to the
leading edge, so the first point is the LE at (0, 0) and the last point sits
on the TE line x = 1. Each surface is differentiated on its own monotonic x
array, because a concatenated upper+lower array reverses direction at the
trailing edge and `np.gradient` then divides by a zero spacing, producing
NaNs.

Mesh cell count (`build_c_grid`): the circumferential count is the surface
point count for a closed surface and `n_surface - 1` only for an open one, so
a 200x20 closed mesh reports 4000 cells and `cell_count_estimate` agrees.
Radial spacing solves for the growth ratio above 1 (cells grow away from the
wall) by bisecting the geometric-series closed form, then rescales so the
outermost node lands exactly on `max_radial_distance`. A degenerate (empty)
airfoil polyline is treated as a placeholder of zeros rather than an error,
because the study driver only needs the counts.
