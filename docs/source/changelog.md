# Changelog

All notable changes to **sdypy-model** are documented here. The format is based
on [Keep a Changelog](https://keepachangelog.com/), and the project follows
[semantic versioning](https://semver.org/).

## Unreleased

### Added
- **Acoustic radiation via the boundary element method** — new
  `acoustic_external` subpackage exposing
  {class}`~sdypy.model.acoustic_external.AcousticExternalProblem`, a high-level
  driver for the exterior Helmholtz problem (`sd.model.AcousticExternalProblem`).
  Supports Neumann (normal-velocity) and Dirichlet (pressure) boundary
  conditions, continuous and discontinuous P1 collocation assemblers, the
  Burton–Miller formulation for spurious-resonance suppression, and field
  evaluation at arbitrary points. See the [Acoustic radiation tutorial](tutorials/acoustic_external).
- 3-D mesh helpers under `sdypy.model.mesh` supporting the BEM workflow.
- Test suite for the acoustic subpackage
  (`tests/test_acoustic_external_problem.py`) and a pulsating-sphere example
  (`examples/acoustic_external_example.py`) validated against the analytical
  monopole solution.
- Documentation: Getting Started, per-element tutorials (Beam, Shell,
  Tetrahedron, Acoustic BEM), whole-package API reference, and this changelog.

### Changed
- `AcousticExternalProblem.solve_problem` now returns the boundary solution
  `(phi, q)`; `evaluate_field` can reuse a precomputed solution to avoid
  re-solving across field evaluations.
- The `tqdm` progress bar auto-detects the runtime environment
  (notebook vs. terminal).

### Fixed
- Selecting Timoshenko beam theory (`"Timoshenko"` → `"T"`).
- Circular-import bug in the acoustic subpackage.
- Namespace-package resolution: removed a stray `__init__.py` from the top-level
  `sdypy/` folder that broke PEP 420 namespace resolution.

## [0.1.4] — 2026-01-28

### Changed
- Release bookkeeping and version sync.

## [0.1.3] — 2026-01-28

### Added
- `mode` parameter on the `Beam` constructor (Euler–Bernoulli / Timoshenko
  selection).

### Changed
- Updated examples.

### Removed
- `pyqt5` dropped from the required dependencies (only needed for interactive
  plotting in the examples).

## [0.1.2] — 2025-06-02

### Changed
- Version bump and metadata fixes.

## [0.1.1] — 2025-06-02

### Added
- Read the Docs configuration and documentation requirements.

## [0.1.0] — 2025-05-09

Initial public release.

### Added
- FEM elements: `Beam` (Euler–Bernoulli / Timoshenko), `Shell` (MITC4), and
  `Tetrahedron` (10-node `tet10`).
- General eigenvalue solver shared across the element types.
- 2-D quadrilateral semi-structured mesh generation.
- Packaging as the `sdypy` namespace package, GitHub Actions CI, and PyPI
  publishing.

[0.1.4]: https://github.com/sdypy/sdypy-model/releases/tag/v0.1.4
[0.1.3]: https://github.com/sdypy/sdypy-model/releases/tag/v0.1.3
[0.1.2]: https://github.com/sdypy/sdypy-model/releases/tag/v0.1.2
[0.1.1]: https://github.com/sdypy/sdypy-model/releases/tag/v0.1.1
[0.1.0]: https://github.com/sdypy/sdypy-model/releases/tag/v0.1.0
