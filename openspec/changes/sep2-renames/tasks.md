## 1. Renames with deprecated aliases

- [x] 1.1 `_deprecation.py`: `DeprecatedAttribute` (class-level alias, warns on read, write and delete), `reject_both` and `require`
- [x] 1.2 `Beam`, `matrices_k_e_timoshenko`, `matrices_k_e`: `nodes`, `elements`, `area_moment`, `area`, `bending_stiffness`, `mass_matrix`, `stiffness_matrix`, `natural_freq`, `mode_shape`, `solve(n_modes=)`; old names as aliases
- [x] 1.3 `Tetrahedron` and `tet10` helpers: `jacobian`, `jacobian_det`, `derivative_young_modulus_idx`, `derivative_density_idx`, `mode_idx`, `young_modulus`, `density`, `poisson_ratio`, `nodes`, `elements`; old names as aliases
- [x] 1.4 `Shell`, `get_constitutive_tensor`, `lump_mass_matrix`, `solve_eigenvalue`: SEP 2 names, old names as aliases
- [x] 1.5 `Mesh3D.transform(rotation_matrix=)`, acoustic `shape_functions`, `greens_function`, `n_points`, `density`, `freq`; old names as aliases
- [x] 1.6 Package internals, tests, README, tutorials and examples use the new names; changelog entry

## 2. Verification

- [x] 2.1 `tests/test_sep2_renames.py`: every old name warns and equals the new one; new names and positional calls do not warn; old and new together raise; missing required arguments raise
- [x] 2.2 `pytest` with `-W error::DeprecationWarning:sdypy` (no internal use of an old name): 128 passed, 2 skipped
- [x] 2.3 Hub checkers against this clone: `check_nomenclature.py` and `check_public_api.py` pass
- [ ] 2.4 CI green on the PR

## 3. After merge

- [ ] 3.1 Release to PyPI
- [ ] 3.2 Hub: `tests/test_interop.py` uses `natural_freq`; `public-api` inventory gains the names listed in the proposal
