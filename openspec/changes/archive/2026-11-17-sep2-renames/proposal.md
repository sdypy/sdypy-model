## Why

The sdypy hub's nomenclature checker (`tools/check_nomenclature.py`) reports
57 public names in sdypy-model that diverge from SEP 2, which keeps the
umbrella's `test_nomenclature` gate red for this package (issue
sdypy/sdypy-model#8).

Non-trivial: it renames public names. The contract is the hub's
`public-api` requirement *Evidenced divergences carry deprecated aliases to
the canonical names*; this change has no spec delta of its own
(`skip_specs`) and carries the tasks.

## What Changes

Names fixed by the SEP 2 table:

- `E`, `Young` → `young_modulus`; `nu`, `Poisson` → `poisson_ratio`;
  `rho`, `ro`, `Density` → `density`.
- `org`, `conec` → `nodes`, `elements`; `M`, `K` → `mass_matrix`,
  `stiffness_matrix`; `nat_freq` → `natural_freq`; `frequency` → `freq`;
  `Beam.solve(n=)` → `n_modes`; the eigenvector attribute `A` → `mode_shape`.

Names the table does not cover, chosen here (maintainer decision):

- Beam: `I` → `area_moment`; `A` (cross-section area argument) → `area`;
  `EI` → `bending_stiffness`.
- Tetrahedron: `J` → `jacobian`; `J_det` → `jacobian_det`;
  `derivative_E_ind` → `derivative_young_modulus_idx`;
  `derivative_ro_ind` → `derivative_density_idx`; `eig_ind` → `mode_idx`.
- `Mesh3D.transform`: `R` → `rotation_matrix`.
- Acoustic: `N` → `shape_functions`; `G` → `greens_function`;
  `gauss_legendre_1d(n=)` → `n_points`.

Function and class names do not change, and `__all__` is untouched. Every old
name stays as a deprecated alias: passing an old keyword, or reading, writing
or deleting an old attribute, emits `DeprecationWarning` and acts on the new
name. Positional calls are unchanged. Passing an old and a new keyword
together raises `TypeError`. Aliases are kept through v1.x.

The package, tests, README, tutorials and examples use the new names.
`Tetrahedron.ro`, which was a silent property, now warns like the others.

## Impact

- `sdypy/model/beam/beam.py`, `shell/shell.py`, `tetrahedron/tet10.py`,
  `mesh/mesh_3D.py`, `eigenvalue_solution.py`, `acoustic_external/*`; new
  private module `sdypy/model/_deprecation.py`.
- Users see `DeprecationWarning` for old names; nothing breaks before v2.0.
- Hub follow-up: the `public-api` inventory (around lines 300-312) gains
  `I` → `area_moment`, `A` → `area` / `mode_shape`, `EI` →
  `bending_stiffness`, `J` → `jacobian`, `R` → `rotation_matrix`,
  `N` → `shape_functions`, `G` → `greens_function`, `n` → `n_points`,
  `frequency` → `freq`.
- Hub follow-up: `tests/test_interop.py` moves from `nat_freq` to
  `natural_freq` once this is on PyPI.
