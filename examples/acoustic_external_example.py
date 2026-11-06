"""
Test the AcousticExternalProblem class with a pulsating sphere.

Compares the BEM solution against the analytical solution for a sphere
with uniform radial velocity (monopole-like radiation).

The analytical surface pressure for a pulsating sphere of radius `a`
with uniform normal velocity v0 is:

    p_surface = (rho0 * c0 * v0) * (j * k * a) / (1 + j * k * a)

and the field pressure at distance r >= a is:

    p(r) = p_surface * (a / r) * exp(-j * k * (r - a))
"""

import os
import numpy as np
import pyvista as pv
import matplotlib
matplotlib.use("Agg")  # headless: write figures to file, no display needed
import matplotlib.pyplot as plt

from sdypy.model.acoustic_external import AcousticExternalProblem


def evaluate_field_batched(prob, pts, chunk=4000):
    """Evaluate the velocity potential in chunks to keep peak memory bounded.

    ``evaluate_field`` builds a dense (n_points x n_dof x n_quad) intermediate,
    so a large grid (e.g. 200x200) can need many GB at once. Splitting the
    evaluation points into chunks caps the peak memory at the chunk size.
    Returns the potential ``phi``; convert to pressure with ``p = j ω ρ φ``.
    """
    out = np.empty(pts.shape[0], dtype=complex)
    for i in range(0, pts.shape[0], chunk):
        out[i:i + chunk] = prob.evaluate_field(
            pts[i:i + chunk], verbose=False
        )
    return out


# =============================================================================
# PARAMETERS (SI units)
# =============================================================================
RHO = 1.225       # medium density  [kg/m3]
C0  = 343.0       # speed of sound  [m/s]
FREQ = 500.0      # frequency       [Hz]
V0  = 0.01        # normal velocity [m/s]

# Sphere radius [m]
A = 0.15

# BEM solver settings
ASSEMBLER_TYPE = "continuous"
USE_BM = False

print("=" * 60)
print("Pulsating sphere -- BEM test")
print("=" * 60)
print(f"  radius a    = {A*1e3:.1f} mm")
print(f"  frequency   = {FREQ:.0f} Hz")
print(f"  wavelength  = {C0/FREQ*1e3:.0f} mm")
print(f"  ka          = {2*np.pi*FREQ/C0 * A:.3f}")
print(f"  normal vel. = {V0*1e3:.2f} mm/s")
print(f"  assembler   = {ASSEMBLER_TYPE}")
print(f"  Burton-Miller = {USE_BM}")
print()

# =============================================================================
# MESH -- sphere from PyVista
# =============================================================================
sphere = pv.Sphere(radius=A, theta_resolution=30, phi_resolution=30)
n_pts = sphere.n_points
n_cells = sphere.n_cells

sphere.compute_normals(point_normals=True, cell_normals=False,
                       consistent_normals=True, auto_orient_normals=True,
                       inplace=True)

print(f"Mesh: {n_pts} nodes, {n_cells} triangles")
print(f"      mean edge ~ {2*np.pi*A / 30 * 1e3:.1f} mm")

# =============================================================================
# SET UP PROBLEM  (Neumann BC = uniform outward normal velocity)
# =============================================================================
vn = V0 * np.ones(n_pts, dtype=np.float64)

prob = AcousticExternalProblem(
    mesh=sphere,
    rho=RHO,
    c0=C0,
    boundary_condition=vn,
    boundary_condition_type="Neumann",
    frequency=FREQ,
    assembler_type=ASSEMBLER_TYPE,
    use_burton_miller=USE_BM,
    quad_order=3,
    near_threshold=2.0,
)

print("\nAssembling matrices and solving ...")
phi, q = prob.solve_problem(verbose=True)

# Surface pressure from BEM potential:  p = j ω ρ φ
p_bem = 1j * 2 * np.pi * FREQ * RHO * phi
p_max = np.max(np.abs(p_bem))
print(f"\n  max |p| on boundary (BEM) = {p_max:.4e} Pa")

# Analytical surface pressure
k  = 2 * np.pi * FREQ / C0
omega = 2 * np.pi * FREQ
ka = k * A
p_surface_analytical = (RHO * C0 * V0) * (1j * ka) / (1 + 1j * ka)
p_surface_bem_mean = np.mean(p_bem)
print(f"  Analytical p_surface       = {np.abs(p_surface_analytical):.4e} Pa")
print(f"  BEM mean p_surface         = {np.abs(p_surface_bem_mean):.4e} Pa")
print(f"  Relative error             = "
      f"{abs(np.abs(p_surface_bem_mean) - np.abs(p_surface_analytical)) / np.abs(p_surface_analytical) * 100:.2f} %")


# =============================================================================
# FIELD EVALUATION -- XZ plane (y = 0)
# =============================================================================
extent = 0.6                     # +/-0.6 m
n_grid = 200                     # points per side
x  = np.linspace(-extent, extent, n_grid)
z  = np.linspace(-extent, extent, n_grid)
XX, ZZ = np.meshgrid(x, z, indexing="ij")
YY = np.zeros_like(XX)
field_pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])

print(f"\nEvaluating pressure field on {field_pts.shape[0]} points ...")
phi_field = evaluate_field_batched(prob, field_pts)       # velocity potential
p_field = 1j * omega * RHO * phi_field                    # pressure  p = j ω ρ φ
p_field = p_field.reshape(XX.shape)

# Analytical field pressure (exclude points inside sphere)
p_analytical = np.zeros_like(p_field, dtype=complex)
r_grid = np.sqrt(XX**2 + YY**2 + ZZ**2)
mask_out = r_grid >= A
p_analytical[mask_out] = (
    p_surface_analytical * (A / r_grid[mask_out])
    * np.exp(-1j * k * (r_grid[mask_out] - A))
)

# Compute error on evaluation grid (outside sphere)
error = np.abs(np.abs(p_field) - np.abs(p_analytical))
rel_error = np.where(mask_out, error / (np.abs(p_analytical) + 1e-30), 0.0)
rms_error = np.sqrt(np.mean(error[mask_out]**2))
rms_ref   = np.sqrt(np.mean(np.abs(p_analytical[mask_out])**2))
rms_rel   = rms_error / rms_ref if rms_ref > 0 else 0.0
print(f"  RMS error (outside sphere) = {rms_error:.4e} Pa")
print(f"  RMS relative error         = {rms_rel*100:.2f} %")


# =============================================================================
# VISUALISATION
# =============================================================================
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
fig.suptitle(f"Pulsating sphere  (f = {FREQ:.0f} Hz,  a = {A*1e3:.0f} mm,  "
             f"ka = {ka:.3f})", fontsize=13)

# BEM SPL magnitude
im0 = axes[0,0].pcolormesh(XX*1e3, ZZ*1e3, 20*np.log10(np.abs(p_field) / 2e-5),
                           shading="auto", cmap="magma")
axes[0,0].set_title("BEM -- SPL [dB]")
axes[0,0].set_xlabel("x [mm]"); axes[0,0].set_ylabel("z [mm]")
fig.colorbar(im0, ax=axes[0,0])

# BEM real part
im1 = axes[0,1].pcolormesh(XX*1e3, ZZ*1e3, np.real(p_field),
                           shading="auto", cmap="RdBu_r",
                           vmin=-np.max(np.abs(np.real(p_field))),
                           vmax= np.max(np.abs(np.real(p_field))))
axes[0,1].set_title("BEM -- Re(p) [Pa]")
axes[0,1].set_xlabel("x [mm]"); axes[0,1].set_ylabel("z [mm]")
fig.colorbar(im1, ax=axes[0,1])

# Cross-section along z = 0 (x-axis)
r_line = np.linspace(A, extent, 200)
line_pts = np.column_stack([r_line, np.zeros_like(r_line), np.zeros_like(r_line)])
p_line_bem = 1j * omega * RHO * prob.evaluate_field(line_pts, verbose=False)
p_line_ana = p_surface_analytical * (A / r_line) * np.exp(-1j * k * (r_line - A))

ax2 = axes[1,0]
ax2.semilogy(r_line*1e3, np.abs(p_line_bem), "b-", label="BEM", lw=2)
ax2.semilogy(r_line*1e3, np.abs(p_line_ana), "r--", label="Analytical", lw=1.5)
ax2.set_xlabel("x [mm]"); ax2.set_ylabel("|p| [Pa]")
ax2.set_title("Field along x-axis (z = 0)")
ax2.set_ylim(min(np.hstack((np.abs(p_line_bem),np.abs(p_line_ana))))/10, max(np.hstack((np.abs(p_line_bem),np.abs(p_line_ana))))*10)
ax2.legend(); ax2.grid(True, alpha=0.3)

# Relative error map
im3 = axes[1,1].pcolormesh(XX*1e3, ZZ*1e3,
                           np.where(mask_out, rel_error * 100, np.nan),
                           shading="auto", cmap="hot", vmin=0, vmax=40)
axes[1,1].set_title("Relative error [%]")
axes[1,1].set_xlabel("x [mm]"); axes[1,1].set_ylabel("z [mm]")
fig.colorbar(im3, ax=axes[1,1])

# Draw sphere outline on all subplots
for ax in axes.ravel():
    circle = plt.Circle((0, 0), A*1e3, fill=False, color="white", ls="--",
                        lw=0.8, alpha=0.6)
    ax.add_patch(circle)

plt.tight_layout()

# Save
out_dir = os.path.dirname(__file__)
save_path = os.path.join(out_dir, "pulsating_sphere_test_B-M.png")
plt.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"\nFigure saved to: {save_path}")

# Also render the mesh and BEM boundary pressure with PyVista (headless screenshot)
print("\nRendering 3-D mesh with boundary pressure ...")
sphere_mesh = sphere.copy()
sphere_mesh["p_bem"] = np.abs(p_bem)

pv.set_plot_theme("document")
pl = pv.Plotter(off_screen=True)
pl.add_mesh(sphere_mesh, scalars="p_bem", cmap="magma",
            show_edges=True, edge_color="gray", smooth_shading=True)
pl.add_title(f"Boundary pressure |p|  (f = {FREQ:.0f} Hz)", font_size=12)
mesh_path = os.path.join(out_dir, "pulsating_sphere_boundary_pressure.png")
pl.screenshot(mesh_path)
pl.close()
print(f"3-D render saved to: {mesh_path}")

print("\nDone.")
