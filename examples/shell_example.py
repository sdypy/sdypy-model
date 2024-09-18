import os
import sys
import pickle
import meshio
from pyvistaqt import BackgroundPlotter
import pyvista as pv
import numpy as np
from scipy.sparse.linalg import eigsh

current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

from sdypy.model import Shell


# Load the mesh
mesh = meshio.read("data/L_bracket.msh")
# extract nodes and elements from mesh
nodes = mesh.points / 1000
elements = []
for cells in mesh.cells:
    if cells.type == "quad":
        elements.append(cells.data)
elements = np.vstack(elements)

#  Properties
thickness = 0.001
# E = 210e9
E = 2.069e11
nu = 0.3
rho = 7829

# Compute the stiffness and mass matrices
mass_lumping = False
force_nonsingularity = True

shell_obj = Shell(nodes, elements, E, nu, rho, thickness, verbose=1, mass_lumping=mass_lumping, force_nonsingularity=force_nonsingularity)
K = shell_obj.K
M = shell_obj.M

# Compute the first 20 modes
eigenvalues, eigenvectors = eigsh(K, M=M, k=20, sigma=0, which='LM')

# Plot the modes
n_plots = 6
rows = int(np.floor(np.sqrt(n_plots)))
cols = int(n_plots // rows)
shape = (rows, cols)

# Initialize a plotter with 6 subplots (2 rows, 3 columns)
plotter = BackgroundPlotter(shape=shape)

mode_start = 3
for ii in range(int(rows*cols)):
    mode_number = mode_start + ii
    modal_shape = eigenvectors[:, mode_number]

    # Extract translational components (ux, uy, uz) from modal shape
    num_nodes = nodes.shape[0]
    translational_modal_shape = np.zeros_like(nodes)

    for i in range(num_nodes):
        translational_modal_shape[i, 0] = modal_shape[i * 6]     # ux
        translational_modal_shape[i, 1] = modal_shape[i * 6 + 1] # uy
        translational_modal_shape[i, 2] = modal_shape[i * 6 + 2] # uz

    # Scale the modal shape for better visualization
    # scale_factor = 1e-2  # Adjust this as needed
    scale_factor = np.linalg.norm(nodes)/np.linalg.norm(modal_shape)
    displaced_nodes = nodes + scale_factor * translational_modal_shape

    # Compute the norm of the displacements for coloring
    displacement_norm = np.linalg.norm(translational_modal_shape, axis=1)

    # Select the subplot to work on
    plotter.subplot(ii // cols, ii % cols)

    # Create a mesh for the current mode
    mesh = pv.PolyData(displaced_nodes)

    # Add cells (elements) to the mesh
    faces = []
    for element in elements:
        face = np.hstack([[4], element])  # Add number of points per cell (4 for quadrilateral)
        faces.append(face)
    faces = np.hstack(faces).astype(np.int64)
    mesh.faces = faces

    # Add displacement norm as point data for coloring
    displacement_norm = np.linalg.norm(translational_modal_shape, axis=1)  # Assuming displacements is a 3D array
    mesh["Displacement"] = displacement_norm

    # Add the mesh to the plotter
    plotter.add_mesh(mesh, show_edges=True, scalars="Displacement", cmap="viridis", edge_color='black')
    plotter.add_text(f"Mode {mode_number} ({np.sqrt(eigenvalues[mode_number])/(2*np.pi):.2f} Hz)", font_size=12)
    plotter.show_grid()

# Show the plotter with all subplots
plotter.show()
input()