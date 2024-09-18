import os
import sys
import numpy as np
import meshio
from pyvistaqt import BackgroundPlotter
import pyvista as pv
from scipy.sparse.linalg import eigsh

current_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)

from sdypy.model import Tetrahedron

def pad_facets(facets):
    return np.hstack((np.ones((facets.shape[0], 1))*facets.shape[1], facets)).astype(int)

# Load mesh
path_mesh = 'data/Beam - simple.msh'

mesh = meshio.read(path_mesh)

org = mesh.points
org_orig = np.copy(org)
conec = mesh.cells_dict['tetra10']

# Create triangles
inds = np.array([[2, 1, 0, 4, 5, 6],
                 [1, 3, 0, 6, 7, 8],
                 [3, 2, 0, 5, 7, 9],
                 [2, 3, 1, 4, 8, 9]])

triangles = []
for element in conec:
    for i in inds:
        triangles.append(element[i])
triangles = np.unique(triangles, axis=0)
triangles_1 = pad_facets(triangles[:, :3])

# Plot the mesh
plotter = BackgroundPlotter()
grid = pv.PolyData(org, triangles_1)
grid_points = pv.PolyData(org)

plotter.add_mesh(grid_points, render_points_as_spheres=True, opacity=1, color='blue', point_size=7)
plotter.add_mesh(grid, render_points_as_spheres=True, opacity=1, color='white', show_edges=True)
plotter.add_axes()


# Assemble matrices
parameters = {
    'Young': np.ones((conec.shape[0])) * 190000,
    'Density': np.ones((conec.shape[0])) * 7850e-12,
    'Poisson': 0.3
}

tet_obj = Tetrahedron(org, conec, parameters['Young'], parameters['Density'], parameters['Poisson'], 60)

# Solve the eigenvalue problem
eigenvalues, eigenvectors = eigsh(tet_obj.K, M=tet_obj.M, k=20, sigma=0, which='LM')

nat_freq = np.sqrt(eigenvalues) / (2 * np.pi)

# %%
# Plot the modes
n_plots = 6
rows = int(np.floor(np.sqrt(n_plots)))
cols = int(n_plots // rows)
shape = (rows, cols)

# Initialize a plotter with 6 subplots (2 rows, 3 columns)
plotter = BackgroundPlotter(shape=shape)

mode_start = 6
for ii in range(int(rows*cols)):
    mode_number = mode_start + ii
    modal_shape = eigenvectors[:, mode_number].reshape(-1, 3)


    # Scale the modal shape for better visualization
    # scale_factor = 1e-2  # Adjust this as needed
    scale_factor = np.linalg.norm(org)/np.linalg.norm(modal_shape)/10
    displaced_nodes = org + scale_factor * modal_shape

    # Compute the norm of the displacements for coloring
    displacement_norm = np.linalg.norm(modal_shape, axis=1)

    # Select the subplot to work on
    plotter.subplot(ii // cols, ii % cols)

    # Create a mesh for the current mode
    mesh = pv.PolyData(displaced_nodes)

    triangles = []
    for element in conec:
        for i in inds:
            triangles.append(element[i])
    mesh.faces = pad_facets(np.array(triangles)[:, :3])

    # Add displacement norm as point data for coloring
    displacement_norm = np.linalg.norm(modal_shape, axis=1)  # Assuming displacements is a 3D array
    mesh["Displacement"] = displacement_norm

    # Add the mesh to the plotter
    plotter.add_mesh(mesh, show_edges=True, scalars="Displacement", cmap="viridis", edge_color='black')
    plotter.add_text(f"Mode {mode_number} ({np.sqrt(eigenvalues[mode_number])/(2*np.pi):.2f} Hz)", font_size=12)
    plotter.show_grid()

# Show the plotter with all subplots
plotter.show()
# %%
