import os
import numpy as np
import pyvista as pv
from pyvistaqt import BackgroundPlotter

def generate_quadrilateral_mesh(path_to_step, element_size, path_to_mesh=None, show_mesh="None"):
    """Generate quadrilateral mesh (semi-structured).
    
    Parameters
    ----------
    path_to_step : str
        Path to the STEP file.
    element_size : float
        Desired element size in STEP units.
    path_to_mesh : str
        Save path for the mesh. If not given, the mesh will not be saved.
    show_mesh : bool
        If True, display the mesh.
    
    Returns
    -------
    nodes : np.ndarray
        Node coordinates.
    elements : np.ndarray
        Element connectivity.
    """
    import gmsh
    model_name = os.path.basename(path_to_step).split(".")[0]

    # Initialize GMSH
    gmsh.initialize()

    # Add a new model named 'imported_model'
    gmsh.model.add(model_name)

    # Import the STEP file
    gmsh.model.occ.importShapes(path_to_step)

    # Synchronize to ensure all imported geometries are available in the model
    gmsh.model.occ.synchronize()

    # Identify surfaces to be meshed
    # You can list all surfaces and then select the desired ones
    entities = gmsh.model.getEntities(dim=2)

    gmsh.model.mesh.field.add("MathEval", 1)
    gmsh.model.mesh.field.setString(1, "F", f"{element_size}")
    gmsh.model.mesh.field.setAsBackgroundMesh(1)

    # Set mesh options for 2D quadrilateral semi-structured mesh
    gmsh.option.setNumber("Mesh.Algorithm", 8)  # Use the transfinite algorithm for semi-structured meshes
    gmsh.option.setNumber("Mesh.RecombineAll", 1)  # Recombine into quadrilaterals

    # Optionally set transfinite meshing parameters (if you have specific preferences)
    # For each surface, define the transfinite algorithm parameters
    for entity in entities:
        gmsh.model.mesh.setTransfiniteSurface(entity[1])
        gmsh.model.mesh.setRecombine(2, entity[1])

    # Generate the 2D mesh
    gmsh.model.mesh.generate(2)

    # Save the mesh to a file
    if path_to_mesh:
        gmsh.write(path_to_mesh)

    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    element_types, element_tags, element_node_tags = gmsh.model.mesh.getElements()
    nodes = np.reshape(node_coords, (-1, 3))
    i = np.argmin(np.abs(element_types - 3)) # element type 3 represent quadrilaterals
    elements = np.reshape(element_node_tags[i], (-1, 4)) - 1

    if show_mesh == "gmsh":
        gmsh.fltk.run()
    elif show_mesh == "pyvista":
        mesh = pv.PolyData(nodes)
        # Add cells (elements) to the mesh
        faces = []
        for element in elements:
            face = np.hstack([[4], element])  # Add number of points per cell (4 for quadrilateral)
            faces.append(face)
        faces = np.hstack(faces).astype(np.int64)
        mesh.faces = faces

        # Visualize the mesh with displacement coloring
        plotter = BackgroundPlotter()
        plotter.add_mesh(mesh, show_edges=True, edge_color='black')
        plotter.show_grid()
        plotter.show()
    else:
        pass

    # Finalize GMSH
    gmsh.finalize()

    return nodes, elements