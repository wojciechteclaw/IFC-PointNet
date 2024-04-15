import trimesh
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import sys
import logging


def obj_to_xyz(file_path, point_count=2048, output_directory=False):

    if file_path.lower().endswith(".obj"):
        # Load the OBJ file
        mesh = trimesh.load_mesh(file_path)
        # TODO obróć do pionu
        mesh, scale_factor = normalize_geometry(mesh, min=0.0, max=1.0)

        # Sample points on the mesh surface
        points = mesh.sample(point_count)
        # Define the path for the output XYZ file
        output_file_name = f"{Path(file_path).stem}_x{scale_factor:.2f}.xyz"
        rootdir = Path(file_path).parent.parent
        subdir = os.path.basename(Path(file_path).parent)
        if output_directory:
            rootdir = output_directory
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
        new_path = os.path.join(rootdir, subdir)
        # Save points to a .XYZ file
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        np.savetxt(
            os.path.join(new_path, output_file_name),
            points,
            fmt="%f",
            delimiter=" ",
        )
        logging.debug(f"Saved file {new_path}")


def normalize_geometry(mesh, min=0.0, max=1.0):
    # Calculate the bounding box's size in each dimensions
    dimensions = mesh.bounds[1] - mesh.bounds[0]
    # Find the maximum dimension
    max_dimension = dimensions.max()
    # Calculate the scale factor
    scale_factor = 1.0 / max_dimension
    # Translate the mesh to the origin
    mesh.apply_translation(-mesh.bounds[0])
    # Scale the mesh uniformly to fit into the unit cube
    mesh.apply_scale(scale_factor)
    # matrix = np.eye(4)
    # matrix[:3, 3] = [mesh.bounds[1][i] + mesh.bounds[0][i] for i in range(3)]
    # mesh.apply_transform(matrix)
    return mesh, 1 / scale_factor


if __name__ == "__main__":
    FOLDER_PATH = sys.argv[0]
    POINT_COUNT = sys.argv[1]
    OUTPUT_PATH = sys.argv[2]

    for subdir, dirs, files in os.walk(FOLDER_PATH):
        for file in tqdm(files, desc=f"Processing {subdir}"):
            obj_to_xyz(
                os.path.join(subdir, file),
                point_count=POINT_COUNT,
                output_directory=OUTPUT_PATH,
            )

    logging.info("All OBJ files have been processed.")
