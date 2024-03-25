import trimesh
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import sys


FOLDER_PATH = sys.argv[0]
POINT_COUNT = 2048
output_dir = FOLDER_PATH + "_XYZ"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

for subdir, dirs, files in os.walk(FOLDER_PATH):
    for file in tqdm(files, desc=f"Processing {subdir}"):
        # Check if the file is an OBJ file
        if file.lower().endswith(".obj"):
            file_path = os.path.join(subdir, file)
            # Load the OBJ file
            mesh = trimesh.load_mesh(file_path)
            # Sample points on the mesh surface
            points = mesh.sample(POINT_COUNT)
            # Define the path for the output XYZ file
            output_file_name = Path(file_path).stem + ".xyz"
            # Save points to a .XYZ file
            new_path = str(Path(file_path).parent).replace(FOLDER_PATH, output_dir)
            if not os.path.exists(new_path):
                os.makedirs(new_path)
            np.savetxt(
                os.path.join(new_path, output_file_name),
                points,
                fmt="%f",
                delimiter=" ",
            )

            # print(f"Processed {file_path} and saved to {output_file_path}")

print("All OBJ files have been processed.")
