"""
Based on https://blenderbim.org/docs-python/ifcopenshell-python/geometry_processing.html
"""

import os
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.shape
import logging
import numpy as np
from datetime import datetime

now = datetime.now()
logging.basicConfig(level=logging.INFO, filename=f'logs\\Log_{now.strftime("%Y.%m.%d_%H.%M.%S")}.log', filemode='w', format='%(asctime)s %(levelname)s - %(message)s')


def get_geometry(element):
    try: 
        shape = ifcopenshell.geom.create_shape(settings, element)

        if shape:
            # X Y Z of vertices in flattened list e.g. [v1x, v1y, v1z, v2x, v2y, v2z, ...]
            verts = shape.geometry.verts
            # Indices of vertices per triangle face e.g. [f1v1, f1v2, f1v3, f2v1, f2v2, f2v3, ...]. Note that faces are always triangles.
            faces = shape.geometry.faces
            #spit into sublists:
            faces = [faces[i * 3:(i + 1) * 3] for i in range((len(faces) + 2) // 3 )]
            # A 4x4 matrix representing the location and rotation of the element, in the form:
            # [ [ x_x, y_x, z_x, x   ]
            #   [ x_y, y_y, z_y, y   ]
            #   [ x_z, y_z, z_z, z   ]
            #   [ 0.0, 0.0, 0.0, 1.0 ] ]
            # The position is given by the last column: (x, y, z)
            # The rotation is described by the first three columns, by explicitly specifying the local X, Y, Z axes.
            # The first column is a normalised vector of the local X axis: (x_x, x_y, x_z)
            # The second column is a normalised vector of the local Y axis: (y_x, y_y, y_z)
            # The third column is a normalised vector of the local Z axis: (z_x, z_y, z_z)
            # The axes follow a right-handed coordinate system.
            # Objects are never scaled, so the scale factor of the matrix is always 1.
            matrix = shape.transformation.matrix.data
            matrix = ifcopenshell.util.shape.get_shape_matrix(shape)
            # Extract the XYZ location and rotation of the matrix:
            location = np.array2string(matrix[:,3][0:3], separator=',', precision=3, suppress_small=True).replace(' ', '').replace('\n', '')
            rotation = np.array2string(matrix[:3, :3], separator=',', precision=3, suppress_small=True).replace(' ', '').replace('\n', '')
            return True, verts, faces, shape.guid, shape.geometry.id, location, rotation
            # shape.geometry.id is an unique geometry ID, useful to check whether or not two geometries are identical for caching and reuse. 
        else:
            return False, False, False, False, False, False, False
    except (AttributeError, RuntimeError):
        return False, False, False, False, False, False, False


SOURCE = r"C:\Code\IFCextract\tests\3_source_models"
DESTINATION = r"C:\Code\IFCextract\tests\5_output"

files = os.listdir(SOURCE)

x = len(os.listdir(DESTINATION))+1
j = 1
for f in files:
    try:
        ifc_file = ifcopenshell.open(os.path.join(SOURCE,f))
        settings = ifcopenshell.geom.settings()
        file_name = f.split('\\')[-1]
        print(f"\n\n########### FILE {file_name} #############\n\n")
        logging.info(f"Starting a new file: {file_name}")
        k = 1
        c = sum(1 for _ in ifc_file)
        unique_geometry = set()
        for elem in ifc_file:
            success, vertices, faces, shape_guid, geometry_id, location, rotation = get_geometry(elem)
            if success:
                if geometry_id not in unique_geometry:
                    with open(f"{DESTINATION}\\{x:06}_{elem.is_a()}.obj", 'w+', encoding="utf-8") as new_file:
                        new_file.write(f"# Source file: '{f}'\n")
                        new_file.write(f"# GlobalId: '{elem.GlobalId}'\n")
                        new_file.write(f"# entity: '{elem.is_a()}'\n")
                        new_file.write(f"# Name: {elem.Name}\n")
                        new_file.write(f"# ObjectType: {elem.ObjectType}\n")
                        new_file.write(f"# location: {location}\n")
                        new_file.write(f"# rotation: {rotation}\n")
                        # new_file.write("newmtl Colored\n")
                        new_file.write(f"o {elem.is_a()}\n")  # {elem.Name} - {elem.GlobalId}
                        for i in range(0, len(vertices), 3):
                            new_file.write(f"v {str(vertices[i])} {str(vertices[i+1])} {str(vertices[i+2])}\n")
                        # new_file.write("usemtl Colored\n")
                        # new_file.write("s off\n")
                        for f3 in faces:
                            new_file.write(f"f {str(f3[0]+1)} {str(f3[1]+1)} {str(f3[2]+1)}\n")
                        print(f"Completed: {k}/{c} from file: {j} of {len(files)}")
                        logging.info(f"Object {k}/{c} saved as {x:05}_{elem.is_a()}_({file_name})")
                    unique_geometry.add(geometry_id)
                else:
                    logging.warning("The same geometry already logged previously...")
                x+=1
            else:
                logging.debug(f"Element: {elem.is_a()} does not have vertices.")
            k+=1
        j+=1
    except ifcopenshell.SchemaError:
        logging.warning(f"File {f} is not a proper IFC file.")