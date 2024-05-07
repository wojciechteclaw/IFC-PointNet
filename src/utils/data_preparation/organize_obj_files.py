""" 
It groups the files and copies to a subfolder corresponding to their categories (IFC entities). 
It also adds the .json file for each .obj with the metadata.
The script takes into account the file name following a specific format. 
"""

import os
import shutil
import sys
from tqdm import tqdm

# Path to the folder containing the files
# FOLDER_PATH = sys.argv[0]
FOLDER_PATH = (
    # r"C:\Code\IFC-extracted_elements_dataset\IFC_extracted_elements_sample"
    r"C:\Code\IFC-extracted_elements_dataset\IFC_extracted_elements_dataset_all_2"
)

# List all files in the folder
files = [
    f for f in os.listdir(FOLDER_PATH) if os.path.isfile(os.path.join(FOLDER_PATH, f))
]

# Process each file
unique_categories = set()
categories_count = {}


for file in tqdm(files, desc=f"Processing..."):
    # Extract category from the file name
    id_number, category = file.split("_", 1)

    # merge similar categories:
    if category == "IfcWallStandardCase":
        category = "IfcWall"
    elif category == "IfcStairFlight":
        category = "IfcStair"
    # skip proxy elements:
    elif category == "IfcBuildingElementProxy":
        continue

    # Define the new subfolder path based on the category
    subfolder_path = os.path.join(
        FOLDER_PATH, "sorted", category.split(".")[0]
    )  # Remove file extension from category if present

    # Create the subfolder if it doesn't already exist
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Move the file into the corresponding subfolder
    shutil.copy(
        os.path.join(FOLDER_PATH, file),
        os.path.join(subfolder_path, id_number + ".obj"),
    )
    with open(os.path.join(subfolder_path, id_number + ".obj"), "r", encoding="latin-1") as fileX:
        lines = fileX.readlines()
    with open(os.path.join(subfolder_path, id_number + ".obj"), "w", encoding="latin-1") as fileX:
        linex = []
        for line in lines[7:]:
            if line[0] == "v":
                linex.append(
                    f"v {round(float(line.split(' ')[1]), 5)} {round(float(line.split(' ')[2]), 5)} {round(float(line.split(' ')[3]), 5)}\n"
                )
            else:
                linex.append(line)
        fileX.writelines(linex)

    shutil.copy(
        os.path.join(FOLDER_PATH, file),
        os.path.join(subfolder_path, id_number + ".json"),
    )
    with open(os.path.join(subfolder_path, id_number + ".json"), "r", encoding="latin-1") as fileX:
        lines = fileX.readlines()

    with open(os.path.join(subfolder_path, id_number + ".json"), "w", encoding="latin-1") as fileX:
        new_lines = ["{\n"]
        new_lines.append(f'    "Source file": "{lines[0].split(" ")[3][1:-2]}",\n')
        new_lines.append(f'    "GlobalId": "{lines[1].split(" ")[2][1:-2]}",\n')
        # merge similar categories:
        category = lines[2].split(" ")[2][1:-2]
        if category == "IfcWallStandardCase":
            category = "IfcWall"
        elif category == "IfcStairFlight":
            category = "IfcStair"
        unique_categories.add(category)
        if not category in categories_count:
            categories_count[category] = 0
        categories_count[category] += 1
        new_lines.append(f'    "Entity": "{category}",\n')
        try:
            new_lines.append(f'    "Name": "{lines[3].split(" ")[2][:-1]}",\n')
        except:
            pass
        # Name: D02
        # new_lines.append(f'    "ObjectType": "{lines[4].split(" ")[2][1:-3]}",\n')
        # ObjectType: None
        try:
            new_lines.append(
                f'    "Location": {lines[5].split(" ")[2][:-1].replace(".,",",").replace(".]","]")},\n'
            )
        except:
            pass
        # location: [-22.,26.148,0.]
        try:
            new_lines.append(
                f'    "Rotation": {lines[6].split(" ")[2][:-1].replace(".,",",").replace(".]","]")}\n'
            )
        except:
            pass

        # for i in range(0, 7):
        #     new_lines.append(
        #         f'"{lines[i].split(" ")[1]}": "{lines[i].split(" ")[2]}",\n'
        #     )
        new_lines.append("}")

        fileX.writelines(new_lines)

print("-- Files have been organized into subfolders. --")
for cat in categories_count:
    print(f"{cat}: {categories_count[cat]}")
