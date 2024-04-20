""" It groups the files and copies to a subfolder corresponding to their categories (IFC entities). The script takes into account the file name following a specific format. """

import os
import shutil
import sys


# Path to the folder containing the files
FOLDER_PATH = sys.argv[0]

# List all files in the folder
files = [
    f for f in os.listdir(FOLDER_PATH) if os.path.isfile(os.path.join(FOLDER_PATH, f))
]

# Process each file
# TODO: Remove comments, code is self-explanatory
for file in files:
    # Extract category from the file name
    id_number, category = file.split("_", 1)

    # Define the new subfolder path based on the category
    subfolder_path = os.path.join(
        FOLDER_PATH, category.split(".")[0]
    )  # Remove file extension from category if present

    # Create the subfolder if it doesn't already exist
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # Move the file into the corresponding subfolder
    shutil.copy(os.path.join(FOLDER_PATH, file), os.path.join(subfolder_path, file))

print("Files have been organized into subfolders.")
