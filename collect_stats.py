import os
import re
import csv
from tqdm import tqdm  # Import tqdm for the progress bar

# Directory containing the files
directory_path = r"your\file\path"

# Output CSV file
output_csv = 'data_stats.csv'

# Regular expression to match the 'number_Label' format in filenames
pattern_filename = r'(\d+)_(\w+)'

# Updated regular expression to extract the filename from the first line within single quotes
pattern_first_line = r"^# Source file: '(.+)'$"

# Prepare to write to the CSV file
with open(output_csv, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Number', 'Label', 'Filename', 'Weight (KB)', 'Vertices Count', 'Faces Count'])

    # Get a list of files in the directory and wrap it with tqdm for a progress bar
    files = os.listdir(directory_path)
    for filename in tqdm(files, desc="Processing files"):
        if os.path.isfile(os.path.join(directory_path, filename)):
            # Extract number and label from the filename
            match_filename = re.match(pattern_filename, filename)
            if match_filename:
                number, label = match_filename.groups()

                # Initialize counters for vertices and faces
                vertices_count = 0
                faces_count = 0

                # Open file to extract detailed information
                with open(os.path.join(directory_path, filename), 'r', encoding='utf-8', errors='ignore') as f:
                
                    lines = f.readlines()
                    first_line = lines[0].strip()
                    match_first_line = re.match(pattern_first_line, first_line)
                    source_filename = match_first_line.group(1) if match_first_line else 'Unknown'
                    
                    # Count vertices and faces
                    for line in lines:
                        if line.startswith('v '):
                            vertices_count += 1
                        elif line.startswith('f '):
                            faces_count += 1

                # Get the file weight (size) in KB
                weight_kb = os.path.getsize(os.path.join(directory_path, filename)) / 1024

                # Write the extracted info to the CSV
                writer.writerow([number, label, source_filename, f"{weight_kb:.2f}", vertices_count, faces_count])

print("Extraction completed and data written to", output_csv)
