import os
import random
import shutil


def select_random_file_pairs(source_folder, num_pairs):
    # Get list of files in the source folder
    file_list = os.listdir(source_folder)
    file_set = set(
        [file.replace(".obj", "").replace(".json", "") for file in file_list]
    )

    # Check if the number of files is greater than num_files
    if len(file_set) <= num_pairs:
        print(f"Skipping: {source_folder} as it does not contain more than {num_pairs} files.")
        return

    # Select num_files random files
    random_pair_names = random.sample(list(file_set), num_pairs)

    return random_pair_names


def process_a_folder(folder_name, destination_folder, num_files_to_select=1200):

    random_pair_names = select_random_file_pairs(folder_name, num_files_to_select)

    if random_pair_names:

        # Create destination folder if it doesn't exist
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        # Copy selected files to the destination folder
        for file_name in random_pair_names:
            source_file = os.path.join(folder_name, file_name)
            destination_file = os.path.join(destination_folder, file_name)
            shutil.copyfile(source_file+".obj", destination_file+".obj")
            shutil.copyfile(source_file+".json", destination_file+".json")

        print(f"{num_files_to_select} random files copied to {destination_folder}.")
    else:
        print("No files copied.")


if __name__ == "__main__":

    source_folder = r"C:\Code\IFC-extracted_elements_dataset\all_sorted"
    num_files_to_select = 7000
    destination_folder = "C:\Code\IFC-extracted_elements_dataset\sample_sorted"

    folder = os.listdir(source_folder)
    for subfolder in folder:
        process_a_folder(
            os.path.join(source_folder, subfolder),
            os.path.join(destination_folder, subfolder),
            num_files_to_select,
        )
