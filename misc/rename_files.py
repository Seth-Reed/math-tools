import os


def rename_files(folder_path, prefix, extension):
    # Get list of files in the folder
    files = os.listdir(folder_path)

    # Loop through each file
    for i, file_name in enumerate(files):
        # Check if file ends with the specified extension
        if file_name.endswith(extension):
            new_file_name = f'{prefix}_{i}{extension}'
            os.rename(os.path.join(folder_path, file_name), os.path.join(folder_path, new_file_name))
            print(f'Renamed {file_name} to {new_file_name}')

