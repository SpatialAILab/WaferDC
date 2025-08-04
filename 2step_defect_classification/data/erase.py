import os

folder_path = "./wafer_confirm_ver4_multi_bank_normal_aug_patch_ver6/train/9"  # Replace with the actual path to your folder

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Iterate through the files and delete those that do not match the desired pattern
for file_name in files:
    if file_name.endswith("sorted_003.jpg") or file_name.endswith("sorted_004.jpg"):
        continue  # Skip files you want to keep
    else:
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)
        print(f"Deleted: {file_path}")