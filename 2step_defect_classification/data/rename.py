import os

# Define the directory path where your files are located
folder_path = './dagm/dagm_confirm_ver4_aug_test/train/9'  # Replace with your folder path

# Loop through all files in the directory
for filename in os.listdir(folder_path):
    # Check if the file is a .jpg
    if filename.endswith('.jpg'):
        # Construct the full file path
        old_file = os.path.join(folder_path, filename)
        
        # Create a new file name with .png extension
        new_file = os.path.join(folder_path, filename[:-4] + '.png')
        
        # Rename the file
        os.rename(old_file, new_file)

print("All .jpg files have been renamed to .png")