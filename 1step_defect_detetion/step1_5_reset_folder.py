import shutil
import os
from tqdm import tqdm

def copy_folder(source_path, destination_path):
    try:
        # Remove the existing destination folder if it exists
        shutil.rmtree(destination_path, ignore_errors=True)
        
        # Copy the entire folder and its contents to the new path
        shutil.copytree(source_path, destination_path)
        print(f"Folder copied from {source_path} to {destination_path}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":

    # for wafer

    name="wafer"
    subdatasets=('0' '1' '2' '3' '4' '5' '6')
    cluster_num = 7
    #for Dagm
    #name="dagm"
    #subdatasets=('0'  '1'  '2' '3' '4' '5' '6' '7' '8')

    #for magnetic
    # name="magnetic"
    # subdatasets=('0')
    # cluster_num = 5
    
    reset_dataset_path=f"data/{name}/lets_cluster/{name}_{cluster_num}_cluster_1_5"


    os.makedirs(reset_dataset_path,exist_ok=True)


    for subdataset in tqdm(subdatasets):
        os.makedirs(reset_dataset_path,exist_ok=True)
        os.makedirs(os.path.join(reset_dataset_path,subdataset),exist_ok=True)
        os.makedirs(os.path.join(reset_dataset_path,subdataset,"test"),exist_ok=True)

        train_good_folder = f"data/{name}/lets_cluster/{name}_{cluster_num}_cluster/{subdataset}/train/good"
        train_clustered_abnormal = f"data/{name}/lets_cluster/{name}_{cluster_num}_cluster/{subdataset}/test/clustered_abnormal"
        #train_augmented_defect = f"data/{name}/lets_cluster/{name}_{cluster_num}_cluster/{subdataset}/test/augmented_defect/result"

        reset_train_good_folder = os.path.join(reset_dataset_path,subdataset,"test/good")
        reset_train_clustered_abnormal = os.path.join(reset_dataset_path,subdataset,"test/clustered_abnormal")
        #reset_train_augmented_defect = os.path.join(reset_dataset_path,subdataset,"test/augmented_defect")


        os.makedirs(reset_train_good_folder,exist_ok=True)
        os.makedirs(reset_train_clustered_abnormal,exist_ok=True)
        #os.makedirs(reset_train_augmented_defect,exist_ok=True)

        # Example usage:
        copy_folder(train_good_folder, reset_train_good_folder)
        copy_folder(train_clustered_abnormal, reset_train_clustered_abnormal)
        #copy_folder(train_augmented_defect, reset_train_augmented_defect)