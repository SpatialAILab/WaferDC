#for wafer

#name="wafer"
#loadpath="./results/wafer_real_real/"

# one memory bank
# modelfolder="DINO_naive"
# data_path_threshold_train="data/wafer_confirm_ver4_one_bank_folder_reset"
# data_path_test="../test_data/wafer_confirm_ver4_test_one_bank"
# datasets=('0')             

# multi-class-non-aug
# modelfolder="DINO_normal_aug_before"
# data_path_threshold_train="data/wafer_confirm_ver4_multi_bank_normal_cluster_folder_reset"
# data_path_test="../test_data/wafer_confirm_ver4_test_multi_bank_normal_cluster"
# datasets=('0'  '1'  '2' '3' '4')              

#multi-class-aug
# modelfolder="wafer_real_real_2"
# data_path_threshold_train="data/wafer/wafer_confirm_ver5_step1_5_multi_bank_normal_aug_false_1000_real_ver2"
# data_path_test="../test_data/wafer_test/wafer_confirm_ver5_test_multi_bank_normal"
# datasets=('0' '1' '2' '3' '4')      

#for dagm

# name="dagm"
# loadpath="./results/dagm_original"
     
# multi-class-aug
# modelfolder="dagm_original"
# data_path_threshold_train="data/dagm/dagm_confirm_ver4_step1_5_multi_bank_original"
# data_path_test="../test_data/dagm_test/dagm_confirm_ver4_test_multi_bank_normal"
# datasets=('0' '1'  '2' '3' '4' '5' '6' '7' '8' '9')

#for magnetic
#magnetic_multi bank normal aug x
name="magnetic"
loadpath="./results/magnetic_cluster"
     
modelfolder="magnetic_cluster_num_4_1"
data_path_threshold_train="data/magnetic/lets_cluster/magnetic_4_cluster_1_5"
data_path_test="../test_data/magnetic_test/magnetic_4_cluster_test"
datasets=('0' '1' '2' '3')

# name="magnetic"
# loadpath="./results/magnetic_cluster"
     
# modelfolder="magnetic_cluster_num_0"
# data_path_threshold_train="data/magnetic/lets_cluster/magnetic_0_cluster_1_5"
# data_path_test="../test_data/magnetic_test/magnetic_no_cluster_test"
# datasets=('0') 


#change model_flag = 'models/wafer_' to 'model/dagm_'
savefolder=evaluated_results'/'$name'/'$modelfolder

model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/magnetic_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
#DINO_Ours
python3 test1_2_load_and_evaluate_patchcore_magnetic.py --gpu 1 --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 224 --imagesize 224 "${dataset_flags[@]}" "$name" "$data_path_test" "$data_path_threshold_train"

python3 balance_test_auto_magnetic.py

