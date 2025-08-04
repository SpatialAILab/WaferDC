
# magnetic
#name="magnetic"
# datasets=('0'  '1' )
#loadpath="./results/magnetic_results"

# # one memory bank
#datasets=('0')
#datapath="./data/magnetic/magnetic_confirm_ver4_step1_1_one_bank_half_5std" 

name="magnetic"
datasets=('0' '1' '2' '3')
datapath="./data/magnetic/lets_cluster/magnetic_4_cluster"
loadpath="./results/magnetic_cluster"
modelfolder="magnetic_cluster_num_4"

# # multi-class-aug
# multi-class-non-aug
# datapath="./data/magnetic/magnetic_confirm_ver4_step1_1_multi_bank" 
# modelfolder="DINO_normal_aug_before"

#have to change modelfolder ' model/wafer_'!!!!!!!!!!!!!!!
savefolder=evaluated_results'/'$modelfolder

model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/magnetic_'$dataset; done))
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))

#DINO_Ours
python3 step1_4_abnormal_augmentation_magnetic.py --gpu 3 --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu  \
dataset --resize 224 --imagesize 224 "${dataset_flags[@]}" "$name" "$datapath"
