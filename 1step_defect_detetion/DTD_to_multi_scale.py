import os
import numpy as np
from torchvision import transforms
from PIL import Image
import random
def random_crop_and_resize(image, crop_params, crop_size=(480, 480)):
    crop_size = (192, 192) # chgd

    i, j, h, w = crop_params
    transform = transforms.Compose([
        transforms.Lambda(lambda img: img[:, i:i+h, j:j+w]),
        transforms.Resize(crop_size)
    ])
    return transform(image)
def is_empty_mask(mask_image):
    # Check if the mask image is completely black (all pixels are 0)
    return np.all(np.array(mask_image) == 0)
def process_images(src_folder, dest_folder):
    for subfolder in ['good_all', 'Mesh_114', 'Perforated_037', 'Woven_001', 'Woven_068', 'Woven_127']:
        subfolder_path = os.path.join(src_folder, subfolder)

        if os.path.exists(subfolder_path):
            process_subfolder(subfolder_path, subfolder, dest_folder)
def process_subfolder(subfolder_path, subfolder_name, dest_folder):
    for item in os.listdir(subfolder_path):
        item_path = os.path.join(subfolder_path, item)

        print(item_path)

        if os.path.isdir(item_path):
            if item == 'train':
                process_good_images(item_path, subfolder_name, item, dest_folder)
            elif (item == 'test'):
                process_good_images(item_path, subfolder_name, item, dest_folder)

def process_good_images(good_folder_path, subfolder_name, item_name, dest_folder):
    for image_file in os.listdir(good_folder_path):


        if image_file.endswith('.png'):
            image_path = os.path.join(good_folder_path, image_file)
            image = Image.open(image_path)
            image = transforms.ToTensor()(image)
            # Random crop coordinates
            random_int = random.randint(image.shape[-1]*(3/4), image.shape[-1]) # chgd
            crop_transform = transforms.RandomCrop((random_int, random_int))
            i, j, h, w = crop_transform.get_params(image, (random_int, random_int))
            cropped_image = random_crop_and_resize(image, (i, j, h, w))
            save_path = os.path.join(dest_folder, subfolder_name, item_name)
            os.makedirs(save_path, exist_ok=True)
            cropped_image = transforms.ToPILImage()(cropped_image)
            cropped_image.save(os.path.join(save_path, image_file))
            print(f"Processed and saved: {os.path.join(save_path, image_file)}")

def process_defect_and_gt_images(defect_folder_path, subfolder_name, item_name, subfolder_path, dest_folder):
    gt_folder_path = os.path.join(subfolder_path.replace('test', 'ground_truth'), item_name)
    for image_file in os.listdir(defect_folder_path):
        if image_file.endswith('.png'):
            # Process defect image
            image_path = os.path.join(defect_folder_path, image_file)
            image = Image.open(image_path)
            image = transforms.ToTensor()(image)

            # Generate random crop coordinates from defect image
            random_int = random.randint(image.shape[-1]*(3/4), image.shape[-1]) # chgd
            crop_transform = transforms.RandomCrop((random_int, random_int))
            #rop_transform = transforms.RandomCrop((480, 480))
            i, j, h, w = crop_transform.get_params(image, (random_int, random_int))
            cropped_image = random_crop_and_resize(image, (i, j, h, w))
            # Process corresponding ground truth image with the same crop
            #gt_image_file = image_file.replace('.png', '_mask.png')
            gt_image_file = image_file # chgd
            gt_image_path = os.path.join(gt_folder_path, gt_image_file)
            if os.path.exists(gt_image_path):
                gt_image = Image.open(gt_image_path)
                gt_image = transforms.ToTensor()(gt_image)
                cropped_gt_image = random_crop_and_resize(gt_image, (i, j, h, w))
                # Convert back to PIL for checking if the mask is empty
                cropped_gt_image_pil = transforms.ToPILImage()(cropped_gt_image)
                if not is_empty_mask(cropped_gt_image_pil):
                    # Save the cropped images only if the mask is not empty
                    save_gt_path = os.path.join(dest_folder, 'ground_truth', item_name)
                    os.makedirs(save_gt_path, exist_ok=True)
                    cropped_gt_image_pil.save(os.path.join(save_gt_path, gt_image_file))
                    print(f"Processed and saved ground truth: {os.path.join(save_gt_path, gt_image_file)}")
                    save_defect_path = os.path.join(dest_folder, subfolder_name, item_name)
                    os.makedirs(save_defect_path, exist_ok=True)
                    cropped_image = transforms.ToPILImage()(cropped_image)
                    cropped_image.save(os.path.join(save_defect_path, image_file))
                    print(f"Processed and saved: {os.path.join(save_defect_path, image_file)}")
                else:
                    print(f"Skipped saving for {image_file} and corresponding ground truth due to empty mask.")
            else:
                print(f"Ground truth mask not found for: {image_file}")

# 사용 예시
src_folder = '/data/yhson/DTD-Synthetic-longtail/'  # 원본 폴더 경로
dest_folder = '/data/yhson/DTD-Synthetic-ms-longtail/'  # 결과 이미지 저장 폴더 경로
process_images(src_folder, dest_folder)