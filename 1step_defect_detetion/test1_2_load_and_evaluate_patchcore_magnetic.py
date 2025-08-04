import contextlib
import gc
import logging
import os
import sys
import json
import click
import numpy as np
import torch

import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

import dataloader

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True)
def main(**kwargs):
    pass


@main.result_callback()
def run(methods, results_path, gpu, seed, log_group, log_project, save_segmentation_images):
    methods = {key: item for (key, item) in methods}

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    os.makedirs(results_path, exist_ok=True)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []
    result_collect2 = []

    dataloader_iter, n_dataloaders = methods["get_dataloaders_iter"]
    dataloader_iter = dataloader_iter(seed)
    patchcore_iter, n_patchcores = methods["get_patchcore_iter"]
    patchcore_iter = patchcore_iter(device)
    if not (n_dataloaders == n_patchcores or n_patchcores == 1):
        raise ValueError(
            "Please ensure that #PatchCores == #Datasets or #PatchCores == 1!"
        )


    for dataloader_count, dataloaders in enumerate(dataloader_iter):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["thresholding"].name, dataloader_count + 1, n_dataloaders
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        #set thresholding for each multi-class memory bank based on augmented normal and abnormal images
        dataset_name = dataloaders["thresholding"].name

        with device_context:

            torch.cuda.empty_cache()
            if dataloader_count < n_patchcores:
                PatchCore_list = next(patchcore_iter)

            aggregator = {"train_scores": [], "train_segmentations": [], "scores": [], "segmentations": []}

            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding train_test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, segmentations, labels_gt, images = PatchCore.predict(
                    dataloaders["thresholding"], is_defect_aug=True
                )
                aggregator["train_scores"].append(scores)
                aggregator["train_segmentations"].append(segmentations)

            labels_gt = np.array(labels_gt)

            scores = np.array(aggregator["train_scores"])
            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            image_auroc= patchcore.metrics.compute_train_threshold(
                scores, labels_gt
            )

            train_auroc = image_auroc["train_auroc"]
            train_roc_optimal_threshold = image_auroc["train_roc_optimal_threshold"]
            train_F1_optimal_threshold = image_auroc["train_F1_optimal_threshold"]

            LOGGER.info("Computing evaluation metrics for train_test thresholding.")
            

            result_collect.append(
                {
                    "dataset_name": dataset_name,
                    "train_auroc": train_auroc,
                    "train_roc_optimal_threshold":train_roc_optimal_threshold,
                    "train_F1_optimal_threshold":train_F1_optimal_threshold,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.4f}".format(key, item))

            #inference test_data with threshold based on augmented normal and abnormal images

            torch.cuda.empty_cache()
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, segmentations, labels_gt, masks_gt, images = PatchCore.predict(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            labels_gt = np.array(labels_gt)

            scores = np.array(aggregator["scores"])
            #min_scores = scores.min(axis=-1).reshape(-1, 1)
            #max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            # (Optional) Plot example images.
            if save_segmentation_images:
                image_paths = dataloaders["testing"].dataset.x
                mask_paths = dataloaders["testing"].dataset.mask

                IMAGENET_MEAN = [0.5, 0.5, 0.5]
                IMAGENET_STD = [0.5, 0.5, 0.5]
                # IMAGENET_MEAN = [0.485, 0.456, 0.406]
                # IMAGENET_STD = [0.229, 0.224, 0.225]
                def image_transform(image):
                    in_std = np.array(
                        IMAGENET_STD
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        IMAGENET_MEAN
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_x(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)
                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()
                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )

                os.makedirs(image_save_path, exist_ok=True)
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                )

            LOGGER.info("Computing evaluation metrics for testset.")
            image_file_path = dataloaders["testing"].dataset.x
            # # trainset optimal threshold를 testset에 넣어서 계산~
            image_auroc= patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, labels_gt, images, image_file_path, train_roc_optimal_threshold, train_F1_optimal_threshold, results_path, \
                log_project, log_group, dataloaders["thresholding"].name, run_save_path
            )
            number_of_test=len(labels_gt)
            print(number_of_test)
            auroc = image_auroc["auroc"]

            test_FNR_from_train_F1_optimal_threshold = image_auroc["test_FNR_from_train_F1_optimal_threshold"]
            test_FNR_from_train_roc_optimal_threshold = image_auroc["test_FNR_from_train_roc_optimal_threshold"]
            test_FNR_from_test_F1_optimal_threshold = image_auroc["test_FNR_from_test_F1_optimal_threshold"]
            test_FNR_from_test_roc_optimal_threshold = image_auroc["test_FNR_from_test_roc_optimal_threshold"]

            test_FPR_from_train_F1_optimal_threshold = image_auroc["test_FPR_from_train_F1_optimal_threshold"]
            test_FPR_from_train_roc_optimal_threshold = image_auroc["test_FPR_from_train_roc_optimal_threshold"]
            test_FPR_from_test_F1_optimal_threshold = image_auroc["test_FPR_from_test_F1_optimal_threshold"]
            test_FPR_from_test_roc_optimal_threshold = image_auroc["test_FPR_from_test_roc_optimal_threshold"]



            result_collect.append(
                {
                    "instance_auroc": auroc,
                    "test_FNR_from_train_F1_optimal_threshold":test_FNR_from_train_F1_optimal_threshold,
                    "test_FPR_from_train_F1_optimal_threshold":test_FPR_from_train_F1_optimal_threshold,
                    "test_FNR_from_train_roc_optimal_threshold":test_FNR_from_train_roc_optimal_threshold,
                    "test_FPR_from_train_roc_optimal_threshold":test_FPR_from_train_roc_optimal_threshold,

                    "test_FNR_from_test_F1_optimal_threshold":test_FNR_from_test_F1_optimal_threshold,
                    "test_FPR_from_test_F1_optimal_threshold":test_FPR_from_test_F1_optimal_threshold,
                    "test_FNR_from_test_roc_optimal_threshold":test_FNR_from_test_roc_optimal_threshold,
                    "test_FPR_from_test_roc_optimal_threshold":test_FPR_from_test_roc_optimal_threshold


                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.4f}".format(key, item))

                
            # Compute PRO score & PW Auroc for all images
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]

            # Compute PRO score & PW Auroc only images with anomalies
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )
            anomaly_pixel_auroc = pixel_scores["auroc"]


            #pdb.set_trace()
            result_collect.append(
                {
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                }
            )

            result_collect2.append(

                {
                    "instance_auroc": auroc,
                    "test_FNR_from_train_F1_optimal_threshold":test_FNR_from_train_F1_optimal_threshold,
                    "test_FPR_from_train_F1_optimal_threshold":test_FPR_from_train_F1_optimal_threshold,
                    "test_FNR_from_train_roc_optimal_threshold":test_FNR_from_train_roc_optimal_threshold,
                    "test_FPR_from_train_roc_optimal_threshold":test_FPR_from_train_roc_optimal_threshold,
                    "test_FNR_from_test_F1_optimal_threshold":test_FNR_from_test_F1_optimal_threshold,
                    "test_FPR_from_test_F1_optimal_threshold":test_FPR_from_test_F1_optimal_threshold,
                    "test_FNR_from_test_roc_optimal_threshold":test_FNR_from_test_roc_optimal_threshold,
                    "test_FPR_from_test_roc_optimal_threshold":test_FPR_from_test_roc_optimal_threshold,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                    "number_of_test":number_of_test
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info("{0}: {1:3.4f}".format(key, item))
            #print(result_collect)
            #print(dataset_name)
            data_convert = {k:float(v) for k,v in result_collect2[-1].items()}

            result_folder = './results_final/magnetic_cluster'
            if not os.path.exists(result_folder):
                os.makedirs(result_folder)
            file_path = os.path.join(result_folder, f'result_{dataset_name}.json')
            with open(file_path,'w') as json_file:
                json.dump(data_convert, json_file, indent=4)

           

            del PatchCore_list
            gc.collect()

        LOGGER.info("\n\n-----\n")

    result_metric_names = list(result_collect[-1].keys())[1:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_scores = [list(results.values())[1:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        results_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
    )
    


@main.command("patch_core_loader")
# Pretraining-specific parameters.
@click.option("--patch_core_paths", "-p", type=str, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
def patch_core_loader(patch_core_paths, faiss_on_gpu, faiss_num_workers):
    def get_patchcore_iter(device):
        for patch_core_path in patch_core_paths:
            loaded_patchcores = []
            gc.collect()
            n_patchcores = len(
                [x for x in os.listdir(patch_core_path) if ".faiss" in x]
            )
            if n_patchcores == 1:
                nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
                patchcore_instance = patchcore.patchcore.PatchCore(device)
                patchcore_instance.load_from_path(
                    load_path=patch_core_path, device=device, nn_method=nn_method
                )
                loaded_patchcores.append(patchcore_instance)
            else:
                for i in range(n_patchcores):
                    nn_method = patchcore.common.FaissNN(
                        faiss_on_gpu, faiss_num_workers
                    )
                    patchcore_instance = patchcore.patchcore.PatchCore(device)
                    patchcore_instance.load_from_path(
                        load_path=patch_core_path,
                        device=device,
                        nn_method=nn_method,
                        prepend="Ensemble-{}-{}_".format(i + 1, n_patchcores),
                    )
                    loaded_patchcores.append(patchcore_instance)

            yield loaded_patchcores

    return ("get_patchcore_iter", [get_patchcore_iter, len(patch_core_paths)])


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path_test", type=click.Path(exists=True, file_okay=False))
@click.argument("data_path_threshold_train", type=click.Path(exists=True, file_okay=False))

@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)

@click.option("--batch_size", default=1, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option("--augment", is_flag=True)


def dataset(
    name,
    data_path_test,
    data_path_threshold_train,
    subdatasets,
    batch_size,
    resize,
    imagesize,
    num_workers,
    augment,
):
    # dataset_info = _DATASETS[name]
    # dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders_iter(seed):

        for subdataset in subdatasets:
            test_dataset = dataloader.MyDataset(dataset_path=data_path_test,
                                                class_name=subdataset,
                                                resize=resize,
                                                cropsize=imagesize,
                                                is_train=False,
                                                )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            threshold_train_dataset = dataloader.MyDataset(dataset_path=data_path_threshold_train,
                                                class_name=subdataset,
                                                resize=resize,
                                                cropsize=imagesize,
                                                is_train=False,
                                                have_gt=False,
                                                is_defect_aug=True
                                                )

            threshold_train_dataloader = torch.utils.data.DataLoader(
                threshold_train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader.name = name
            threshold_train_dataloader.name = name
            if subdataset is not None:
                test_dataloader.name += "_" + subdataset
                threshold_train_dataloader.name += "_" + subdataset


            dataloader_dict = {
                "testing": test_dataloader,
                "thresholding": threshold_train_dataloader
            }

            yield dataloader_dict

    return ("get_dataloaders_iter", [get_dataloaders_iter, len(subdatasets)])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()