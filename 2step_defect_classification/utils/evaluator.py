import numpy as np
import os
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
from scipy.sparse import coo_matrix
from scipy.stats import hmean, gmean

from torchvision.utils import save_image
from torchvision import transforms


class Evaluator:
    """Evaluator for classification."""

    def __init__(self, cfg, many_idxs=None, med_idxs=None, few_idxs=None):
        self.cfg = cfg
        self.many_idxs = many_idxs
        self.med_idxs = med_idxs
        self.few_idxs = few_idxs
        self.reset()

    def reset(self):
        self._correct = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._y_conf = []  # Store prediction confidences

        self._pred_normal = 0
        self._pred_abnormal = 0
        self._total_fnr = 0
        self._total_fpr = 0

        self._image_check_fpr = 0
        self._image_check_fnr = 0
        self._image_check = 0



    def fnr_process(self, image, mo, gt, normal_label, output_dir, image_check):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        image = self.denormalize(image)

        # Find the indices where the value changes from a non-10 to 10
        image_start_idx = torch.nonzero(gt != normal_label, as_tuple=False)

        if image_start_idx.nelement()!=0:
            image_start_idx = image_start_idx.min()
            print("Start index of the number 10:", image_start_idx)

        # Print the result
        

        pred = mo.max(1)[1]

        # minus gt=="normal"
        check = gt != normal_label
        pred_normal = pred[check]
        gt_abnormal = gt[check]

        # collect pred=="normal"
        binary_masks = pred_normal == normal_label
        pred_normal = torch.sum(binary_masks == True)

        image_idx = binary_masks.nonzero().squeeze()
        image_idx_check = binary_masks.nonzero().squeeze()

        if image_idx.nelement()!=0:
            image_idx = image_idx + image_start_idx
            image_idx_check = image_idx_check + image_start_idx


        if image_check == True:
            if image_idx.nelement() != 0:
                # grayscale = transforms.Grayscale()
                image_idx = image_idx.tolist()
                if type(image_idx) == int:
                    i = image_idx
                    image_split = image[i]
                    # image_split = grayscale(image_split)
                    self._image_check_fnr +=1
                    if output_dir == None:
                        pass
                        # save_image(image_split, './figure/increase_normal/fnr_check_{}.jpg'.format(self._image_check_fnr))     
                    else:
                        save_image(image_split, './image_check/fnr_image/{}/{}.jpg'.format(output_dir, self._image_check_fnr))     
                else:
                    for i in image_idx:
                        image_split = image[i]
                        # image_split = grayscale(image_split)
                        self._image_check_fnr +=1
                        if output_dir == None:
                            pass
                            # save_image(image_split, './figure/increase_normal/fnr_check_{}.jpg'.format(self._image_check_fnr))     
                        else:
                            save_image(image_split, './image_check/fnr_image/{}/{}.jpg'.format(output_dir, self._image_check_fnr))         

        # conf = torch.softmax(mo, dim=1).max(1)[0]  # Compute prediction confidences
        self._pred_normal += pred_normal
        self._total_fnr += gt_abnormal.shape[0]

        return image_idx_check

    def fpr_process(self, image, mo, gt, normal_label, output_dir, image_check):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]

        image = self.denormalize(image)

        pred = mo.max(1)[1]

        # minus gt=="normal"
        check = gt == normal_label
        pred_abnormal = pred[check]
        gt_normal = gt[check]

        # collect pred=="normal"
        binary_masks = pred_abnormal !=normal_label
        pred_abnormal = torch.sum(binary_masks == True)

        image_idx = binary_masks.nonzero().squeeze()
        image_idx_check = binary_masks.nonzero().squeeze()


        if image_check == True:
            if image_idx.nelement() != 0:
                # grayscale = transforms.Grayscale()
                image_idx = image_idx.tolist()
                if type(image_idx) == int:
                    i = image_idx
                    image_split = image[i]
                    # image_split = grayscale(image_split)
                    self._image_check_fpr +=1
                    if output_dir == None:
                        pass
                        # save_image(image_split, './figure/increase_normal/fpr_check_{}.jpg'.format(self._image_check_fpr))     
                    else:
                        save_image(image_split, './image_check/fpr_image/{}/{}.jpg'.format(output_dir, self._image_check_fpr))   
                else:            
                    for i in image_idx:
                        image_split = image[i]
                        # image_split = grayscale(image_split)
                        self._image_check_fpr +=1
                        if output_dir == None:
                            pass
                            # save_image(image_split, './figure/increase_normal/fpr_check_{}.jpg'.format(self._image_check_fpr))     
                        else:
                            save_image(image_split, './image_check/fpr_image/{}/{}.jpg'.format(output_dir, self._image_check_fpr))    

        # conf = torch.softmax(mo, dim=1).max(1)[0]  # Compute prediction confidences
        self._pred_abnormal += pred_abnormal
        self._total_fpr += gt_normal.shape[0]


        return image_idx_check

    def fnr_evaluate(self):
        fnr = self._pred_normal / self._total_fnr

        print(
            "=> fnr_result\n"
            f"* # of abnormal data total, must be 9000 : {self._total_fnr}\n"
            f"* fnr: {fnr}\n"
        )

    def fpr_evaluate(self):
        fpr = self._pred_abnormal / self._total_fpr

        print(
            "=> fpr_result\n"
            f"* # of abnormal data total, must be 9000 : {self._total_fpr}\n"
            f"* fpr: {fpr}\n"
        )

    def denormalize(self, image):
        # Define the mean and std used in normalization
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        # Create a denormalization transform
        denormalize = transforms.Normalize(mean=[-m/s for m, s in zip(mean, std)], std=[1/s for s in std])

        # Denormalize the tensor_images
        image = denormalize(image)
        image = image.squeeze()


        return image


    def process(self, image, mo, gt, output_dir, is_best, image_check):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        image = self.denormalize(image)
        pred = mo.max(1)[1]
        conf = torch.softmax(mo, dim=1).max(1)[0]  # Compute prediction confidences
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())
        self._y_conf.extend(conf.data.cpu().numpy().tolist())

        image_idx = matches.data.cpu().numpy()
        image_idx = np.where(image_idx==0)[0]
        class_idx = gt.data.cpu().numpy().tolist()
        pred_idx = pred.data.cpu().numpy().tolist()

        if image_check == True:
            for i in image_idx:
                wrong_class_idx = class_idx[i]
                wrong_pred_idx = pred_idx[i]
                image_split = image[i]
                # image_split = grayscale(image_split)
                self._image_check +=1
                if self.cfg.tsne:
                    pass
                else:
                    save_image(image_split, './image_check/top_1_image/{}/class{}_to_predict_{}_{}th.jpg'.format(output_dir, wrong_class_idx, wrong_pred_idx, i%30))  


    # def top1_delete_process(self, output_dir, is_best):
    #     if is_best == True:
    #         folder_path =  './image_check/top_1_image/{}'.format(output_dir)
    #         for filename in os.listdir(folder_path):
    #             file_path = os.path.join(folder_path, filename)
    #             os.remove(file_path)
    #             print(f"삭제됨: {file_path}")



    def evaluate(self):
        results = OrderedDict()

        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        self._per_class_res = defaultdict(list)

        for label, pred in zip(self._y_true, self._y_pred):
            matches = int(label == pred)
            self._per_class_res[label].append(matches)

        labels = list(self._per_class_res.keys())
        labels.sort()

        cls_accs = []
        for label in labels:
            res = self._per_class_res[label]
            correct = sum(res)
            total = len(res)
            acc = 100.0 * correct / total
            cls_accs.append(acc)
        
        accs_string = np.array2string(np.array(cls_accs), precision=2)
        print(f"* class acc: {accs_string}")

        # Compute worst case accuracy
        worst_case_acc = min([acc for acc in cls_accs])

        # Compute lowest recall
        # lowest_recall = min([100.0 * sum(res) / self.cls_num_list[label] for label, res in self._per_class_res.items()])

        # Compute harmonic mean
        hmean_acc = 100.0 / np.mean([1.0 / (max(acc, 0.001) / 100.0) for acc in cls_accs])

        # Compute geometric mean
        gmean_acc = 100.0 * np.prod([acc / 100.0 for acc in cls_accs]) ** (1.0 / len(cls_accs))

        results["worst_case_acc"] = worst_case_acc
        # results["lowest_recall"] = lowest_recall
        results["hmean_acc"] = hmean_acc
        results["gmean_acc"] = gmean_acc

        print(
            f"* worst_case_acc: {worst_case_acc:.1f}%\n"
            # f"* lowest_recall: {lowest_recall:.1f}%\n"
            f"* hmean_acc: {hmean_acc:.1f}%\n"
            f"* gmean_acc: {gmean_acc:.1f}%"
        )

        if self.many_idxs is not None and self.med_idxs is not None and self.few_idxs is not None:
            many_acc = np.mean(np.array(cls_accs)[self.many_idxs])
            med_acc = np.mean(np.array(cls_accs)[self.med_idxs])
            few_acc = np.mean(np.array(cls_accs)[self.few_idxs])
            results["many_acc"] = many_acc
            results["med_acc"] = med_acc
            results["few_acc"] = few_acc
            print(f"* many: {many_acc:.1f}%  med: {med_acc:.1f}%  few: {few_acc:.1f}%")

        mean_acc = np.mean(cls_accs)
        results["mean_acc"] = mean_acc
        print(f"* average: {mean_acc:.1f}%")

        # for wafer
        # many_shot = np.array([ True,  True,  True,  True ,False ,False, False, False ,False, False])
        # medium_shot = np.array([False, False, False, False , True  ,True , True, False, False ,False])
        # few_shot = np.array([False, False , False ,False ,False, False, False , True,  True , True])

        # # for magnetic
        many_shot = np.array([ False,  False,  False,  False ,False])
        medium_shot = np.array([True, True, True, False , True])
        few_shot = np.array([False, False , False ,True ,False])

        # # for dagm
        # many_shot = np.array([ False,  False,  False,  False ,False ,False, True, True ,True, True])
        # medium_shot = np.array([True, True, True, True , True  ,True , False, False, False ,False])
        # few_shot = np.array([False, False , False ,False ,False, False, False , False,  False , True])

        if self.cfg.patch == True:
            #confusion

            cmat = confusion_matrix(self._y_true, self._y_pred)
            cmat = torch.from_numpy(cmat)
        

            print(cmat)

            overall_with_normal = cmat.diag().sum()/cmat.sum()  

            acc_per_class = cmat.diag()/cmat.sum(1) 
            acc = acc_per_class.cpu().numpy() 


            many_shot_acc = acc[many_shot].mean()
            medium_shot_acc = acc[medium_shot].mean()
            few_shot_acc = acc[few_shot].mean() 

            print("overall_with_normal:",  overall_with_normal,

                    "many_class_num:", many_shot.sum(),
                    "medium_class_num:", medium_shot.sum(),
                    "few_class_num:", few_shot.sum(),
                    "many_shot_acc:", many_shot_acc,
                    "medium_shot_acc:", medium_shot_acc,
                    "few_shot_acc:", few_shot_acc)

            results["overall_with_normal"] = overall_with_normal

            results["many_class_num"] = many_shot.sum()
            results["medium_class_num"] = medium_shot.sum()
            results["few_class_num"] = few_shot.sum()

            results["many_shot_acc"] = many_shot_acc
            results["medium_shot_acc"] = medium_shot_acc
            results["few_shot_acc"] = few_shot_acc


            return results
            # Compute expected calibration error
            # ece = 100.0 * expected_calibration_error(
            #     self._y_conf,
            #     self._y_pred,
            #     self._y_true
            # )
            # results["expected_calibration_error"] = ece
            # print(f"* expected_calibration_error: {ece:.2f}%")
            
            # cmat = coo_matrix(cmat)
            # save_path = os.path.join(self.cfg.output_dir, "cmat.pt")
            # torch.save(cmat, save_path)
            # print(f"Confusion matrix is saved to {save_path}")

        else:
            "non patch"
            # Compute confusion matrix
            cmat = confusion_matrix(self._y_true, self._y_pred)
            cmat = torch.from_numpy(cmat)
        
            normal_label = self.cfg.normal_label

            print(cmat)
            # print(cmat[normal_label][normal_label])
            # print(cmat[normal_label])

            # print(cmat[:normal_label, normal_label])
            # print(cmat[:normal_label].sum())

            # print(cmat.diag())
            # print(cmat.sum())
            # print(cmat.diag()[:normal_label])
            print(cmat[:normal_label, :normal_label])

            #fpr
            fpr = 1-(cmat[normal_label][normal_label]/cmat[normal_label].sum())

            #fnr
            fnr = cmat[:normal_label, normal_label].sum()/cmat[:normal_label].sum()
            print(fpr,fnr)

            overall_with_normal = cmat.diag().sum()/cmat.sum()  
            overall_without_normal = cmat.diag()[:normal_label].sum()/cmat[:normal_label, :].sum()

            acc_per_class = cmat.diag()/cmat.sum(1) 
            acc = acc_per_class.cpu().numpy() 
            print("acc_per_class:",acc_per_class)
            print("acc:",acc)

            acc_many = np.delete(acc, [normal_label], axis=None)
            print("acc_many:",acc_many)

            many_shot_acc = acc_many[many_shot].mean()
            medium_shot_acc = acc_many[medium_shot].mean()
            few_shot_acc = acc_many[few_shot].mean() 

            # n_samples = len(data_loader.sampler)
            # log = {}
            # log.update({
            #     met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
            # })

            print("overall_with_normal:",  overall_with_normal,
                    "overall_without_normal:", overall_without_normal,
                    "FNR:", fnr,
                    "FPR:", fpr,

                    "many_class_num:", many_shot.sum(),
                    "medium_class_num:", medium_shot.sum(),
                    "few_class_num:", few_shot.sum(),
                    "many_shot_acc:", many_shot_acc,
                    "medium_shot_acc:", medium_shot_acc,
                    "few_shot_acc:", few_shot_acc)

            results["overall_with_normal"] = overall_with_normal
            results["overall_without_normal"] = overall_without_normal
            results["FNR"] = fnr
            results["FPR"] = fpr

            results["many_class_num"] = many_shot.sum()
            results["medium_class_num"] = medium_shot.sum()
            results["few_class_num"] = few_shot.sum()

            results["many_shot_acc"] = many_shot_acc
            results["medium_shot_acc"] = medium_shot_acc
            results["few_shot_acc"] = few_shot_acc
            

            return results




def compute_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


def expected_calibration_error(confs, preds, labels, num_bins=10):
    def _populate_bins(confs, preds, labels, num_bins):
        bin_dict = defaultdict(lambda: {'bin_accuracy': 0, 'bin_confidence': 0, 'count': 0})
        bins = np.linspace(0, 1, num_bins + 1)
        for conf, pred, label in zip(confs, preds, labels):
            bin_idx = np.searchsorted(bins, conf) - 1
            bin_dict[bin_idx]['bin_accuracy'] += int(pred == label)
            bin_dict[bin_idx]['bin_confidence'] += conf
            bin_dict[bin_idx]['count'] += 1
        return bin_dict

    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    num_samples = len(labels)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i]['bin_accuracy']
        bin_confidence = bin_dict[i]['bin_confidence']
        bin_count = bin_dict[i]['count']
        ece += (float(bin_count) / num_samples) * \
               abs(bin_accuracy / bin_count - bin_confidence / bin_count)
    return ece
