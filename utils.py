import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
import os.path
import scipy.misc
from numpy import loadtxt
import random
import matplotlib.pyplot as plt
from PIL import Image
import skimage.measure
import random
import skimage.measure
from skimage.transform import resize
import matplotlib.patches as patches

def compute_metrics(confusion_mat):
    n = confusion_mat.shape[0]
    rates = np.zeros((n,4)) # TP FP TN FN
    metrics = np.zeros((n+1,2))

    for i in range(n):
        other_indicies = np.arange(n)
        other_indicies = np.delete(other_indicies, i)

        TP = confusion_mat[i,i]
        FN = np.sum(confusion_mat[i,other_indicies])

        FP = np.sum(confusion_mat[other_indicies, i])
        TN = np.sum(confusion_mat[:]) - (TP + FN + FP)

        rates[i,0] = TP
        rates[i,1] = FN
        rates[i,2] = FP
        rates[i,3] = TN


        metrics[i,0] = TP /(TP+0.5*(FP+FN))
        metrics[i, 1] = (TP*TN - FP*FN)/ np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    metrics[-1, 0]= np.nanmean(metrics[:-1,0])
    metrics[-1, 1]= np.nanmean(metrics[:-1,1])
    return metrics

def _fast_hist(label_true, label_pred, n_class):
    # source https://github.com/Juliachang/SC-CAM
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def score_hist(hist, label_trues, label_preds, n_class):

    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)

    return hist

def compute_mIOU(hist, n_class):

    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    sc = {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,}

    mIoU = np.asarray([v for k, v in sc["Class IoU"].items()])
    mIoU = np.hstack([mIoU, np.nanmean(mIoU)])

    return mIoU


def extract_bboxes(mask):
    ## stolen from mask-RCNN matteport repo
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])
    return boxes.astype(np.int32)


class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]


    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1



def crop_and_pad(img, cropsize):
    h, w, c = img.shape
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = int((w_space + 1)/2)

    else:
        cont_left = int((-w_space+1)/2)

        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = int((h_space+1)/2)

    else:
        cont_top = int((-h_space+1)/2)

        img_top = 0

    final_image = np.zeros((cropsize, cropsize, img.shape[-1]), np.float32)
    final_image[cont_top:cont_top + ch, cont_left:cont_left + cw] = \
        img[img_top:img_top + ch, img_left:img_left + cw]

    return final_image



def AvgPool2d(img,ksize):
    ## source: https://github.com/jiwoon-ahn/psa

    return skimage.measure.block_reduce(img, (ksize, ksize, 1), np.mean)



def ExtractAffinityLabelInRadius(label, cropsize, radius=5):


        search_dist = []

        for x in range(1, radius):
            search_dist.append((0, x))

        for y in range(1, radius):
            for x in range(-radius + 1, radius):
                if x * x + y * y < radius * radius:
                    search_dist.append((y, x))

        radius_floor = radius - 1

        crop_height = cropsize - radius_floor
        crop_width = cropsize - 2 * radius_floor


        labels_from = label[:-radius_floor, radius_floor:-radius_floor]
        labels_from = np.reshape(labels_from, [-1])

        labels_to_list = []
        valid_pair_list = []

        for dy, dx in search_dist:
            labels_to = label[dy:dy+crop_height, radius_floor+dx:radius_floor+dx+crop_width]
            labels_to = np.reshape(labels_to, [-1])

            valid_pair = np.logical_and(np.less(labels_to, 255), np.less(labels_from, 255))

            labels_to_list.append(labels_to)
            valid_pair_list.append(valid_pair)

        bc_labels_from = np.expand_dims(labels_from, 0)
        concat_labels_to = np.stack(labels_to_list)
        concat_valid_pair = np.stack(valid_pair_list)

        pos_affinity_label = np.equal(bc_labels_from, concat_labels_to)

        bg_pos_affinity_label = np.logical_and(pos_affinity_label, np.equal(bc_labels_from, 0)).astype(np.float32)

        fg_pos_affinity_label = np.logical_and(np.logical_and(pos_affinity_label, np.not_equal(bc_labels_from, 0)), concat_valid_pair).astype(np.float32)

        neg_affinity_label = np.logical_and(np.logical_not(pos_affinity_label), concat_valid_pair).astype(np.float32)

        return torch.from_numpy(bg_pos_affinity_label).cuda(), torch.from_numpy(fg_pos_affinity_label).cuda(), torch.from_numpy(neg_affinity_label).cuda()

def convert_bbox(size, box):
    W = size[1]
    H = size[0]
    dW = W * box[2]
    dH = H * box[3]
    w0 = np.int(max(0, np.floor(W * box[0] - dW/2)))
    h0 = np.int(max(0, np.floor(H * box[1] - dH/2)))
    wl = np.int(min(W-1, np.floor(w0 + dW)))
    hl = np.int(min(H-1, np.floor(h0 + dH)))

    return  (w0, h0, wl, hl)

def pad_resize(img,desired_size):

    old_size = img.shape[:2]
    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])
    img = resize(img, new_size, preserve_range=True)

    new_im = np.zeros((desired_size,desired_size,3))
    new_im[int((desired_size - new_size[0]) // 2):(int((desired_size - new_size[0]) // 2)+img.shape[0]),
    int((desired_size - new_size[1]) // 2):(int((desired_size - new_size[1]) // 2)+img.shape[1]),:] = img

    img_window = [int((desired_size - new_size[0]) // 2),(int((desired_size - new_size[0]) // 2)+img.shape[0]),
                  int((desired_size - new_size[1]) // 2), (int((desired_size - new_size[1]) // 2)+img.shape[1])]

    return new_im, img_window

class MMTDataset_baseline(Dataset):

    ### Overwriting some functions of Dataset build in class
    def __init__(self, img_names, labels_dict, voc12_img_folder, input_dim, transform = None):

        self.labels_dict = np.load(labels_dict, allow_pickle=True).item()
        self.input_dim = input_dim
        self.transform = transform


        with open(img_names) as file:
            # self.img_paths = np.asarray([voc12_img_folder + l.rstrip("\n") for l in file])[1000:]
            self.img_paths = np.asarray([voc12_img_folder + l.rstrip("\n") for l in file])

        self.pretrained_mean = np.asarray([123.68, 116.779, 103.939], np.float32) ## mean pixel value of the dataset that our pretrained have been trained with (Imagenet)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):

        current_path = self.img_paths[idx]
        img_orig = plt.imread(current_path)
        img, window = pad_resize(img_orig, self.input_dim)

        W, H = img_orig.shape[:2]

        W_prime = W - int(np.ceil((120 - window[0]) / (window[1] - window[0]) * W))
        ## removing the depicted meta data
        img[:120, :, :] = 0
        window[0] = 120

        img_mean = np.mean(img, 2)

        if self.transform:
            img_mean = img_mean/255
            # Transformer
            img = np.tile(img_mean, (3, 1, 1))
            img = img.astype(np.float32)
            img = self.transform(torch.from_numpy(img).cuda())
            W, H = img_orig.shape[:2]

        else:
            ## VGG
            img_mean = img_mean - np.mean(self.pretrained_mean)

            img = np.tile(img_mean, (3, 1, 1))
            img = img.astype(np.float32)
            img = torch.from_numpy(img).cuda()

        label = torch.from_numpy(np.ndarray.astype(np.asarray(self.labels_dict["/".join(current_path.split("/")[2:])]), np.int64)).cuda()

        ## checking if frame annotated
        gt_mask = np.zeros(img_orig.shape[:2], dtype=np.int32)
        plt.imshow(img_orig[95:])

        if "annotated" in current_path:
            current_path_txt = current_path[:-3]+"txt"

            with open(current_path_txt) as f:
                bbox = f.readlines()

            if len(bbox) > 0:
                bbox = [" ".join(b.split(" ")[1:])[:-1].split(" ") for b in bbox]
                bboxes = [[np.float(b) for b in bbox[i]] for i in range(len(bbox))]
                ## ploting bbox
                bboxes = [convert_bbox( img_orig.shape[:2], bbox) for bbox in bboxes]
                # plt.imshow(img_orig)
                for index in range(len(bbox)):
                    current_box = bboxes[index]
                    rectangle = patches.Rectangle((current_box[0], current_box[1]-95), current_box[2]-current_box[0], current_box[3]-current_box[1], linewidth=1, edgecolor='r', facecolor='none')
                    plt.gca().add_patch(rectangle)
                    gt_mask[current_box[1]:current_box[3], current_box[0]:current_box[2]] = label.data.cpu().numpy()
        gt_mask = gt_mask[-W_prime:,:]
        gt_mask = torch.from_numpy(gt_mask).cuda()

        gt_mask = gt_mask[-W_prime:, :]
        plt.axis('off')
        plt.savefig(str(idx)+".png", bbox_inches='tight')
        plt.close()

        ## constructing img meta data
        img_meta = [{
            "path": current_path,
            "shape": (W_prime,H),
            "window": window,
        }]

        return img_meta, gt_mask, img, label

class MMTDataset_multiview(MMTDataset_baseline):

    ### Overwriting some functions of Dataset build in class
    def __init__(self, img_names, labels_dict, voc12_img_folder, input_dim):
        super(MMTDataset_multiview, self).__init__(img_names, labels_dict, voc12_img_folder, input_dim)


    def __getitem__(self, idx):
        triplet_path = self.img_paths[idx]

        triplet_interior = os.listdir(triplet_path)
        triplet_interior  = [i for i in triplet_interior if i.split(".")[-1] == "jpg" ]

        if triplet_path.split("/")[-1].split("_")[0] != "Clean":
            idx = np.asarray([ int(i.split("HD")[-1][0]) for i in triplet_interior])
        else:
            idx = np.asarray([int(i.split("_")[1]) for i in triplet_interior])

        multiview_images = []
        multiview_labels = []
        multiview_gt_masks = []

        for index in [1,2,3]:


            current_path = triplet_path+"/"+triplet_interior[np.where(idx == index)[0][0]]

            img_orig = plt.imread(current_path)
            img, window = pad_resize(img_orig, self.input_dim)

            W, H = img_orig.shape[:2]
            W_prime = W - int(np.ceil((120 - window[0]) / (window[1] - window[0]) * W))

            img[:120, :, :] = 0

            img_mean = np.mean(img, 2)

            ## Transformer
            # img_mean = (img_mean - 0.5) / 0.5

            ## VGG
            img_mean = img_mean - np.mean(self.pretrained_mean)

            img = np.tile(img_mean, (3, 1, 1))
            img = img.astype(np.float32)

            label = np.asarray(self.labels_dict["/".join(current_path.split("/")[2:])])

            ## checking if frame annotated
            gt_mask = np.zeros(img_orig.shape[:2], dtype=np.int32)
            plt.imshow(img_orig[95:])

            if "annotated" in current_path:
                current_path_txt = current_path[:-3] + "txt"

                with open(current_path_txt) as f:
                    bbox = f.readlines()


                if len(bbox) > 0:
                    bbox = [" ".join(b.split(" ")[1:])[:-1].split(" ") for b in bbox]
                    bboxes = [[np.float(b) for b in bbox[i]] for i in range(len(bbox))]
                    ## ploting bbox
                    # plt.imshow(img_orig)

                    bboxes = [convert_bbox(img_orig.shape[:2], bbox) for bbox in bboxes]
                    for index_b in range(len(bbox)):
                        current_box = bboxes[index_b]
                        rectangle = patches.Rectangle((current_box[0], current_box[1]-95), current_box[2]-current_box[0], current_box[3]-current_box[1], linewidth=1, edgecolor='r', facecolor='none')
                        plt.gca().add_patch(rectangle)
                        gt_mask[current_box[1]:current_box[3], current_box[0]:current_box[2]] = label
            gt_mask = gt_mask[-W_prime:, :]
            plt.axis('off')
            plt.savefig(str(index) + ".png", bbox_inches='tight')
            plt.close()

            multiview_gt_masks.append(torch.from_numpy(gt_mask).unsqueeze(0).cuda())
            multiview_images.append(torch.from_numpy(img).unsqueeze(0).cuda())

            multiview_labels.append(torch.from_numpy(
                np.ndarray.astype(label, np.int64)).unsqueeze(0).cuda())



        ## constructing img meta data
        img_meta = [{
            "shape": (W_prime,H),
            "window": window,
        }]


        imgs = torch.cat(multiview_images)
        multiview_gt_masks = torch.cat(multiview_gt_masks)
        labels = torch.cat(multiview_labels)
        triplet_label = max(labels)

        return img_meta, multiview_gt_masks, imgs, labels, triplet_label

