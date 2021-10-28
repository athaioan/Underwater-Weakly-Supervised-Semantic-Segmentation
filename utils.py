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
    def __init__(self, img_names, labels_dict, voc12_img_folder, input_dim):

        self.labels_dict = np.load(labels_dict, allow_pickle=True).item()
        self.input_dim = input_dim


        with open(img_names) as file:
            self.img_paths = np.asarray([voc12_img_folder + l.rstrip("\n") for l in file])

        self.pretrained_mean = np.asarray([123.68, 116.779, 103.939], np.float32) ## mean pixel value of the dataset that our pretrained have been trained with (Imagenet)


    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        current_path = self.img_paths[idx]

        img_orig = plt.imread(current_path)
        img, window = pad_resize(img_orig, self.input_dim)

        ## removing the depicted meta data
        img[:120, :, :] = 0
        window[0] = 120

        img_mean = np.mean(img, 2)

        ## Transformer
        # img_mean = (img_mean - 0.5) / 0.5

        ## VGG
        img_mean = img_mean - np.mean(self.pretrained_mean)

        img = np.tile(img_mean, (3, 1, 1))
        img = img.astype(np.float32)

        img = torch.from_numpy(img).cuda()

        label = torch.from_numpy(np.ndarray.astype(np.asarray(self.labels_dict["/".join(current_path.split("/")[2:])]), np.int64)).cuda()

        ## checking if frame annotated
        bbox =[]
        if "annotated" in current_path:
            current_path_txt = current_path[:-3]+"txt"

            with open(current_path_txt) as f:
                bbox = f.readlines()

            if len(bbox) > 0:
                bbox = [" ".join(b.split(" ")[1:])[:-1].split(" ") for b in bbox]




        ## constructing img meta data
        img_meta = {
            "path": current_path,
            "shape": img_orig.shape[:2],
            "window": window,
            "bbox": bbox
        }

        return img_meta, img, label

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

        for index in [1,2,3]:


            current_path = triplet_path+"/"+triplet_interior[np.where(idx == index)[0][0]]

            img = plt.imread(current_path)
            img = pad_resize(img, self.input_dim)
            img[:120, :, :] = 0

            img_mean = np.mean(img, 2)

            ## Transformer
            # img_mean = (img_mean - 0.5) / 0.5

            ## VGG
            img_mean = img_mean - np.mean(self.pretrained_mean)

            img = np.tile(img_mean, (3, 1, 1))
            img = img.astype(np.float32)

            multiview_images.append(torch.from_numpy(img).unsqueeze(0).cuda())

            multiview_labels.append(torch.from_numpy(
                np.ndarray.astype(np.asarray(self.labels_dict["/".join(current_path.split("/")[2:])]), np.int64)).unsqueeze(0).cuda())

        imgs = torch.cat(multiview_images)
        labels = torch.cat(multiview_labels)
        triplet_label = max(labels)

        return imgs, labels, triplet_label

