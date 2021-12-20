import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label as measure_segments
from scipy import ndimage
from utils import *

class VGG16(nn.Module):
    def __init__(self, n_classes, fc6_dilation=1):
        super(VGG16, self).__init__()

        self.train_history = {"loss": [],
                              "accuracy": []}
        self.val_history = {"loss": [],
                            "accuracy": []}
        self.min_val = np.inf
        self.n_classes = n_classes

        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=2, dilation=2)
        self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.pool5a = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)

        self.fc6 = nn.Conv2d(512, 1024, 3, padding=fc6_dilation, dilation=fc6_dilation)
        self.drop6 = nn.Dropout2d() # 0.5 dropout

        self.fc7 = nn.Conv2d(1024, 1024, 1)
        self.drop7 = nn.Dropout2d() # 0.5 dropout


        self.fc8 = nn.Conv2d(1024, self.n_classes, 1, bias=False)

        self.fc8 = nn.Conv2d(1024, self.n_classes, 1, bias=False)

        self.fc8_triplet = nn.Conv2d(1024*3, self.n_classes, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc8.weight)
        torch.nn.init.xavier_uniform_(self.fc8_triplet.weight)
        self.from_scratch_layers = [self.fc8, self.fc8_triplet]

        return

    def feature_extractor(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool1(x)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool2(x)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool3(x)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        conv4 = x
        x = self.pool4(x)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        conv5 = x

        x = F.relu(self.fc6(x))
        x = self.drop6(x)
        x = F.relu(self.fc7(x))
        conv5fc = x

        return conv5fc

    def forward(self, x):

        x1 = self.feature_extractor(x)

        x1 = self.drop7(x1)
        x1 = self.fc8(x1)
        x1 = F.avg_pool2d(x1, kernel_size=(x1.size(2), x1.size(3)), padding=0)
        x1 = x1.view(-1, self.n_classes)

        return x1


    def cam_output(self, x):

        x = self.feature_extractor(x)
        x = self.fc8(x) ## fc8 was set with bias=False and this is why applying the fc8 is like array multiplication as intructed in equation (1) of paper (1)
        x = F.relu(x)
        x = torch.sqrt(x) ## smoothed by square rooting to obtain a more uniform cam visualization
        return x

    def freeze_layers(self, frozen_stages):

        for layer in self.named_parameters():
            if "conv" in layer[0] and np.int(layer[0].split("conv")[-1][0]) in frozen_stages:
                layer[1].requires_grad = False

    def load_pretrained(self, pth_file):

        weights_dict = torch.load(pth_file)
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in weights_dict.items() if
                           k in model_dict and weights_dict[k].shape == model_dict[k].shape}
        #
        # no_pretrained_dict = {k: v for k, v in model_dict.items() if
        #                    not (k in weights_dict) or weights_dict[k].shape != model_dict[k].shape}

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)

        return


    def train_epoch(self, dataloader, optimizer, criterion,verbose=True):

        train_loss = 0
        train_accuracy = 0
        self.train()

        for index,data in enumerate(dataloader):


            img = data[2]
            label = data[3]

            x =  self(img)

            loss = criterion(x, label)
            loss = loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### adding batch loss into the overall loss
            train_loss += loss.item()

            preds = torch.argmax(x, 1)
            ### adding batch loss into the overall loss
            batch_accuracy = sum(preds == label) / x.shape[0]
            train_accuracy += batch_accuracy

            if verbose:
                ### Printing epoch results
                print('Train Epoch: {}/{}\n'
                      'Step: {}/{}\n'
                      'Batch ~ Loss: {:.4f}\n'
                      'Batch ~ Accuracy: {:.4f}\n'.format(self.epoch + 1, self.epochs,
                                                          index + 1, len(dataloader),
                                                          loss.data.cpu().numpy(),
                                                          batch_accuracy.data.cpu().numpy()))
            torch.cuda.empty_cache()

        self.train_history["loss"].append(train_loss / (index + 1))
        self.train_history["accuracy"].append(train_accuracy / (index + 1))

        return


    def val_epoch(self, dataloader,criterion, verbose=True):

           val_loss = 0
           val_accuracy = 0
           self.eval()

           conf_matrix = np.zeros((18,18))
           with torch.no_grad():

               for index, data in enumerate(dataloader):


                   img = data[2]
                   label = data[3]

                   x = self(img)
                   loss =criterion(x, label)

                   ### adding batch loss into the overall loss
                   val_loss += loss.item()

                   preds = torch.argmax(x, 1)
                   ### adding batch loss into the overall loss
                   batch_accuracy = sum(preds == label) / x.shape[0]
                   val_accuracy += batch_accuracy

                   preds = preds.data.cpu().numpy()
                   labels = label.data.cpu().numpy()

                   for p_index in range(preds.shape[0]):
                       conf_matrix[labels[p_index], preds[p_index]]+=1


                   if verbose:
                       ### Printing epoch results
                       print('Val Epoch: {}/{}\n'
                             'Step: {}/{}\n'
                             'Batch ~ Loss: {:.4f}\n'
                             'Batch ~ Accuracy: {:.4f}\n'.format(self.epoch + 1, self.epochs,
                                                                 index + 1, len(dataloader),
                                                                 loss.data.cpu().numpy(),
                                                                 batch_accuracy.data.cpu().numpy()))
                   torch.cuda.empty_cache()

               self.val_history["loss"].append(val_loss / len(dataloader))
               self.val_history["accuracy"].append(val_accuracy / len(dataloader))


           return conf_matrix

    def evaluate(self,dataloader):


        self.eval()

        conf_matrix = np.zeros((self.n_classes, self.n_classes))

        hist = np.zeros((self.n_classes, self.n_classes))

        with torch.no_grad():



            for index, data in enumerate(dataloader):


                img_meta = data[0][0] # img_meta identical for all samples
                gt_mask = data[1]
                img = data[2]
                label = data[3]
                x = self(img)

                preds = torch.argmax(x, 1).data.cpu().numpy()


                ## all input image have identical width and height
                X_min = img_meta["window"][0][0].data.cpu().numpy()
                X_max =  img_meta["window"][1][0].data.cpu().numpy()
                Y_min = img_meta["window"][2][0].data.cpu().numpy()
                Y_max =  img_meta["window"][3][0].data.cpu().numpy()

                X = img_meta['shape'][0][0].data.cpu().numpy()
                Y = img_meta['shape'][1][0].data.cpu().numpy()

                ## extracting CAM explainability cues
                img_cam = img[:,:,X_min:X_max,Y_min:Y_max]
                x = self.cam_output(img_cam)
                x = F.upsample(x, [X, Y], mode='bilinear', align_corners=False)

                # resizing label to (batch , 1, 240, 320)
                label_idx = label.view(-1, 1, 1, 1).repeat((1, 1) + x.size()[-2:])

                cams = torch.gather(x, 1, label_idx).squeeze()

                normalizer = torch.amax(cams, (1, 2))
                normalizer += (normalizer == 0)*1 ## avoid dividing by zero
                cams = cams / normalizer.view(-1,1,1)

                ## thresholding cams
                cams = cams*label.view(-1, 1, 1)

                ## insterting
                for current_index in range(cams.shape[0]):
                    current_label = label[current_index].data.cpu().numpy()
                    if current_label>0:
                        current_cam = cams[current_index].data.cpu().numpy()
                        current_cam = (current_cam > 0.4)

                        current_gt_mask = gt_mask[current_index].data.cpu().numpy()
                        current_pred_mask = np.zeros_like(current_gt_mask)

                        label_im, nb_labels = ndimage.label(current_cam)
                        if nb_labels > 0:
                            ## catching the case of full zero cams
                            ## extracting bboxes
                            masks = [label_im==(i+1) for i in range(nb_labels)]
                            masks = np.asarray(masks).transpose(1,2,0)
                            bboxes = extract_bboxes(masks)
                            for i in range(nb_labels):
                                current_bbox = bboxes[i,:]
                                current_pred_mask[current_bbox[0]:current_bbox[2],
                                current_bbox[1]:current_bbox[3]] = current_label

                            hist = score_hist(hist, current_gt_mask, current_pred_mask, self.n_classes)

                labels = label.data.cpu().numpy()
                for p_index in range(preds.shape[0]):
                    conf_matrix[labels[p_index], preds[p_index]] += 1

                print('Step: {}/{}\n'.format(index + 1, len(dataloader)))
            mIoU = compute_mIOU(hist,self.n_classes)

        return conf_matrix, mIoU


    def visualize_graph(self):

        ## Plotting loss
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Graph")

        plt.plot(np.arange(len(self.train_history["loss"])), self.train_history["loss"], label="train")
        plt.plot(np.arange(len(self.val_history["loss"])), self.val_history["loss"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"loss.png")
        plt.close()


        ## Plotting accyracy
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Graph")

        plt.plot(np.arange(len(self.train_history["accuracy"])), self.train_history["accuracy"], label="train")
        plt.plot(np.arange(len(self.val_history["accuracy"])), self.val_history["accuracy"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"accuracy.png")
        plt.close()

    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)):

                if m.weight is not None and m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


class VGG16_multiview(VGG16):

    def __init__(self, n_classes, fc6_dilation=1):
        super(VGG16_multiview, self).__init__(n_classes, fc6_dilation=fc6_dilation)

        self.fc8_triplet = nn.Conv2d(1024*3, self.n_classes, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fc8_triplet.weight)

        self.from_scratch_layers.append(self.fc8_triplet)

        return

    def forward(self, x):

        feat = self.feature_extractor(x)
        x1 = self.drop7(feat)
        x1 = self.fc8(x1)
        x1 = F.avg_pool2d(x1, kernel_size=(x1.size(2), x1.size(3)), padding=0)
        x1 = x1.view(-1, self.n_classes)

        x2 = self.drop7(feat)
        x2 = x2.view(x2.shape[0] // 3,3*x2.shape[1],x2.shape[2],x2.shape[3])
        x2 = self.fc8_triplet(x2)
        x2 = F.avg_pool2d(x2, kernel_size=(x2.size(2), x2.size(3)), padding=0)
        x2 = x2.view(-1, self.n_classes)

        return x1, x2

    def train_epoch(self, dataloader, optimizer, criterion, verbose=True):

        train_loss = 0
        train_accuracy = 0
        self.train()

        for index,data in enumerate(dataloader):


            imgs = data[2]
            labels = data[3]
            triplet_label = data[4]

            [batch,m,c,M,N] = imgs.size()

            imgs = imgs.view(batch*m, c, M, N)
            labels = labels.view(batch*m)


            x, x_triplet =  self(imgs)

            loss = criterion(x, labels)
            loss_triplet = criterion(x_triplet, triplet_label)

            loss = 2/3*loss + 1/3*loss_triplet

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### adding batch loss into the overall loss
            train_loss += loss.item()

            preds = torch.argmax(x, 1)
            ### adding batch loss into the overall loss
            batch_accuracy = sum(preds == labels) / x.shape[0]
            train_accuracy += batch_accuracy

            if verbose:
                ### Printing epoch results
                print('Train Epoch: {}/{}\n'
                      'Step: {}/{}\n'
                      'Batch ~ Loss: {:.4f}\n'
                      'Batch ~ Accuracy: {:.4f}\n'.format(self.epoch + 1, self.epochs,
                                                          index + 1, len(dataloader),
                                                          loss.data.cpu().numpy(),
                                                          batch_accuracy.data.cpu().numpy()))
            torch.cuda.empty_cache()

        self.train_history["loss"].append(train_loss / (index + 1))
        self.train_history["accuracy"].append(train_accuracy / (index + 1))

        return

    def val_epoch(self, dataloader,criterion, verbose=True):

           val_loss = 0
           val_accuracy = 0
           self.eval()

           conf_matrix = np.zeros((18,18))
           with torch.no_grad():

               for index, data in enumerate(dataloader):

                   imgs = data[2]
                   labels = data[3]

                   [batch, m, c, M, N] = imgs.size()

                   imgs = imgs.view(batch * m, c, M, N)
                   labels = labels.view(batch * m)

                   x, _ = self(imgs)
                   loss = criterion(x, labels)

                   ### adding batch loss into the overall loss
                   val_loss += loss.item()

                   preds = torch.argmax(x, 1)
                   ### adding batch loss into the overall loss
                   batch_accuracy = sum(preds == labels) / x.shape[0]
                   val_accuracy += batch_accuracy

                   preds = preds.data.cpu().numpy()
                   labels = labels.data.cpu().numpy()

                   for p_index in range(preds.shape[0]):
                       conf_matrix[labels[p_index], preds[p_index]]+=1


                   if verbose:
                       ### Printing epoch results
                       print('Val Epoch: {}/{}\n'
                             'Step: {}/{}\n'
                             'Batch ~ Loss: {:.4f}\n'
                             'Batch ~ Accuracy: {:.4f}\n'.format(self.epoch + 1, self.epochs,
                                                                 index + 1, len(dataloader),
                                                                 loss.data.cpu().numpy(),
                                                                 batch_accuracy.data.cpu().numpy()))
                   torch.cuda.empty_cache()

               self.val_history["loss"].append(val_loss / len(dataloader))
               self.val_history["accuracy"].append(val_accuracy / len(dataloader))


           return conf_matrix

    def evaluate(self, dataloader):

        self.eval()

        conf_matrix = np.zeros((self.n_classes, self.n_classes))

        hist = np.zeros((self.n_classes, self.n_classes))

        with torch.no_grad():

            for index, data in enumerate(dataloader):


                img_meta = data[0][0]  # img_meta identical for all samples
                gt_mask = data[1]
                imgs = data[2]
                labels = data[3]

                [batch, m, c, M, N] = imgs.size()

                imgs = imgs.view(batch * m, c, M, N)
                gt_mask = gt_mask.squeeze(0)
                labels = labels.view(batch * m)

                x, _ = self(imgs)

                preds = torch.argmax(x, 1).data.cpu().numpy()

                ## all input image have identical width and height
                X_min = img_meta["window"][0][0].data.cpu().numpy()
                X_max = img_meta["window"][1][0].data.cpu().numpy()
                Y_min = img_meta["window"][2][0].data.cpu().numpy()
                Y_max = img_meta["window"][3][0].data.cpu().numpy()

                X = img_meta['shape'][0][0].data.cpu().numpy()
                Y = img_meta['shape'][1][0].data.cpu().numpy()

                ## extracting CAM explainability cues
                img_cam = imgs[:, :, X_min:X_max, Y_min:Y_max]
                x = self.cam_output(img_cam)
                x = F.upsample(x, [X, Y], mode='bilinear', align_corners=False)

                # resizing label to (batch , 1, 240, 320)
                label_idx = labels.view(-1, 1, 1, 1).repeat((1, 1) + x.size()[-2:])

                cams = torch.gather(x, 1, label_idx).squeeze()

                normalizer = torch.amax(cams, (1, 2))
                normalizer += (normalizer == 0) * 1  ## avoid dividing by zero
                cams = cams / normalizer.view(-1, 1, 1)

                ## thresholding cams
                cams = (cams > 0.4) * labels.view(-1, 1, 1)

                ## insterting
                for current_index in range(cams.shape[0]):
                    current_label = labels[current_index].data.cpu().numpy()
                    if current_label > 0:
                        current_cam = cams[current_index].data.cpu().numpy()
                        current_gt_mask = gt_mask[current_index].data.cpu().numpy()
                        current_pred_mask = np.zeros_like(current_gt_mask)

                        label_im, nb_labels = ndimage.label(current_cam)
                        if nb_labels > 0:
                            ## catching the case of full zero cams
                            ## extracting bboxes
                            masks = [label_im == (i + 1) for i in range(nb_labels)]
                            masks = np.asarray(masks).transpose(1, 2, 0)
                            bboxes = extract_bboxes(masks)
                            for i in range(nb_labels):
                                current_bbox = bboxes[i, :]
                                current_pred_mask[current_bbox[0]:current_bbox[2],
                                current_bbox[1]:current_bbox[3]] = current_label

                            hist = score_hist(hist, current_gt_mask, current_pred_mask, self.n_classes)

                labels = labels.data.cpu().numpy()
                for p_index in range(preds.shape[0]):
                    conf_matrix[labels[p_index], preds[p_index]] += 1

                print('Step: {}/{}\n'.format(index + 1, len(dataloader)))
            mIoU = compute_mIOU(hist, self.n_classes)

        return conf_matrix, mIoU

    def visualize_graph(self):

        ## Plotting loss
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Graph")

        plt.plot(np.arange(len(self.train_history["loss"])), self.train_history["loss"], label="train")
        plt.plot(np.arange(len(self.val_history["loss"])), self.val_history["loss"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"loss.png")
        plt.close()


        ## Plotting accyracy
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Graph")

        plt.plot(np.arange(len(self.train_history["accuracy"])), self.train_history["accuracy"], label="train")
        plt.plot(np.arange(len(self.val_history["accuracy"])), self.val_history["accuracy"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"accuracy.png")
        plt.close()



    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)):

                if m.weight is not None and m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


class VGG16_multiview_plus(VGG16):

    def __init__(self, n_classes, fc6_dilation=1):
        super(VGG16_multiview_plus, self).__init__(n_classes, fc6_dilation=fc6_dilation)

        self.fc8_triplet = nn.Conv2d(1024*3, self.n_classes, 1, bias=False)
        self.fc8_left = nn.Conv2d(1024*3, self.n_classes, 1, bias=False)
        self.fc8_centre = nn.Conv2d(1024*3, self.n_classes, 1, bias=False)
        self.fc8_right = nn.Conv2d(1024*3, self.n_classes, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc8_triplet.weight)
        torch.nn.init.xavier_uniform_(self.fc8_left.weight)
        torch.nn.init.xavier_uniform_(self.fc8_centre.weight)
        torch.nn.init.xavier_uniform_(self.fc8_right.weight)

        self.from_scratch_layers.append(self.fc8_triplet)
        self.from_scratch_layers.append(self.fc8_left)
        self.from_scratch_layers.append(self.fc8_centre)
        self.from_scratch_layers.append(self.fc8_right)
        return

    def cam_output(self, x):

        x = self.feature_extractor(x)
        x = x.view(x.shape[0] // 3,3*x.shape[1],x.shape[2],x.shape[3])

        x_left = self.fc8_left(x) ## fc8 was set with bias=False and this is why applying the fc8 is like array multiplication as intructed in equation (1) of paper (1)
        x_centre = self.fc8_centre(x) ## fc8 was set with bias=False and this is why applying the fc8 is like array multiplication as intructed in equation (1) of paper (1)
        x_right = self.fc8_right(x) ## fc8 was set with bias=False and this is why applying the fc8 is like array multiplication as intructed in equation (1) of paper (1)
        x = torch.cat((x_left, x_centre, x_right ), dim=0)
        x = F.relu(x)
        x = torch.sqrt(x) ## smoothed by square rooting to obtain a more uniform cam visualization
        return x


    def forward(self, x):

        feat = self.feature_extractor(x)

        x1 = feat
        x1 = x1.view(x1.shape[0] // 3,3*x1.shape[1],x1.shape[2],x1.shape[3])
        x1 = self.drop7(x1)
        x1_left = self.fc8_left(x1)
        x1_centre = self.fc8_centre(x1)
        x1_right = self.fc8_right(x1)
        x1 = torch.cat((x1_left, x1_centre, x1_right ), dim=0)
        x1 = F.avg_pool2d(x1, kernel_size=(x1.size(2), x1.size(3)), padding=0)
        x1 = x1.view(-1, self.n_classes)

        x2 = self.drop7(feat)
        x2 = x2.view(x2.shape[0] // 3,3*x2.shape[1],x2.shape[2],x2.shape[3])
        x2 = self.fc8_triplet(x2)
        x2 = F.avg_pool2d(x2, kernel_size=(x2.size(2), x2.size(3)), padding=0)
        x2 = x2.view(-1, self.n_classes)

        return x1, x2

    def train_epoch(self, dataloader, optimizer, criterion, verbose=True):

        train_loss = 0
        train_accuracy = 0
        self.train()

        for index,data in enumerate(dataloader):


            imgs = data[2]
            labels = data[3]
            triplet_label = data[4]

            [batch,m,c,M,N] = imgs.size()

            imgs = imgs.view(batch*m, c, M, N)
            labels = labels.view(batch*m)


            x, x_triplet =  self(imgs)

            loss = criterion(x, labels)
            loss_triplet = criterion(x_triplet, triplet_label)

            loss = 2/3*loss + 1/3*loss_triplet

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### adding batch loss into the overall loss
            train_loss += loss.item()

            preds = torch.argmax(x, 1)
            ### adding batch loss into the overall loss
            batch_accuracy = sum(preds == labels) / x.shape[0]
            train_accuracy += batch_accuracy

            if verbose:
                ### Printing epoch results
                print('Train Epoch: {}/{}\n'
                      'Step: {}/{}\n'
                      'Batch ~ Loss: {:.4f}\n'
                      'Batch ~ Accuracy: {:.4f}\n'.format(self.epoch + 1, self.epochs,
                                                          index + 1, len(dataloader),
                                                          loss.data.cpu().numpy(),
                                                          batch_accuracy.data.cpu().numpy()))
            torch.cuda.empty_cache()

        self.train_history["loss"].append(train_loss / (index + 1))
        self.train_history["accuracy"].append(train_accuracy / (index + 1))

        return

    def val_epoch(self, dataloader,criterion, verbose=True):

           val_loss = 0
           val_accuracy = 0
           self.eval()

           conf_matrix = np.zeros((18,18))
           with torch.no_grad():

               for index, data in enumerate(dataloader):

                   imgs = data[2]
                   labels = data[3]

                   [batch, m, c, M, N] = imgs.size()

                   imgs = imgs.view(batch * m, c, M, N)
                   labels = labels.view(batch * m)

                   x, _ = self(imgs)
                   loss = criterion(x, labels)

                   ### adding batch loss into the overall loss
                   val_loss += loss.item()

                   preds = torch.argmax(x, 1)
                   ### adding batch loss into the overall loss
                   batch_accuracy = sum(preds == labels) / x.shape[0]
                   val_accuracy += batch_accuracy

                   preds = preds.data.cpu().numpy()
                   labels = labels.data.cpu().numpy()

                   for p_index in range(preds.shape[0]):
                       conf_matrix[labels[p_index], preds[p_index]]+=1


                   if verbose:
                       ### Printing epoch results
                       print('Val Epoch: {}/{}\n'
                             'Step: {}/{}\n'
                             'Batch ~ Loss: {:.4f}\n'
                             'Batch ~ Accuracy: {:.4f}\n'.format(self.epoch + 1, self.epochs,
                                                                 index + 1, len(dataloader),
                                                                 loss.data.cpu().numpy(),
                                                                 batch_accuracy.data.cpu().numpy()))
                   torch.cuda.empty_cache()

               self.val_history["loss"].append(val_loss / len(dataloader))
               self.val_history["accuracy"].append(val_accuracy / len(dataloader))


           return conf_matrix

    def evaluate(self, dataloader):

        self.eval()

        conf_matrix = np.zeros((self.n_classes, self.n_classes))

        hist = np.zeros((self.n_classes, self.n_classes))

        with torch.no_grad():

            for index, data in enumerate(dataloader):


                img_meta = data[0][0]  # img_meta identical for all samples
                gt_mask = data[1]
                imgs = data[2]
                labels = data[3]

                [batch, m, c, M, N] = imgs.size()

                imgs = imgs.view(batch * m, c, M, N)
                gt_mask = gt_mask.squeeze(0)
                labels = labels.view(batch * m)

                x, _ = self(imgs)

                preds = torch.argmax(x, 1).data.cpu().numpy()

                ## all input image have identical width and height
                X_min = img_meta["window"][0][0].data.cpu().numpy()
                X_max = img_meta["window"][1][0].data.cpu().numpy()
                Y_min = img_meta["window"][2][0].data.cpu().numpy()
                Y_max = img_meta["window"][3][0].data.cpu().numpy()

                X = img_meta['shape'][0][0].data.cpu().numpy()
                Y = img_meta['shape'][1][0].data.cpu().numpy()

                ## extracting CAM explainability cues
                img_cam = imgs[:, :, X_min:X_max, Y_min:Y_max]
                x = self.cam_output(img_cam)
                x = F.upsample(x, [X, Y], mode='bilinear', align_corners=False)

                # resizing label to (batch , 1, 240, 320)
                label_idx = labels.view(-1, 1, 1, 1).repeat((1, 1) + x.size()[-2:])

                cams = torch.gather(x, 1, label_idx).squeeze()

                normalizer = torch.amax(cams, (1, 2))
                normalizer += (normalizer == 0) * 1  ## avoid dividing by zero
                cams = cams / normalizer.view(-1, 1, 1)

                ## thresholding cams
                cams = (cams > 0.4) * labels.view(-1, 1, 1)

                ## insterting
                for current_index in range(cams.shape[0]):
                    current_label = labels[current_index].data.cpu().numpy()
                    if current_label > 0:
                        current_cam = cams[current_index].data.cpu().numpy()
                        current_gt_mask = gt_mask[current_index].data.cpu().numpy()
                        current_pred_mask = np.zeros_like(current_gt_mask)

                        label_im, nb_labels = ndimage.label(current_cam)
                        if nb_labels > 0:
                            ## catching the case of full zero cams
                            ## extracting bboxes
                            masks = [label_im == (i + 1) for i in range(nb_labels)]
                            masks = np.asarray(masks).transpose(1, 2, 0)
                            bboxes = extract_bboxes(masks)
                            for i in range(nb_labels):
                                current_bbox = bboxes[i, :]
                                current_pred_mask[current_bbox[0]:current_bbox[2],
                                current_bbox[1]:current_bbox[3]] = current_label

                            hist = score_hist(hist, current_gt_mask, current_pred_mask, self.n_classes)

                labels = labels.data.cpu().numpy()
                for p_index in range(preds.shape[0]):
                    conf_matrix[labels[p_index], preds[p_index]] += 1

                print('Step: {}/{}\n'.format(index + 1, len(dataloader)))
            mIoU = compute_mIOU(hist, self.n_classes)

        return conf_matrix, mIoU

    def visualize_graph(self):

        ## Plotting loss
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Graph")

        plt.plot(np.arange(len(self.train_history["loss"])), self.train_history["loss"], label="train")
        plt.plot(np.arange(len(self.val_history["loss"])), self.val_history["loss"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"loss.png")
        plt.close()


        ## Plotting accyracy
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Graph")

        plt.plot(np.arange(len(self.train_history["accuracy"])), self.train_history["accuracy"], label="train")
        plt.plot(np.arange(len(self.val_history["accuracy"])), self.val_history["accuracy"], label="val")

        plt.legend()
        plt.savefig(self.session_name+"accuracy.png")
        plt.close()



    def get_parameter_groups(self):
        groups = ([], [], [], [])

        for m in self.modules():

            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d)):

                if m.weight is not None and m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        return groups


class VGG16_binary_aux(VGG16):

    def __init__(self, n_classes, fc6_dilation=1):
        super(VGG16_binary_aux, self).__init__(n_classes, fc6_dilation=fc6_dilation)

        self.fc8_binary = nn.Conv2d(1024, 2, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc8_binary.weight)

        self.from_scratch_layers.append(self.fc8_binary)

        return

    def forward(self, x):

        feat = self.feature_extractor(x)

        x1 = self.drop7(feat)
        x1 = self.fc8(x1)
        x1 = F.avg_pool2d(x1, kernel_size=(x1.size(2), x1.size(3)), padding=0)
        x1 = x1.view(-1, self.n_classes)

        x2 = self.drop7(feat)
        x2 = self.fc8_binary(x2)
        x2 = F.avg_pool2d(x2, kernel_size=(x2.size(2), x2.size(3)), padding=0)
        x2 = x2.view(-1, 2)


        return x1, x2


    def train_epoch(self, dataloader, optimizer, criterion, criterion_binary,verbose=True):

        train_loss = 0
        train_accuracy = 0
        self.train()

        for index,data in enumerate(dataloader):

            img = data[2]
            label = data[3]
            binary_label = (label!=0)*1
            x, x_binary =  self(img)

            loss = criterion(x, label)
            loss_binary = criterion_binary(x_binary, binary_label)
            loss = 2/3*loss + 1/3*loss_binary

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### adding batch loss into the overall loss
            train_loss += loss.item()

            preds = torch.argmax(x, 1)
            ### adding batch loss into the overall loss
            batch_accuracy = sum(preds == label) / x.shape[0]
            train_accuracy += batch_accuracy

            if verbose:
                ### Printing epoch results
                print('Train Epoch: {}/{}\n'
                      'Step: {}/{}\n'
                      'Batch ~ Loss: {:.4f}\n'
                      'Batch ~ Accuracy: {:.4f}\n'.format(self.epoch + 1, self.epochs,
                                                          index + 1, len(dataloader),
                                                          loss.data.cpu().numpy(),
                                                          batch_accuracy.data.cpu().numpy()))
            torch.cuda.empty_cache()

        self.train_history["loss"].append(train_loss / (index + 1))
        self.train_history["accuracy"].append(train_accuracy / (index + 1))

        return

    def val_epoch(self, dataloader,criterion, verbose=True):

           val_loss = 0
           val_accuracy = 0
           self.eval()

           conf_matrix = np.zeros((18,18))
           with torch.no_grad():

               for index, data in enumerate(dataloader):

                   img = data[2]
                   label = data[3]

                   x,_ = self(img)
                   loss =criterion(x, label)

                   ### adding batch loss into the overall loss
                   val_loss += loss.item()

                   preds = torch.argmax(x, 1)
                   ### adding batch loss into the overall loss
                   batch_accuracy = sum(preds == label) / x.shape[0]
                   val_accuracy += batch_accuracy

                   preds = preds.data.cpu().numpy()
                   labels = label.data.cpu().numpy()

                   for p_index in range(preds.shape[0]):
                       conf_matrix[labels[p_index], preds[p_index]]+=1


                   if verbose:
                       ### Printing epoch results
                       print('Val Epoch: {}/{}\n'
                             'Step: {}/{}\n'
                             'Batch ~ Loss: {:.4f}\n'
                             'Batch ~ Accuracy: {:.4f}\n'.format(self.epoch + 1, self.epochs,
                                                                 index + 1, len(dataloader),
                                                                 loss.data.cpu().numpy(),
                                                                 batch_accuracy.data.cpu().numpy()))
                   torch.cuda.empty_cache()

               self.val_history["loss"].append(val_loss / len(dataloader))
               self.val_history["accuracy"].append(val_accuracy / len(dataloader))


           return conf_matrix


    def evaluate(self,dataloader):


        self.eval()

        conf_matrix = np.zeros((self.n_classes, self.n_classes))

        hist = np.zeros((self.n_classes, self.n_classes))

        with torch.no_grad():



            for index, data in enumerate(dataloader):



                img_meta = data[0][0] # img_meta identical for all samples
                gt_mask = data[1]
                img = data[2]
                label = data[3]
                x, _ = self(img)

                preds = torch.argmax(x, 1).data.cpu().numpy()


                ## all input image have identical width and height
                X_min = img_meta["window"][0][0].data.cpu().numpy()
                X_max =  img_meta["window"][1][0].data.cpu().numpy()
                Y_min = img_meta["window"][2][0].data.cpu().numpy()
                Y_max =  img_meta["window"][3][0].data.cpu().numpy()

                X = img_meta['shape'][0][0].data.cpu().numpy()
                Y = img_meta['shape'][1][0].data.cpu().numpy()

                ## extracting CAM explainability cues
                img_cam = img[:,:,X_min:X_max,Y_min:Y_max]
                x = self.cam_output(img_cam)
                x = F.upsample(x, [X, Y], mode='bilinear', align_corners=False)

                # resizing label to (batch , 1, 240, 320)
                label_idx = label.view(-1, 1, 1, 1).repeat((1, 1) + x.size()[-2:])

                cams = torch.gather(x, 1, label_idx).squeeze()

                normalizer = torch.amax(cams, (1, 2))
                normalizer += (normalizer == 0)*1 ## avoid dividing by zero
                cams = cams / normalizer.view(-1,1,1)

                ## thresholding cams
                cams = (cams>0.4)*label.view(-1, 1, 1)

                ## insterting
                for current_index in range(cams.shape[0]):
                    current_label = label[current_index].data.cpu().numpy()
                    if current_label>0:
                        current_cam = cams[current_index].data.cpu().numpy()
                        current_gt_mask = gt_mask[current_index].data.cpu().numpy()
                        current_pred_mask = np.zeros_like(current_gt_mask)

                        label_im, nb_labels = ndimage.label(current_cam)
                        if nb_labels > 0:
                            ## catching the case of full zero cams
                            ## extracting bboxes
                            masks = [label_im==(i+1) for i in range(nb_labels)]
                            masks = np.asarray(masks).transpose(1,2,0)
                            bboxes = extract_bboxes(masks)
                            for i in range(nb_labels):
                                current_bbox = bboxes[i,:]
                                current_pred_mask[current_bbox[0]:current_bbox[2],
                                current_bbox[1]:current_bbox[3]] = current_label

                            hist = score_hist(hist, current_gt_mask, current_pred_mask, self.n_classes)

                labels = label.data.cpu().numpy()
                for p_index in range(preds.shape[0]):
                    conf_matrix[labels[p_index], preds[p_index]] += 1

                print('Step: {}/{}\n'.format(index + 1, len(dataloader)))
            mIoU = compute_mIOU(hist,self.n_classes)

        return conf_matrix, mIoU

class VGG16_binary_aux_perb(VGG16):

    def __init__(self, n_classes, fc6_dilation=1):
        super(VGG16_binary_aux_perb, self).__init__(n_classes, fc6_dilation=fc6_dilation)

        self.fc8_binary = nn.Conv2d(1024, 2, 1, bias=False)

        torch.nn.init.xavier_uniform_(self.fc8_binary.weight)

        self.from_scratch_layers.append(self.fc8_binary)

        return

    def forward(self, x):

        feat = self.feature_extractor(x)

        x1 = self.drop7(feat)
        x1 = self.fc8(x1)
        x1 = F.avg_pool2d(x1, kernel_size=(x1.size(2), x1.size(3)), padding=0)
        x1 = x1.view(-1, self.n_classes)

        x2 = self.drop7(feat)
        x2 = self.fc8_binary(x2)
        x2 = F.avg_pool2d(x2, kernel_size=(x2.size(2), x2.size(3)), padding=0)
        x2 = x2.view(-1, 2)


        return x1, x2


    def train_epoch(self, dataloader, optimizer, criterion, criterion_binary,verbose=True):

        train_loss = 0
        train_accuracy = 0
        self.train()

        for index,data in enumerate(dataloader):

            img = data[2]
            label = data[3]
            binary_label = (label!=0)*1
            x, x_binary =  self(img)


            ## extracting cams
            label_cam = label.clone()
            label_cam[label_cam == 0] = torch.argmax(x, dim=1)[label_cam == 0]

            x_cam = self.cam_output(img)
            x_cam = F.upsample(x_cam, [img.shape[-1], img.shape[-2]], mode='bilinear', align_corners=False)

            label_cam_idx = label_cam.view(-1, 1, 1, 1).repeat((1, 1) + x_cam.size()[-2:])
            cams_perb = torch.gather(x_cam, 1, label_cam_idx ).squeeze(1)

            perb_idx = torch.amax(cams_perb, dim=(1, 2))

            perb_idx = perb_idx.view(-1, 1, 1, 1).repeat((1, 1) + x_cam.size()[-2:]).squeeze(1)

            perb_idx = cams_perb > 0.2*perb_idx
            perb_idx = perb_idx.unsqueeze(1)
            perb_idx = perb_idx.repeat(1,3,1,1)

            img[perb_idx] = torch.min(img) ## all images have identical min values


            x_perb, x_binary_perb =  self(img)


            loss = criterion(x, label)
            loss_x_perb = criterion(x_perb, torch.zeros_like(label))

            loss_binary = criterion_binary(x_binary, binary_label)
            loss_binary_perb = criterion_binary(x_binary_perb, torch.zeros_like(binary_label))

            loss = 2/6*(loss+loss_x_perb) + 1/6*(loss_binary+loss_binary_perb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ### adding batch loss into the overall loss
            train_loss += loss.item()

            preds = torch.argmax(x, 1)
            ### adding batch loss into the overall loss
            batch_accuracy = sum(preds == label) / x.shape[0]
            train_accuracy += batch_accuracy

            if verbose:
                ### Printing epoch results
                print('Train Epoch: {}/{}\n'
                      'Step: {}/{}\n'
                      'Batch ~ Loss: {:.4f}\n'
                      'Batch ~ Accuracy: {:.4f}\n'.format(self.epoch + 1, self.epochs,
                                                          index + 1, len(dataloader),
                                                          loss.data.cpu().numpy(),
                                                          batch_accuracy.data.cpu().numpy()))
            torch.cuda.empty_cache()

        self.train_history["loss"].append(train_loss / (index + 1))
        self.train_history["accuracy"].append(train_accuracy / (index + 1))

        return

    def val_epoch(self, dataloader,criterion, verbose=True):

           val_loss = 0
           val_accuracy = 0
           self.eval()

           conf_matrix = np.zeros((18,18))
           with torch.no_grad():

               for index, data in enumerate(dataloader):

                   img = data[2]
                   label = data[3]

                   x,_ = self(img)
                   loss =criterion(x, label)

                   ### adding batch loss into the overall loss
                   val_loss += loss.item()

                   preds = torch.argmax(x, 1)
                   ### adding batch loss into the overall loss
                   batch_accuracy = sum(preds == label) / x.shape[0]
                   val_accuracy += batch_accuracy

                   preds = preds.data.cpu().numpy()
                   labels = label.data.cpu().numpy()

                   for p_index in range(preds.shape[0]):
                       conf_matrix[labels[p_index], preds[p_index]]+=1


                   if verbose:
                       ### Printing epoch results
                       print('Val Epoch: {}/{}\n'
                             'Step: {}/{}\n'
                             'Batch ~ Loss: {:.4f}\n'
                             'Batch ~ Accuracy: {:.4f}\n'.format(self.epoch + 1, self.epochs,
                                                                 index + 1, len(dataloader),
                                                                 loss.data.cpu().numpy(),
                                                                 batch_accuracy.data.cpu().numpy()))
                   torch.cuda.empty_cache()

               self.val_history["loss"].append(val_loss / (index + 1))
               self.val_history["accuracy"].append(val_accuracy / (index + 1))


           return conf_matrix


    def evaluate(self,dataloader):


        self.eval()

        conf_matrix = np.zeros((self.n_classes, self.n_classes))

        hist = np.zeros((self.n_classes, self.n_classes))

        with torch.no_grad():



            for index, data in enumerate(dataloader):



                img_meta = data[0][0] # img_meta identical for all samples
                gt_mask = data[1]
                img = data[2]
                label = data[3]
                x, _ = self(img)

                preds = torch.argmax(x, 1).data.cpu().numpy()


                ## all input image have identical width and height
                X_min = img_meta["window"][0][0].data.cpu().numpy()
                X_max =  img_meta["window"][1][0].data.cpu().numpy()
                Y_min = img_meta["window"][2][0].data.cpu().numpy()
                Y_max =  img_meta["window"][3][0].data.cpu().numpy()

                X = img_meta['shape'][0][0].data.cpu().numpy()
                Y = img_meta['shape'][1][0].data.cpu().numpy()

                ## extracting CAM explainability cues
                img_cam = img[:,:,X_min:X_max,Y_min:Y_max]
                x = self.cam_output(img_cam)
                x = F.upsample(x, [X, Y], mode='bilinear', align_corners=False)

                # resizing label to (batch , 1, 240, 320)
                label_idx = label.view(-1, 1, 1, 1).repeat((1, 1) + x.size()[-2:])

                print("")
                cams = torch.gather(x, 1, label_idx).squeeze(dim=1)


                normalizer = torch.amax(cams, (1, 2))
                normalizer += (normalizer == 0)*1 ## avoid dividing by zero
                cams = cams / normalizer.view(-1,1,1)

                img_orig = plt.imread(img_meta["path"][0])[95:]


                ## plotting cam
                # display_cam = cams[0].data.cpu().numpy()
                # plt.imshow(np.uint8(img_orig * np.tile(display_cam[..., None], (1, 1, 3))))
                # plt.axis("off")
                # plt.savefig(str(index) + ".png", bbox_inches='tight')


                # plt.imshow(display_cam)
                # plt.axis("off")
                # plt.savefig(str(index) + ".png", bbox_inches='tight')
                ## thresholding cams
                cams = (cams>0.4)*label.view(-1, 1, 1)


                ## plotting cam on input
                # display_cam = (cams>0.4)[0].data.cpu().numpy()
                # plt.imshow(np.uint8(img_orig * np.tile(display_cam[..., None], (1, 1, 3))))
                # plt.axis("off")
                # plt.savefig(str(index) + ".png", bbox_inches='tight')

                ####
                # plt.imshow(plt.imread(img_meta["path"][0]))
                _, ax = plt.subplots()
                ax.imshow(plt.imread(img_meta["path"][0])[95:])
#               ###

                ## insterting
                for current_index in range(cams.shape[0]):
                    current_label = label[current_index].data.cpu().numpy()
                    if current_label>0:
                        current_cam = cams[current_index].data.cpu().numpy()
                        current_gt_mask = gt_mask[current_index].data.cpu().numpy()
                        current_pred_mask = np.zeros_like(current_gt_mask)

                        label_im, nb_labels = ndimage.label(current_cam)
                        if nb_labels > 0:
                            ## catching the case of full zero cams
                            ## extracting bboxes
                            masks = [label_im==(i+1) for i in range(nb_labels)]
                            masks = np.asarray(masks).transpose(1,2,0)
                            bboxes = extract_bboxes(masks)
                            for i in range(nb_labels):
                                current_bbox = bboxes[i,:]
                                current_pred_mask[current_bbox[0]:current_bbox[2],
                                current_bbox[1]:current_bbox[3]] = current_label

                                # ###
                                # rect = patches.Rectangle((current_bbox[1], current_bbox[0]), current_bbox[2]-current_bbox[1], current_bbox[3]-current_bbox[0], linewidth=1, edgecolor='r',
                                #                          facecolor='none')
                                # ax.add_patch(rect)
                                # ###


                            hist = score_hist(hist, current_gt_mask, current_pred_mask, self.n_classes)

                labels = label.data.cpu().numpy()
                for p_index in range(preds.shape[0]):
                    conf_matrix[labels[p_index], preds[p_index]] += 1

                print('Step: {}/{}\n'.format(index + 1, len(dataloader)))
            mIoU = compute_mIOU(hist,self.n_classes)

        return conf_matrix, mIoU
