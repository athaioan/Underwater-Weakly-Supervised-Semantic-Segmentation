import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt

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


            img = data[1]
            label = data[2]

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

                   img = data[1]
                   label = data[2]

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


    def extract_cams(self, dataloader, low_a=4, high_a=16):

        if not os.path.exists(cam_folder):
            os.makedirs(cam_folder)


        with torch.no_grad():
            for index, data in enumerate(dataloader):
                print(str(index)+" / " + str(len(dataloader)))
                #current_path, imgs, label, windows, orginal_shape, original_img
                original_shape = data[4]
                label = data[2]

                imgs = data[1]
                windows = data[3]
                img_original = data[5][0].data.cpu().numpy()

                final_cam = np.zeros([self.n_classes, original_shape[0], original_shape[1]])
                final_cam_unlabeled = np.zeros([self.n_classes, original_shape[0], original_shape[1]])

                for index, img in enumerate(imgs):

                    window = windows[index]


                    x = self.cam_output(img)


                    x = F.upsample(x, [img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)[0]

                    ## removing the crop window
                    x = x[:, window[0]:window[2], window[1]:window[3]]

                    x = F.upsample(x.unsqueeze(0), [original_shape[0].data.cpu().numpy()[0], original_shape[1].data.cpu().numpy()[0]], mode='bilinear', align_corners=False)[0]

                    ## filter out non-existing classes
                    cam = x.cpu().numpy() * label.clone().view(self.n_classes, 1, 1).data.cpu().numpy()
                    cam_unlabeled = x.cpu().numpy()

                    if index % 2 == 1:
                        cam = np.flip(cam, axis=2)
                        cam_unlabeled = np.flip(cam_unlabeled, axis=2)

                    final_cam += cam
                    final_cam_unlabeled += cam_unlabeled

                ## normalizing final_cam
                denom = np.max(final_cam, (1, 2))
                denom_unlabeled = np.max(final_cam_unlabeled, (1, 2))

                ## when class does not exist then divide by one
                denom += 1 - (denom > 0)
                denom_unlabeled += 1 - (denom_unlabeled > 0)

                final_cam /= denom.reshape(self.n_classes, 1, 1)

                ########## savings cams as dict
                final_cam_dict = {}
                final_cam_dict_unlabeled = {}

                for i in range(self.n_classes):
                    final_cam_dict_unlabeled[i] = final_cam_unlabeled[i]
                    if label[0][i] == 1:
                        final_cam_dict[i] = final_cam[i]



                existing_cams = np.asarray(list(final_cam_dict.values()))
                existing_cams_unlabeled = np.asarray(list(final_cam_dict_unlabeled.values()))

                bg_score_low = np.power(1 - np.max(existing_cams, axis=0), low_a)
                bg_score_high = np.power(1 - np.max(existing_cams, axis=0), high_a)

                bg_score_low_cams = np.concatenate((np.expand_dims(bg_score_low, 0), existing_cams), axis=0)
                bg_score_high_cams = np.concatenate((np.expand_dims(bg_score_high, 0), existing_cams), axis=0)

                crf_score_low = crf_inference(img_original, bg_score_low_cams, labels=bg_score_low_cams.shape[0])
                crf_score_high = crf_inference(img_original, bg_score_high_cams, labels=bg_score_high_cams.shape[0])

                crf_low_dict = {}
                crf_high_dict = {}

                crf_low_dict[0] = crf_score_low[0]
                crf_high_dict[0] = crf_score_high[0]

                for i, key in enumerate(final_cam_dict.keys()):
                    # plus one to account for the added BG class
                    crf_low_dict[key+1] = crf_score_low[i+1]
                    crf_high_dict[key+1] = crf_score_high[i+1]

                if not os.path.exists(cam_folder+"low"):
                    os.makedirs(cam_folder+"low")

                if not os.path.exists(cam_folder+"high"):
                    os.makedirs(cam_folder+"high")

                if not os.path.exists(cam_folder + "default"):
                    os.makedirs(cam_folder + "default")


                np.save(cam_folder+"low/"+data[0][0].split("/")[-1][:-4]+".npy", crf_low_dict)
                np.save(cam_folder+"high/"+data[0][0].split("/")[-1][:-4]+".npy", crf_high_dict)
                np.save(cam_folder+"default/"+data[0][0].split("/")[-1][:-4]+".npy", existing_cams_unlabeled)

    def evaluate_cams(self,dataloader, cam_folder, gt_mask_folder):

        t_hold = 0.2
        c_num = np.zeros(21)
        c_denom = np.zeros(21)
        gt_masks = []
        preds = []


        for index, data in enumerate(dataloader):

            img = data[1]
            label = data[2]

            img_key = data[0][0].split("/")[-1].split(".")[0]
            # I = plt.imread("C:/Users/johny/Desktop/ProjectV2/VOCdevkit/VOC2012/JPEGImages/"+img_key+".jpg")
            ## loading generated cam
            # cam_low = np.load(cam_folder +"low/" + img_key + '.npy').item()
            cam_default = np.load(cam_folder +"default/" + img_key + '.npy')
            # cam_high = np.load(cam_folder +"high/" + img_key + '.npy').item()

            ## considering only the labeled cams
            cam_default = cam_default * label.clone().view(self.n_classes, 1, 1).data.cpu().numpy()

            ## normalizing final_cam
            denom = np.max(cam_default, (1, 2))

            ## when class does not exist then divide by one
            denom += 1 - (denom > 0)

            cam_default /= denom.reshape(self.n_classes, 1, 1)


            bg_score = np.expand_dims(np.ones_like(cam_default[0])*t_hold,0)
            pred = np.argmax(np.concatenate((bg_score, cam_default)), 0)



            ## loading ground truth annotated mask
            gt_mask = Image.open(gt_mask_folder + img_key + '.png')
            gt_mask = np.array(gt_mask)

            preds.append(pred)
            gt_masks.append(gt_mask)


        sc = scores(gt_masks, preds, self.n_classes+1)

        return sc

    def extract_sub_category(self,dataloader,sub_folder):

        features={}
        ids = {}

        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)


        for i in range(20):
            features[i] = []
            ids[i] =[]

        with torch.no_grad():
            for index, data in enumerate(dataloader):

                img = data[1]
                key = data[0][0].split("/")[-1].split(".")[0]
                label = data[2].data.cpu().numpy()
                x = self.feature_extractor(img)
                feat = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0).squeeze().data.cpu().numpy()
                feat /= np.linalg.norm(feat)

                id = np.where(label[0])[0]

                for i in id:
                    ids[i].append(key)
                    features[i].append(feat)

            np.save(sub_folder+"features.npy", features)
            np.save(sub_folder+"ids.npy", ids)

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

        return x1,x2



    def train_epoch(self, dataloader, optimizer, criterion, verbose=True):

        train_loss = 0
        train_accuracy = 0
        self.train()

        for index,data in enumerate(dataloader):

            imgs = data[0]
            labels = data[1]
            triplet_label = data[2]

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

                   imgs = data[0]
                   labels = data[1]

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


    def extract_cams(self, dataloader, low_a=4, high_a=16):

        if not os.path.exists(cam_folder):
            os.makedirs(cam_folder)


        with torch.no_grad():
            for index, data in enumerate(dataloader):
                print(str(index)+" / " + str(len(dataloader)))
                #current_path, imgs, label, windows, orginal_shape, original_img
                original_shape = data[4]
                label = data[2]

                imgs = data[1]
                windows = data[3]
                img_original = data[5][0].data.cpu().numpy()

                final_cam = np.zeros([self.n_classes, original_shape[0], original_shape[1]])
                final_cam_unlabeled = np.zeros([self.n_classes, original_shape[0], original_shape[1]])

                for index, img in enumerate(imgs):

                    window = windows[index]


                    x = self.cam_output(img)


                    x = F.upsample(x, [img.shape[2], img.shape[3]], mode='bilinear', align_corners=False)[0]

                    ## removing the crop window
                    x = x[:, window[0]:window[2], window[1]:window[3]]

                    x = F.upsample(x.unsqueeze(0), [original_shape[0].data.cpu().numpy()[0], original_shape[1].data.cpu().numpy()[0]], mode='bilinear', align_corners=False)[0]

                    ## filter out non-existing classes
                    cam = x.cpu().numpy() * label.clone().view(self.n_classes, 1, 1).data.cpu().numpy()
                    cam_unlabeled = x.cpu().numpy()

                    if index % 2 == 1:
                        cam = np.flip(cam, axis=2)
                        cam_unlabeled = np.flip(cam_unlabeled, axis=2)

                    final_cam += cam
                    final_cam_unlabeled += cam_unlabeled

                ## normalizing final_cam
                denom = np.max(final_cam, (1, 2))
                denom_unlabeled = np.max(final_cam_unlabeled, (1, 2))

                ## when class does not exist then divide by one
                denom += 1 - (denom > 0)
                denom_unlabeled += 1 - (denom_unlabeled > 0)

                final_cam /= denom.reshape(self.n_classes, 1, 1)

                ########## savings cams as dict
                final_cam_dict = {}
                final_cam_dict_unlabeled = {}

                for i in range(self.n_classes):
                    final_cam_dict_unlabeled[i] = final_cam_unlabeled[i]
                    if label[0][i] == 1:
                        final_cam_dict[i] = final_cam[i]



                existing_cams = np.asarray(list(final_cam_dict.values()))
                existing_cams_unlabeled = np.asarray(list(final_cam_dict_unlabeled.values()))

                bg_score_low = np.power(1 - np.max(existing_cams, axis=0), low_a)
                bg_score_high = np.power(1 - np.max(existing_cams, axis=0), high_a)

                bg_score_low_cams = np.concatenate((np.expand_dims(bg_score_low, 0), existing_cams), axis=0)
                bg_score_high_cams = np.concatenate((np.expand_dims(bg_score_high, 0), existing_cams), axis=0)

                crf_score_low = crf_inference(img_original, bg_score_low_cams, labels=bg_score_low_cams.shape[0])
                crf_score_high = crf_inference(img_original, bg_score_high_cams, labels=bg_score_high_cams.shape[0])

                crf_low_dict = {}
                crf_high_dict = {}

                crf_low_dict[0] = crf_score_low[0]
                crf_high_dict[0] = crf_score_high[0]

                for i, key in enumerate(final_cam_dict.keys()):
                    # plus one to account for the added BG class
                    crf_low_dict[key+1] = crf_score_low[i+1]
                    crf_high_dict[key+1] = crf_score_high[i+1]

                if not os.path.exists(cam_folder+"low"):
                    os.makedirs(cam_folder+"low")

                if not os.path.exists(cam_folder+"high"):
                    os.makedirs(cam_folder+"high")

                if not os.path.exists(cam_folder + "default"):
                    os.makedirs(cam_folder + "default")


                np.save(cam_folder+"low/"+data[0][0].split("/")[-1][:-4]+".npy", crf_low_dict)
                np.save(cam_folder+"high/"+data[0][0].split("/")[-1][:-4]+".npy", crf_high_dict)
                np.save(cam_folder+"default/"+data[0][0].split("/")[-1][:-4]+".npy", existing_cams_unlabeled)



    def evaluate_cams(self,dataloader, cam_folder, gt_mask_folder):

        t_hold = 0.2
        c_num = np.zeros(21)
        c_denom = np.zeros(21)
        gt_masks = []
        preds = []


        for index, data in enumerate(dataloader):

            img = data[1]
            label = data[2]

            img_key = data[0][0].split("/")[-1].split(".")[0]
            # I = plt.imread("C:/Users/johny/Desktop/ProjectV2/VOCdevkit/VOC2012/JPEGImages/"+img_key+".jpg")
            ## loading generated cam
            # cam_low = np.load(cam_folder +"low/" + img_key + '.npy').item()
            cam_default = np.load(cam_folder +"default/" + img_key + '.npy')
            # cam_high = np.load(cam_folder +"high/" + img_key + '.npy').item()

            ## considering only the labeled cams
            cam_default = cam_default * label.clone().view(self.n_classes, 1, 1).data.cpu().numpy()

            ## normalizing final_cam
            denom = np.max(cam_default, (1, 2))

            ## when class does not exist then divide by one
            denom += 1 - (denom > 0)

            cam_default /= denom.reshape(self.n_classes, 1, 1)


            bg_score = np.expand_dims(np.ones_like(cam_default[0])*t_hold,0)
            pred = np.argmax(np.concatenate((bg_score, cam_default)), 0)



            ## loading ground truth annotated mask
            gt_mask = Image.open(gt_mask_folder + img_key + '.png')
            gt_mask = np.array(gt_mask)

            preds.append(pred)
            gt_masks.append(gt_mask)


        sc = scores(gt_masks, preds, self.n_classes+1)

        return sc

    def extract_sub_category(self,dataloader,sub_folder):

        features={}
        ids = {}

        if not os.path.exists(sub_folder):
            os.makedirs(sub_folder)


        for i in range(20):
            features[i] = []
            ids[i] =[]

        with torch.no_grad():
            for index, data in enumerate(dataloader):

                img = data[1]
                key = data[0][0].split("/")[-1].split(".")[0]
                label = data[2].data.cpu().numpy()
                x = self.feature_extractor(img)
                feat = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0).squeeze().data.cpu().numpy()
                feat /= np.linalg.norm(feat)

                id = np.where(label[0])[0]

                for i in id:
                    ids[i].append(key)
                    features[i].append(feat)

            np.save(sub_folder+"features.npy", features)
            np.save(sub_folder+"ids.npy", ids)


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
