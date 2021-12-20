import numpy as np
import os

import shutil, errno

def copyanything(src, dst):
    try:
        shutil.copytree(src, dst)
    except OSError as exc: # python >2.5
        if exc.errno in (errno.ENOTDIR, errno.EINVAL):
            shutil.copy(src, dst)
        else: raise

train_folder = 'C:/Users/johny/Desktop/RPL v2/MMT-Datasetv3/train/'
val_folder = 'C:/Users/johny/Desktop/RPL v2/MMT-Datasetv3/val/'

train_f = os.listdir(train_folder)
val_f = os.listdir(val_folder)

labels_dict = {}

imgsTrain = []
imgsVal = [];
imgsTest = [];


class_to_label =   {'Clean':0,
                    'Anchor':1,
                    'Anode':2,
                    'Boulder':3,
                    'Cable':4,
                    'CWC Abrassion':5,
                    'CWC Abrasion':5,
                    'CWC Cracked':6,
                    'CWC Gouging':7,
                    'CWC Missing':8,
                    'Drum':9,
                    'Field Joint':10,
                    'Metal':11,
                    'Other Hard':12,
                    'Other Soft':13,
                    'Soft Rope':14,
                    'Tarpaulin' : 15,
                    'Unidentified': 16,
                    'Wire': 17,
                    }

label_txt = {}
np.random.seed(0)

count_class = np.zeros((18,1))

for current_triplet in train_f:
    current_triplet = train_folder + current_triplet
    current_t = os.listdir(current_triplet)

    asign_to_val = False
    asign_to_test = False

    current_rand = np.random.random()
    if current_rand < 0.1:
        asign_to_val = True
    elif current_rand < 0.2:
        asign_to_test = True


    for img in current_t:
        if "jpg" in img:
            if not "Clean" in current_triplet:
                line = int(img.split("Block")[1][0])
                country = img.split("Block")[1][3:].split("_")[0]
                if "annotated" not in img:
                    label = "Clean"
                else:
                    label = img.split("_")[2]

            else:
                line = img[0]
                country = img.split("_")[0][3:]
                label = "Clean"

            img_label = class_to_label[label]
            img_path = current_triplet.split("/")[-1]+"/" + img
            if "D" in country or (img_label==6 and asign_to_test):
                ## testset
                imgsTest.append(img_path)
            elif (line==1 and "F" in country) or (img_label==6 and asign_to_val):
                ## valset
                imgsVal.append(img_path)

            else:
                ## trainset
                imgsTrain.append(img_path)
                count_class[img_label] += 1

            label_txt[img_path]=img_label

# for current_triplet in val_f:
#     current_triplet = val_folder + current_triplet
#     current_t = os.listdir(current_triplet)
#
#     asign_to_val = False
#     asign_to_test = False
#
#     current_rand = np.random.random()
#     if current_rand < 0.1:
#         asign_to_val = True
#     elif current_rand < 0.2:
#         asign_to_test = True
#
#
#
#     for img in current_t:
#         if "jpg" in img:
#             if not "Clean" in current_triplet:
#                 line = int(img.split("Block")[1][0])
#                 country = img.split("Block")[1][3:].split("_")[0]
#                 if "annotated" not in img:
#                     label = "Clean"
#                 else:
#                     label = img.split("_")[2]
#
#             else:
#                 line = img[0]
#                 country = img.split("_")[0][3:]
#                 label = "Clean"
#
#             img_label = class_to_label[label]
#             img_path = current_triplet.split("/")[-1]+"/" + img
#
#             if img_label == 6:
#                 print("")
#
#             if "D" in country or (img_label==6 and asign_to_test):
#                 ## testset
#                 imgsTest.append(img_path)
#
#             elif (line == 1 and "F" in country) or (img_label == 6 and asign_to_val):
#                 ## valset
#                 imgsVal.append(img_path)
#
#             else:
#                 ## trainset
#                 imgsTrain.append(img_path)
#                 count_class[img_label] += 1
#
#             label_txt[img_path]=img_label


print("")

# np.save('class_dict.npy',  label_txt)

# import pandas as pd
# df = pd.DataFrame (count_class)
# filepath = 'my_excel_file.xlsx'
# df.to_excel(filepath, index=False)
#
# with open('train.txt', 'w') as file:
#     for item in imgsTrain:
#         file.write("%s\n" % item)
#
# with open('val.txt', 'w') as file:
#     for item in imgsVal:
#         file.write("%s\n" % item)
#
# with open('test.txt', 'w') as file:
#     for item in imgsTest:
#         file.write("%s\n" % item)