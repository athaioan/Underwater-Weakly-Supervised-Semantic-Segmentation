import numpy as np
import os

import shutil, errno

#
train_txt = 'C:/Users/johny/Desktop/RPL v2/train.txt'
val_txt = 'C:/Users/johny/Desktop/RPL v2/val.txt'
test_text = 'C:/Users/johny/Desktop/RPL v2/test.txt'

#
# with open(test_text) as file:
#     test_triplets_path = [l.rstrip("\n").split("/")[0] for l in file]
# with open(val_txt) as file:
#     val_triplets_path = [l.rstrip("\n").split("/")[0] for l in file]
# with open(train_txt) as file:
#     train_triplets_path = [l.rstrip("\n").split("/")[0] for l in file]
#
#
# test_triplets = list(np.unique(np.asarray(test_triplets_path)))
# val_triplets = list(np.unique(np.asarray(val_triplets_path)))
#
# train_triplets=[]
# for current_triplet in train_triplets_path:
#     if current_triplet not in val_triplets_path and current_triplet not in test_triplets_path:
#         train_triplets.append(current_triplet)
#
# train_triplets = list(np.unique(np.asarray(train_triplets)))
#
# print("")
#
#
#
# with open('train_multiview.txt', 'w') as file:
#     for item in train_triplets:
#         file.write("%s\n" % item)
# #
# with open('val_multiview.txt', 'w') as file:
#     for item in val_triplets:
#         file.write("%s\n" % item)
#
# with open('test_multiview.txt', 'w') as file:
#     for item in test_triplets:
#         file.write("%s\n" % item)


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


train_txt = 'C:/Users/johny/Desktop/RPL v2/train_multiview.txt'
val_txt = 'C:/Users/johny/Desktop/RPL v2/val_multiview.txt'
test_text = 'C:/Users/johny/Desktop/RPL v2/test_multiview.txt'


with open(test_text) as file:
    test_triplets_path = [l.rstrip("\n") for l in file]
with open(val_txt) as file:
    val_triplets_path = [l.rstrip("\n") for l in file]
with open(train_txt) as file:
    train_triplets_path = [l.rstrip("\n") for l in file]



imgs_dist = np.zeros((1,18))

train_imgs = []
for current_triplet in test_triplets_path:
    current_triplet_fold = os.listdir("C:/Users/johny/Desktop/RPL v2/MMT-Datasetv3/triplets/" + current_triplet)
    for i in current_triplet_fold:
        if i.split(".")[-1]=="jpg":
            train_imgs.append(current_triplet+"/"+i)
            cls = current_triplet.split("_")[0]
            if "annotated" in i:
                label = class_to_label[cls]
            else:
                label = 0

            imgs_dist[0,label] += 1


import pandas as pd
df = pd.DataFrame (imgs_dist)
filepath = 'images.xlsx'
df.to_excel(filepath, index=False)


with open('test_new.txt', 'w') as file:
    for item in train_imgs:
        file.write("%s\n" % item)



print("")



