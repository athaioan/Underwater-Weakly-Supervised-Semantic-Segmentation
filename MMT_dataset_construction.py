import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pickle
from types import SimpleNamespace
from datetime import datetime, timedelta
import cv2
import shutil



args = SimpleNamespace(L1_folder="D:/103536/Annotations_new_29032021/2019/L1/",
                       L2_folder="D:/103536/Annotations/2019/L2/",
                       annotation_folder="Annotation_txt_files",
                       video_folder="D:/103536/Video_2019/",
                       save_folder_L1="D:/103536/Annotations_new_29032021/2019/L1/Clean/",
                       dataset_source="D:/103536/MMT-Dataset/"
                       )

class Line():

    def __init__(self, source_folder, annotation_folder):
        self.source_folder = source_folder
        self.folders = os.listdir(self.source_folder)
        self.annotations_folder = annotation_folder
        self.views = ["left-side", "centric", "right-side"]
        self.category_to_label = {"CWC Abrassion": 0,
                                 "CWC Cracked": 1,
                                 "CWC Gouging": 2,
                                 "CWC Missing": 3,
                                 "Anodes": 4,
                                 "Field Joint": 6,
                                 "Anchor": 7,
                                 "Cable": 8,
                                 "Chain": 9,
                                 "Concrete": 10,
                                 "Drum": 11,
                                 "Fishing Net": 12,
                                 "Metal": 13,
                                 "Other Hard": 14,
                                 "Other Soft": 15,
                                 "Soft Rope": 16,
                                 "Tarpaulin": 17,
                                 "Unidentified": 18,
                                 "Wire": 19,
                                 "SBF Boulder": 20}


        self.missing_datetime_counter = 0


        ### Removing the Template folder (occures only in L1 line)
        if "Template" in self.folders:
            self.folders = self.folders[:-1]

        self.data = {}
        self.blocks = {}

    def print_progress(self, class_folder_idx, num_class_folders, instance_idx, num_instances):
        print("Class Folder: {} out of {} || Instances {} out of {}\n".format(class_folder_idx + 1, num_class_folders,
                                                     instance_idx + 1, num_instances))

    def read_annotation_file(self, folder, file):

        ## from .jpg to .txt
        file = file[:-3] + "txt"

        try:
            with open(folder + file) as f:
                bboxes = [bbox for bbox in f]
        except:
            bboxes = []

        ## Some times a txt file is created but does not contain any annotation
        annotated = len(bboxes) > 0

        return annotated, bboxes

    def investigate_date_format(self,date):
        try:
            date = datetime.strptime(date, "%Y_%m_%d %H_%M_%S_%f")
            return date
        except:
            pass
        try:
            ## Sometimes date is writter in a different format
            date = datetime.strptime(date, "%Y-%m-%d %H_%M_%S_%f")
            return date
        except:
            pass
        try:
            ## Sometimes date is writter in a different format (one more space)
            date = datetime.strptime(date, " %Y-%m-%d %H_%M_%S_%f")
            return date
        except:
            pass
        try:
            date = datetime.strptime(date, " %Y_%m_%d %H_%M_%S_%f")
            return date
        except:
            return

    def sort_blocks(self):
        ### sorting blocks
        for block in self.blocks:
            for view in self.blocks[block]:
                self.blocks[block][view] = sorted(self.blocks[block][view])
        return

    def save_as_dict(self, save_file):
        with open(save_file, 'wb') as fp:
            pickle.dump(self.blocks, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def add_instance(self, meta_data):

        ## adding block if does not already exist
        if meta_data["block"] not in self.data:
            self.data[meta_data["block"]] = {}

        ## adding triplet if does not already exist
        if meta_data["triplet_name"] not in self.data[meta_data["block"]]:
            self.data[meta_data["block"]][meta_data["triplet_name"]] = {}

        ## adding instances in the triplet
        self.data[meta_data["block"]][meta_data["triplet_name"]][meta_data["view_idx"]] = meta_data

        ## dates and times an instance appeared for every block
        if meta_data["block"] not in self.blocks:
            self.blocks[meta_data["block"]] = {}

        ## adding triplet if does not already exist
        if meta_data["view_idx"] not in self.blocks[meta_data["block"]]:
            self.blocks[meta_data["block"]][meta_data["view_idx"]] = []

        self.blocks[meta_data["block"]][meta_data["view_idx"]].append(meta_data["date"])

        return


    def extract_meta_data(self, instance, class_folder):

        meta_data = {}

        ## Block
        meta_data["block"] = instance.split("Block")[-1].split("_")[0]


        ## Category
        meta_data["category"] = class_folder

        ## KP
        KP = instance.split("KP")[-1][1:9].partition(".")

        meta_data["KP"] = int(KP[0] + KP[2])

        ## Offset
        try:
            meta_data["offset"] = int(instance.split("_")[-1].split(".jpg")[0][:-1])
        except:
            ## Not previous and later offsets thus the only offset is 0s
            meta_data["offset"] = 0

        ## Date
        fraction = instance.split(".")[2].split(" ")[0]
        date = instance.split(".")[1][4:] + "_" + fraction

        meta_data["date"] = self.investigate_date_format(date)

        meta_data["date"] += timedelta(seconds=meta_data["offset"])


        ## View
        meta_data["view_idx"] = int(instance.split("HD")[-1][0])
        meta_data["view"] = self.views[meta_data["view_idx"]-1]


        ## Annotation_folder
        annotation_folder = self.source_folder + class_folder + "/" + self.annotations_folder + "/"
        meta_data["is_annotated"], meta_data["bbox"] = self.read_annotation_file(annotation_folder, instance)

        ## Triplet_view_name
        meta_data["triplet_name"] = "_".join(instance.split("_")[:-3]) + "_" + str(meta_data["offset"])

        ## Image path
        meta_data["path"] = instance

        return meta_data

    def extract_data(self):

        for class_folder_idx, class_folder in enumerate(self.folders):

            instances = os.listdir(self.source_folder+class_folder)

            for instance_idx, instance in enumerate(instances):


                if instance[-3:] != "jpg":
                    ## Annotation folder is not an instance
                    continue

                if "PsV" in instance:
                    ## At this moment PsV-Block will not be used
                    continue

                ## Missing date time
                if "datetime" in instance:
                    self.missing_datetime_counter += 1
                    continue

                ## Extracting the meta_data from the current instance
                meta_data = self.extract_meta_data(instance, class_folder)


                ## Adding the current instances
                self.add_instance(meta_data)

                self.print_progress(class_folder_idx, len(self.folders), instance_idx, len(instances))

    def extract_video_meta_data(self, video_instance):
        video_meta_data = {}

        video_meta_data["block"] = video_instance.split("Block")[1].split("_")[0]
        video_meta_data["view_idx"] = int(video_instance.split("HD")[-1][0])
        date = video_instance.split("@")[0]
        video_meta_data["date"] = datetime.strptime(date, "%Y%m%d%H%M%S%f")

        return video_meta_data

    def find_current_time_in_annotated_instances(self, time_stampes, current_time):

        for idx in range(len(time_stampes)):

            if time_stampes[idx] > current_time:
                date_comparison = time_stampes[idx]
                return date_comparison
        return None

    ### stackoverflow
    def seconds_interval(self, start, end):
        """start and end are datetime instances"""
        diff = end - start
        sec = diff.days * 24 * 60 * 60
        sec += diff.seconds * 1
        sec += diff.microseconds / 1000000
        return sec


    def extract_clean_samples(self, video_folder, data_set_save_folder):
        video_blocks = os.listdir(video_folder)

        ### comment out
        for video_block_idx, video_block in enumerate(video_blocks[8:]):

            if "S" in video_block:
                save_folder = data_set_save_folder + "train/"
            else:
                save_folder = data_set_save_folder + "val/"

            if video_block not in L1.blocks:
                ## if folder not an actual block
                continue
            data_videos = os.listdir(video_folder + "/" + video_block)
            data_videos = [_ for _ in data_videos if "DATA" in _]

            ### pick a subsample from data_videos
            np.random.seed(video_block_idx)
            data_videos_idxs = np.arange(len(data_videos))
            np.random.shuffle(data_videos_idxs)
            number_of_data_videos = min(5, len(data_videos))
            data_videos = np.asarray(data_videos)[data_videos_idxs[:number_of_data_videos]].tolist()

            for data_video_idx, data_video in enumerate(data_videos):

                # ## comment out
                # if data_video_idx > 1:
                #     print("")
                #     break

                video_instances = os.listdir(video_folder + "/" + video_block + "/" + data_video)

                identified_timestamps = {}

                for video_instance in video_instances:


                    if "Block" not in video_instance or "SIT" in video_instance:
                        continue

                    if len(identified_timestamps) == 0:
                        ## We do not have the timestamps of clean samples
                        video_meta_data = self.extract_video_meta_data(video_instance)

                        time_stamps_of_instance = self.blocks[video_meta_data["block"]][video_meta_data["view_idx"]]

                        video_path = video_folder + "/" + video_block + "/" + data_video + "/" + video_instance
                        cap = cv2.VideoCapture(video_path)
                        number_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        frame_rate = cap.get(cv2.CAP_PROP_FPS)

                        last_saved = datetime(1, 1, 1, 0, 0)
                        ## if video shorter than 5 minutes
                        if number_frames < frame_rate * 60 * 5:
                            continue

                        video_start_time = video_meta_data["date"]

                        date_comparison = self.find_current_time_in_annotated_instances(time_stamps_of_instance, video_start_time)

                        for frame_idx in range(int(number_frames)):
                            # ## comment out
                            # if frame_idx > 3000:
                            #     print("")
                            #     break

                            print(video_block_idx, " out of ", len(video_blocks))
                            print(data_video_idx, " out of ", len(data_videos))
                            print(frame_idx, " out of ", int(number_frames), "\n")


                            ret, frame = cap.read()


                            ## current_frame close to instance
                            close_to_instance = False
                            close_to_limits = False

                            if date_comparison is not None:
                                close_to_instance = True

                                if video_start_time > date_comparison and self.seconds_interval(date_comparison, video_start_time) > 5:
                                    close_to_instance = False
                                if video_start_time <= date_comparison and self.seconds_interval(video_start_time, date_comparison) > 5:
                                    close_to_instance = False
                                    date_comparison = self.find_current_time_in_annotated_instances(time_stamps_of_instance, video_start_time)

                            ## close to the start/end of the video
                            if frame_idx < frame_rate * 60 * 2 or (number_frames-frame_idx) < frame_rate * 60 * 2:
                                close_to_limits = True

                            if not close_to_instance and not close_to_limits:
                                ### save frame but not 2 cleans in less than 20 seconds
                                if self.seconds_interval(last_saved, video_start_time) > 25:
                                    ## save frame
                                    triplet_name = "Clean_Block" + video_meta_data["block"] + "_" + "_".join(str(str(video_start_time).split(".")[0] + ":" + str(video_start_time).split(".")[1]).split(":"))
                                    if not os.path.exists(save_folder + triplet_name):
                                        os.makedirs(save_folder + triplet_name)

                                    frame_name = video_meta_data["block"] + "_" + "_".join((str(video_meta_data["view_idx"]) + "_" + str(str(video_start_time).split(".")[0] + ":" + str(video_start_time).split(".")[1])).split(":")) + ".jpg"
                                    frame_name = save_folder + triplet_name + "/" + frame_name
                                    plt.imsave(frame_name, frame)
                                    last_saved = video_start_time
                                    identified_timestamps[frame_idx] = "Clean_Block" + video_meta_data["block"] + "_" + "_".join(str(str(video_start_time).split(".")[0] + ":" + str(video_start_time).split(".")[1]).split(":"))

                            ## increasing time
                            video_start_time += timedelta(seconds=1/frame_rate)

                    else:

                        frame_idx_found = [i for i in identified_timestamps]
                        ## We already got the timestamps of clean samples
                        video_meta_data = self.extract_video_meta_data(video_instance)

                        video_path = video_folder + "/" + video_block + "/" + data_video + "/" + video_instance
                        cap = cv2.VideoCapture(video_path)
                        number_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        frame_rate = cap.get(cv2.CAP_PROP_FPS)

                        for frame_idx in range(int(number_frames)):
                            print(video_block_idx, " out of ", len(video_blocks))
                            print(data_video_idx, " out of ", len(data_videos))
                            print(frame_idx, " out of ", int(number_frames), "\n")

                            ret, frame = cap.read()

                            if frame_idx in frame_idx_found:
                            ## save frame
                                triplet_name = identified_timestamps[frame_idx]

                                frame_name = video_meta_data["block"] + "_" + "_".join((str(video_meta_data["view_idx"]) + "_" + str(str(video_start_time).split(".")[0] + ":" + str(video_start_time).split(".")[1])).split(":")) + ".jpg"
                                frame_name = save_folder + triplet_name + "/" + frame_name
                                plt.imsave(frame_name, frame)

                            ## increasing time
                            video_start_time += timedelta(seconds=1/frame_rate)




    def split_dataset(self, dataset_source):

        if not os.path.exists(dataset_source+"train"):
            os.makedirs(dataset_source+"train")

        if not os.path.exists(dataset_source + "val"):
            os.makedirs(dataset_source + "val")

        for block_idx, block in enumerate(self.data):

            # ### comment out
            # if block_idx > 1:
            #     break
            # ### comment out

            print(block_idx, "out of ", len(self.data))

            if "S" in block:
                ## If swedish block train
                subset_path = dataset_source + "train/"
            else:
                ## otherwise val
                subset_path = dataset_source + "val/"

            for triplet in self.data[block]:

                if np.sum([self.data[block][triplet][i]["is_annotated"] for i in self.data[block][triplet]]) == 0:
                    ## not annotated instances in this triplet
                    continue
                if len(self.data[block][triplet]) != 3:
                    ## not complete triplet
                    continue

                triplet_path = subset_path + triplet + "/"

                if not os.path.exists(triplet_path):
                    os.makedirs(triplet_path)

                for instance in self.data[block][triplet]:
                    meta_data = self.data[block][triplet][instance]

                    if meta_data["is_annotated"]:
                        target_path = triplet_path + "annotated_" + str(self.category_to_label[meta_data["category"]]) + "_" + meta_data["path"]

                    else:
                        target_path = triplet_path + str(self.category_to_label[meta_data["category"]]) + "_" + meta_data["path"]

                    source_path = args.L1_folder + meta_data["category"] + "/" + meta_data["path"]
                    shutil.copyfile(source_path, target_path)


# L2 = Line(args.L2_folder, args.annotation_folder)
#
# L2.extract_data()
#
# ## sorting blocks
# L2.sort_blocks()
#
# ## saving dict
# L2.save_as_dict("L2_instances.p")
#
# ## storing L2
#
L1 = Line(args.L1_folder, args.annotation_folder)
L1.extract_data()

## sorting blocks
L1.sort_blocks()
#
L1 = Line(args.L1_folder, args.annotation_folder)
with open('L1_instances.p', 'rb') as fp:
    L1.blocks = pickle.load(fp)


L1.split_dataset(args.dataset_source)

L1.extract_clean_samples(args.video_folder, args.dataset_source)

