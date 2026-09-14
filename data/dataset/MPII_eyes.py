# Notes (2021):
# based on "MPIIGAZE_eyes_tf.py"
# supports gazemap output

# Notes (2022):
# PyTorch version:
# 1. all dataset classes inherit torch.utils.data.Dataset
# 2. images in C*H*W channel order
# 3. gt.shape=(2). Not (1,2).
# 4. if self.preprocess:
#           img = utils.image_preprocess(img, self.mean, self.std, jitter=False)
# 5. __getitem__ returns (data tensors,), (ground truth tensors,)
# self.filter = filter.__dict__

import os
import pickle
import numpy as np
import csv
import random

import tools.utils as utils
import cv2

import torch
from torch.utils.data import Dataset as D

# note, since every image is a pair of eyes, depending on index of the image
# one can easily figure out if it is left or right eye
class Dataset_MPIIGAZE_eyes(D):
    def __init__(self, 
                 img_path = "/data/MPIIGaze/Data/resave_imgs/", 
                 npy_path = "/data/MPIIGaze/Data/resave_gt/", 
                 filter = None, 
                 init=True, 
                 mean = [0.5055, 0.5055, 0.5055],
                 std = [0.2895, 0.2895, 0.2895],
                 shuffle = False,
                 aug = None, # pass how many aug per image, if not then none
                 resize_320=False,
                 use_gazemap_filelist = False,
                 output_gazemap = False,
                 gazemap_path = "/data/MPIIGaze/Data/gazemap/",
                 preprocess = True,
                 **args
                 ):
        self.img_path = img_path
        self.npy_path = npy_path
        self.std = std
        self.mean = mean
        self.resize_image = resize_320
        self.filter = filter
        self.person_day_index = [] # index for each day, each person (2d array) 
        self.person_spam = [] # how many items each person has
        self.retrieval_item_paths = [] # img path and npy path
        self.total_person = 0
        self.aug = aug
        
        self.preprocess = preprocess

        # LZH: Oct 2021 update - gazemap
        self.use_gazemap_filelist = use_gazemap_filelist
        self.output_gazemap = output_gazemap
        if use_gazemap_filelist:
            self.gazemap_path = gazemap_path

        if init:
            # if a filter is defined, we generate the base set using filter rule 
            # note that some filter rule will be contradicting, documentation is left 
            # in the process_filter section to demonstrate which filter is a family
            if filter is not None:
                self._process_filter() # first pre-process the filter
            # if there is no filter, create a general dataset for all individuals
            else:
                self._fill_dict()
            # shuffle only if paramter is true
            if shuffle:
                self.shuffle()
    # get specific item by index only (from person = 0, days = 0)
    def __getitem__(self, index):
        (item_path, npy_path) = self.retrieval_item_paths[index]
        img = None  
        if self.aug is not None:
            rand_result =  "{}.jpg".format(random.randrange(self.aug))
            item_path = item_path.split(".jpg")[0]
            item_path = os.path.join(item_path, rand_result)
        img = cv2.imread(item_path)
        if self.resize_image:
            img = cv2.resize(img, (320, 192))
        npy_xyz = np.load(npy_path, allow_pickle=True)
        npy_pr = utils.xyz_to_pitchyaw(npy_xyz)
        #npy_pr  = (-npy_pr[0], npy_pr[1])
        gt = np.array([-npy_pr[0], npy_pr[1]])
        
        flip = True if item_path[-10] == "1" else False
        if flip:
            img = cv2.flip(img, 1)
            gt = np.array([-npy_pr[0], -npy_pr[1]])

        if self.preprocess:
            img = utils.image_preprocess(img, self.mean, self.std, jitter=False, transpose=True)
        
        if self.output_gazemap:
            if len(item_path.split('/')[-1]) < 10:
                gmap_output = cv2.imread((self.gazemap_path + item_path[-22:-6] + '.jpg'))
            else:
                gmap_output = cv2.imread((self.gazemap_path + item_path[-20:]))
            if self.resize_image:
                gmap_output = cv2.resize(gmap_output, (320, 192))
                gmap_output = cv2.cvtColor(gmap_output, cv2.COLOR_BGR2GRAY)
            # return ((img, gmap_output), npy_pr) (TF1 ver.)
            return (img, gmap_output), (gt,)

        # return ((img,), npy_pr) (TF1 ver.)
        return (img,), (gt,)
        
    # length of the dictionary
    def __len__(self):
        return len(self.retrieval_item_paths)
    
    def print_status(self):
        print("total person: ", self.total_person)
        print("person spam: ", self.person_spam)
        print("total size: ", len(self.retrieval_item_paths))
    
    def _process_filter(self):
        # if there are multiple filter instruction passed which contradict each other,
        # it throws error
        if self.filter is None:
            raise Exception("_process_filter should not be called when self.filter is None")
        filter_set = False 
        if "txt_person_id" in self.filter:
            if filter_set:
                raise ValueError("contradicting filter received")
            filter_set = True
            
            self.filter_txt_path = self.filter["txt_path"]
            self.filter_person_id = "p" + "{:02d}".format(self.filter["txt_person_id"]) if self.filter["txt_person_id"] is not None else None
            self._fill_dict_txt_filtered()
        if "set_person_id" in self.filter:
            if filter_set:
                raise ValueError("contradicting filter received")
            filter_set = True
            
            self.set_person_id = self.filter["set_person_id"]
            self.samples_per_person = self.filter["sample_per_person"] if "sample_per_person" in self.filter else None
            self.inclusive = self.filter["inclusive"] if "inclusive" in self.filter else True
            self._fill_dict_validation_train_filtered()
        
        if filter_set == False:
            raise Exception("incorrect filter passed")

    # method to fill the dictionary
    def _fill_dict_txt_filtered(self):
        self.total_person = 1
        item_counter = 0
        with open(self.filter_txt_path) as f:
            self.person_day_index.append([])
            current_day = None
            for line in sorted(f.readlines()):
                day = line [0:5]
                id = int(line [6: 10]) - 1
                left = "1_" if line [15: 19] == "left" else "0_"
                img_path = os.path.join(self.img_path, self.filter_person_id, day, left + "{:04d}".format(id) + ".jpg")
                npy_path = os.path.join(self.npy_path, self.filter_person_id, day, left + "{:04d}".format(id) + ".npy") 
                self.retrieval_item_paths.append((img_path, npy_path))
                if current_day != day:
                    current_day = day
                    self.person_day_index[self.total_person-1].append(item_counter)
                item_counter+=1
            self.person_spam.append(len(self.retrieval_item_paths))
                
    def _fill_dict_validation_train_filtered(self):
        if not self.inclusive:
            self.set_person_id = [i for i in list(range(15)) if i not in self.set_person_id]
        self.total_person = len(self.set_person_id)
        if self.samples_per_person is None:
            item_counter = 0
            self.total_person = 0
            for person in sorted(os.listdir(self.img_path)):    
                person_str = int(person[1:3]) 
                if person_str in self.set_person_id:
                    item_counter = self._fill_person(person, item_counter)            
        else:
            item_counter = 0
            for person in sorted(os.listdir(self.img_path)):   
                person_str = int(person[1:3]) 
                if person_str in self.set_person_id:
                    item_counter = self._fill_person_with_fix_size(person, item_counter, self.samples_per_person)
            return 
   
    def _fill_dict(self):
        item_counter = 0
        self.total_person = 0
        for person in sorted(os.listdir(self.img_path)):    
            item_counter = self._fill_person(person, item_counter)
    
    def _fill_person_with_fix_size (self, person, item_counter, samples_per_person):
        # note, since the set is completely randomized, there is no person_day_index
        self.person_day_index = []
        item_counter += samples_per_person
        root = self.img_path
        root_filelist = self.gazemap_path if self.use_gazemap_filelist else self.img_path
        person_path = os.path.join(root, person)
        person_path_filelist = os.path.join(root_filelist, person)
        temp_storage = []
        for day in sorted(os.listdir(person_path)):  
            day_path = os.path.join(person_path, day)
            day_path_filelist = os.path.join(person_path_filelist, day)
            for item in sorted(os.listdir(day_path_filelist)):
                if item.endswith(".jpg"):
                    item_path = os.path.join(day_path, item)
                    sub_path = item_path.split(self.img_path)[1]
                    npy_path = os.path.join(self.npy_path, sub_path)
                    npy_path = npy_path.replace(".jpg", ".npy")
                    # now for checking if file exists
                    if not os.path.isfile(item_path) or not os.path.isfile(npy_path):
                        print("error, npy file with subpath : {} does not exists".format(sub_path))
                        continue
                    temp_storage.append((item_path, npy_path))
        
        if len(temp_storage) == 0:
            self.person_spam.append(0)
            return item_counter
        self.person_spam.append(samples_per_person)
        if len(temp_storage) <= samples_per_person:
            temp_counter = samples_per_person
            while temp_counter > len(temp_storage):
                self.retrieval_item_paths.extend(temp_storage)  
                temp_counter -= len(temp_storage)
            random.shuffle(temp_storage)
            diff = samples_per_person % len(temp_storage)
            self.retrieval_item_paths.extend(temp_storage[:diff])
        else:
            random.shuffle(temp_storage)
            self.retrieval_item_paths.extend(temp_storage[:samples_per_person])
        return item_counter
            
        
    def _fill_person(self, person, item_counter):
        self.person_day_index.append([])
        root = self.img_path
        root_filelist = self.gazemap_path if self.use_gazemap_filelist else self.img_path
        person_path = os.path.join(root, person)
        person_path_filelist = os.path.join(root_filelist, person)
        start_counter = item_counter
        self.total_person +=1
        
        for day in sorted(os.listdir(person_path)):
            self.person_day_index[len(self.person_day_index) - 1].append(item_counter)
            day_path = os.path.join(person_path, day)
            day_path_filelist = os.path.join(person_path_filelist, day)
            for item in sorted(os.listdir(day_path_filelist)):
                if item.endswith(".jpg"):
                    item_path = os.path.join(day_path, item)
                    sub_path = item_path.split(self.img_path)[1]
                    npy_path = os.path.join(self.npy_path, sub_path)
                    npy_path = npy_path.replace(".jpg", ".npy")
                    # now for checking if file exists
                    if not os.path.isfile(item_path) or not os.path.isfile(npy_path):
                        print("error, npy file with subpath : {} does not exists".format(sub_path))
                        continue
                    self.retrieval_item_paths.append((item_path, npy_path))
                    item_counter +=1
        self.person_spam.append(item_counter - start_counter)
        return item_counter           
                
    # provided a path save the obj       
    def save(self, save_path):
        with open(save_path, 'wb') as output:
            pickle.dump(self.__dict__, output, pickle.HIGHEST_PROTOCOL)
    # proviaded a path load the old obj
    def load(self, save_path):
        with open(save_path, "rb") as input:
            self.__dict__.update(pickle.load(input)) 
    # save a csv of the current state
    def save_csv(self, save_path):
        with open(save_path, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            counter = 0
            for img_path, npy_path in self.retrieval_item_paths:
                writer.writerow([counter, img_path, npy_path])
                counter += 1
    # save a csv of the current person seperation
    def save_seperation_person(self, save_path):
        if self.person_day_index is not None:
            with open(save_path, "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                counter = 0
                for person in self.person_day_index:
                    writer.writerow([counter, person[0]])
                    counter += 1
    # save a csv of the current person - day seperation
    def save_seperation_person_days(self, save_path):
        if self.person_day_index is not None:
            with open(save_path, "w") as csv_file:
                writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                counter = 0
                for person in self.person_day_index:
                    day_counter = 0
                    for day in person:
                        writer.writerow([counter, day_counter, day])
                        day_counter +=1
                    counter +=1

    def shuffle(self):
        # note that after shuffling person_day_index
        self.person_day_index = None
        random.shuffle(self.retrieval_item_paths)
    
if __name__ == '__main__':
    # ROOT_PATH = "."
    FILTER_PATH = "./p10.txt"
    ROOT_PATH = "./3d_to_2d"
    dataset = Dataset_MPIIGAZE_eyes(ROOT_PATH + "/resave_imgs/", 
                                    ROOT_PATH + "/resave_gt/",
                                    filter = {
                                        "txt_person_id": 10,
                                        "txt_path": "./p10.txt"
                                    },
                                    dont_exists = True,
                                    should_not_give_error = True,
                                )
    dataset.save("./saved_dataset_MPIIGAZE.pkl")
    dataset.load("./saved_dataset_MPIIGAZE.pkl")
    print(len(dataset))
    dataset.save_csv("./saved_dataset_MPIIGAZE_eye.csv")
    dataset.save_seperation_person("./saved_dataset_MPIIGAZE_eye_person.csv")
    dataset.save_seperation_person_days("./saved_dataset_MPIIGAZE_eye_day.csv")
    dataset.print_status()
    dataset.shuffle()
    dataset.save_csv("./saved_dataset_MPIIGAZE_eye.csv")
    dataset.print_status()
    dataset[100]
    