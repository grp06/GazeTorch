import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import cv2
# from cv.segmap import Segmap
import os

FACE_ROOT = "D:/Gaze_Rand_Select_6N/data_collection/face_images"
SEG_ROOT = "D:/Gaze_Rand_Select_6N/data_collection/visualized_segmap"
FACE_SEG_ROOT = "D:/Gaze_Rand_Select_6N/data_collection/face_seg_demo"


class GUI_6N:

    def __init__(self, pair_list):

        self.root = tk.Tk()
        self.root.title("6N Rating")

        # create buttons
        self.button_good = tk.Button(self.root, text="Good", padx=40, pady=20, command=lambda: self.click(2))
        # self.button_good.pack(padx=40, pady=20)
        self.button_ok = tk.Button(self.root, text="OK", padx=40, pady=20, command=lambda: self.click(1))
        # self.button_ok.pack(padx=40, pady=20)
        self.button_poor = tk.Button(self.root, text="Poor", padx=40, pady=20, command=lambda: self.click(0))
        # self.button_poor.pack(padx=40, pady=20)
        self.button_exit = tk.Button(self.root, text="Save", padx=120, pady=20, command=self.save_label, state="disabled")
        # self.button_exit.pack()

        # button location
        self.button_good.grid(row=4, column=2)
        self.button_ok.grid(row=4, column=1)
        self.button_poor.grid(row=4, column=0)
        self.button_exit.grid(row=5, column=0, columnspan=3)

        # list for shuffled indices for images
        self.pair_list = list(pair_list)
        self.idx = 0

        self.progress_label = tk.Label(self.root, text="Image "+str(self.idx+1)+"of "+str(len(self.pair_list)))
        self.progress_label.grid(row=3, column=1)

        # self.display_img()
        #
        self.img_face = cv2.imread(os.path.join(FACE_ROOT, str(int(np.floor(self.pair_list[self.idx]/2)))+".png"), cv2.IMREAD_UNCHANGED)
        self.img_seg = cv2.imread(os.path.join(SEG_ROOT, str(int(np.floor(self.pair_list[self.idx]/2))) + ".png"), cv2.IMREAD_UNCHANGED)
        self.img_face_seg = cv2.imread(os.path.join(FACE_SEG_ROOT, str(int(np.floor(self.pair_list[self.idx]/2)))+".png"), cv2.IMREAD_UNCHANGED)

        # cv2.imshow("image", self.img_face)
        self.img_face_tk = ImageTk.PhotoImage(Image.fromarray(self.img_face[:, :, ::-1]))
        self.img_seg_tk = ImageTk.PhotoImage(Image.fromarray(self.img_seg[:, :, ::-1]))
        self.img_face_seg_tk = ImageTk.PhotoImage(Image.fromarray(self.img_face_seg[:, :, ::-1]))

        # display label
        self.face_img_label = tk.Label(self.root, image=self.img_face_tk)
        self.face_img_label.grid(row=1, column=0, rowspan=2)
        self.seg_img_label = tk.Label(self.root, image=self.img_seg_tk)
        self.seg_img_label.grid(row=1, column=1, rowspan=2)
        self.face_seg_img_label = tk.Label(self.root, image=self.img_face_seg_tk)
        self.face_seg_img_label.grid(row=1, column=2, rowspan=2)

        self.is_eye = True

        if int(np.floor(self.pair_list[self.idx])) % 2 == 0:
            if self.is_eye is True:
                self.lr_label = tk.Label(self.root, text="Please rate the LEFT EYE of the images below", font=("Arial", 20, 'bold'))
                self.lr_label.grid(row=0, column=0, columnspan=3)
            else:
                self.lr_label = tk.Label(self.root, text="Please rate the LEFT IRIS of the images below", font=("Arial", 20, 'bold'))
                self.lr_label.grid(row=0, column=0, columnspan=3)
        else:
            if self.is_eye is True:
                self.lr_label = tk.Label(self.root, text="Please rate the RIGHT EYE of the images below", font=("Arial", 20, 'bold'))
                self.lr_label.grid(row=0, column=0, columnspan=3)
            else:
                self.lr_label = tk.Label(self.root, text="Please rate the RIGHT IRIS of the images below", font=("Arial", 20, 'bold'))
                self.lr_label.grid(row=0, column=0, columnspan=3)

        # self.label = []
        # size 2400 to store separate ratings for left and right eye
        # for a sample with index i, 2i indicates rating for left eye, and 2i+1 indicates rating for right eye
        # self.label = np.zeros(2400)-1

        self.label_left_iris = np.zeros(1200)-1
        self.label_right_iris = np.zeros(1200)-1
        self.label_left_eye = np.zeros(1200)-1
        self.label_right_eye = np.zeros(1200)-1

        #
        print("Interation: ", self.idx)
        print("Image index: ", int(np.floor(self.pair_list[self.idx]/2)))
        print("Index: ", self.pair_list[self.idx])
        # print(self.label)

        self.root.mainloop()

    def click(self, rating):

        # self.label.append(rating)
        # self.label[self.pair_list[self.idx]] = rating

        if int(np.floor(self.pair_list[self.idx])) % 2 == 0:
            if self.is_eye is True:
                self.is_eye = False
                self.label_left_eye[int(np.floor(self.pair_list[self.idx]/2))] = rating
            else:
                self.is_eye = True
                self.label_left_iris[int(np.floor(self.pair_list[self.idx] / 2))] = rating
                self.idx += 1
        else:
            if self.is_eye is True:
                self.is_eye = False
                self.label_right_eye[int(np.floor(self.pair_list[self.idx] / 2))] = rating
            else:
                self.is_eye = True
                self.label_right_iris[int(np.floor(self.pair_list[self.idx] / 2))] = rating
                self.idx += 1

        # print("Interation: ", self.idx-1)
        # print("Image index: ", int(np.floor(self.pair_list[self.idx-1]/2)))
        # print("Index: ", self.pair_list[self.idx-1])
        # print(self.label)
        print("Left Eye Label: ", self.label_left_eye)
        print("Left Iris Label: ", self.label_left_iris)
        print("Right Eye Label: ", self.label_right_eye)
        print("Right Iris Label: ", self.label_right_iris)

        if self.idx > len(self.pair_list)-1:
            print("Attempt to disable buttons")

            self.button_good['state'] = 'disabled'
            self.button_ok['state'] = 'disabled'
            self.button_poor['state'] = 'disabled'
            self.button_exit['state'] = 'normal'

        else:
            print("Interation: ", self.idx)
            print("Image index: ", int(np.floor(self.pair_list[self.idx] / 2)))
            print("Index: ", self.pair_list[self.idx])
            # print("Left Eye Label: ", self.label_left_eye)
            # print("Left Iris Label: ", self.label_left_iris)
            # print("Right Eye Label: ", self.label_right_eye)
            # print("Right Iris Label: ", self.label_right_iris)

            self.progress_label = tk.Label(self.root, text="Image " + str(self.idx + 1) + "of " + str(len(self.pair_list)))
            self.progress_label.grid(row=3, column=1)

            self.img_face = cv2.imread(os.path.join(FACE_ROOT, str(int(np.floor(self.pair_list[self.idx]/2))) + ".png"), cv2.IMREAD_UNCHANGED)
            self.img_seg = cv2.imread(os.path.join(SEG_ROOT, str(int(np.floor(self.pair_list[self.idx]/2))) + ".png"), cv2.IMREAD_UNCHANGED)
            self.img_face_seg = cv2.imread(os.path.join(FACE_SEG_ROOT, str(int(np.floor(self.pair_list[self.idx]/2))) + ".png"), cv2.IMREAD_UNCHANGED)
            # cv2.imshow("image", self.img_face)
            self.img_face_tk = ImageTk.PhotoImage(Image.fromarray(self.img_face[:, :, ::-1]))
            self.img_seg_tk = ImageTk.PhotoImage(Image.fromarray(self.img_seg[:, :, ::-1]))
            self.img_face_seg_tk = ImageTk.PhotoImage(Image.fromarray(self.img_face_seg[:, :, ::-1]))

            # display label
            self.face_img_label = tk.Label(self.root, image=self.img_face_tk)
            self.face_img_label.grid(row=1, column=0, rowspan=2)
            self.seg_img_label = tk.Label(self.root, image=self.img_seg_tk)
            self.seg_img_label.grid(row=1, column=1, rowspan=2)
            self.face_seg_img_label = tk.Label(self.root, image=self.img_face_seg_tk)
            self.face_seg_img_label.grid(row=1, column=2, rowspan=2)
            # disable buttons when reaches end of the list

            if int(np.floor(self.pair_list[self.idx])) % 2 == 0:
                if self.is_eye is True:
                    self.lr_label = tk.Label(self.root, text="Please rate the LEFT EYE of the images below", font=("Arial", 20, 'bold'))
                    self.lr_label.grid(row=0, column=0, columnspan=3)
                else:
                    self.lr_label = tk.Label(self.root, text="Please rate the LEFT IRIS of the images below", font=("Arial", 20, 'bold'))
                    self.lr_label.grid(row=0, column=0, columnspan=3)
            else:
                if self.is_eye is True:
                    self.lr_label = tk.Label(self.root, text="Please rate the RIGHT EYE of the images below", font=("Arial", 20, 'bold'))
                    self.lr_label.grid(row=0, column=0, columnspan=3)
                else:
                    self.lr_label = tk.Label(self.root, text="Please rate the RIGHT IRIS of the images below", font=("Arial", 20, 'bold'))
                    self.lr_label.grid(row=0, column=0, columnspan=3)

    def save_label(self):
        # np.save("D:/Gaze_Rand_Select_6N/n=200/left_eye_labels_0727.npy", self.label_left_eye)
        # np.save("D:/Gaze_Rand_Select_6N/n=200/left_iris_labels_0727.npy", self.label_left_iris)
        # np.save("D:/Gaze_Rand_Select_6N/n=200/right_eye_labels_0727.npy", self.label_right_eye)
        # np.save("D:/Gaze_Rand_Select_6N/n=200/right_iris_labels_0727.npy", self.label_right_iris)
        #
        np.save("D:/Gaze_Rand_Select_6N/data_collection/first600/left_eye_labels_0_599_nina.npy", self.label_left_eye)
        np.save("D:/Gaze_Rand_Select_6N/data_collection/first600/left_iris_labels_0_599_nina.npy", self.label_left_iris)
        np.save("D:/Gaze_Rand_Select_6N/data_collection/first600/right_eye_labels_0_599_nina.npy", self.label_right_eye)
        np.save("D:/Gaze_Rand_Select_6N/data_collection/first600/right_iris_labels_0_599_nina.npy", self.label_right_iris)

        print("Label saved to destination")
        self.button_exit['state'] = 'disabled'
        self.button_exit['text'] = 'Labels saved to destination'
