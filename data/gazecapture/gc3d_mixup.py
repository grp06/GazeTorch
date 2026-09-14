import math
import sys
import numpy as np
import os
import shutil
import cv2
import time
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--start", type=int, required=True)
parser.add_argument("-e", "--end", type=int, required=True)

GAZE_THRESHOLD = 4 * np.pi / 180
HP_THRESHOLD_X = 0.15
HP_THRESHOLD_Y = 0.3
SEGMAP_IOU_THRESHOLD = 0.5
N_LARGEST_SUBJECTS = 80

PATH_TARGET = 'data/gazecapture/mixup_target_ids.npy'
mixup_target_ids = np.load(PATH_TARGET)

SUBJECT_ID_RANK = 'data/gazecapture/mixup_subject_id_rank.npy'
subject_id_rank = np.load(SUBJECT_ID_RANK, allow_pickle=True).item()

def loadMergedGT(path):
    ## load the dataset
    data = np.load(path, allow_pickle=True).item()
    return data

def createMatrixMerged(data_dict, matrix_save_path, diff_matrix_save_path, start_index, end_index):
    # matrix size is num_imgs x num_subjects
    unique_subject_id = list(set(data_dict["subject_cid"]))
    # num_subjects = len(unique_subject_id)
    num_subjects = N_LARGEST_SUBJECTS

    if end_index == -1:
        end_index = len(data_dict["fn"])

    num_imgs = end_index - start_index

    ## initialize the matrix to -1s
    result_matrix = np.ones((num_imgs, num_subjects)) * -1
    diff_matrix = np.ones((num_imgs, num_subjects)) * -1

    ## for each image find the nearest image for each subject
    for i in tqdm(range(start_index, end_index)):
        # print("Currently at: " + str(i) + "/" + str(num_imgs) + " row")

        ## path of the segmap of the original image and read the image
        img1_segmap_path = os.path.join(ROOT, 'seg', f'{data["fn"][i]}.png')
        if not os.path.isfile(img1_segmap_path):
            continue
        img1 = np.asarray(cv2.imread(img1_segmap_path, cv2.IMREAD_UNCHANGED))
        
        ## find the nearest images for all subjects need to be compared
        # findNearestForMatrixMerged(data_dict, iou_dict, img1, i, sorted_subject_cids[i], use_little, result_matrix, diff_matrix, start_index)
        findNearestForMatrixMerged(data_dict, img1, i, unique_subject_id, result_matrix, diff_matrix, start_index)
    
    # save this matrix
    count = np.count_nonzero(result_matrix == -1)
    print("number of -1s: ", count)
    print("shape of result_matrix: ", result_matrix.shape)
    print("shape of diff_matrix: ", diff_matrix.shape)
    np.save(matrix_save_path, result_matrix)
    np.save(diff_matrix_save_path, diff_matrix)
    return True

def findNearestForMatrixMerged(data_dict, img1, i, compare_subject_ids, result_matrix, diff_matrix, start_index):
    # first find all the information needed of the input image
    i_gaze = data_dict["gaze"][i]
    x_gaze1, y_gaze1 = i_gaze

    y_headpose1, x_headpose1 = data_dict["headpose"][i]

    id1 = data_dict['subject_cid'][i]

    row_id = i - start_index

    counter_L1 = 0
    counter_L2 = 0

    # n = len(data_dict["fn"])

    ## go over the WHOLE dataset to find the closest given the compare subject list
    # for data_dir_index in range(n):
    for data_dir_index in mixup_target_ids:
        id2 = data_dict["subject_cid"][data_dir_index]
        # ## compare with the subjects in the compare_subject_ids list but not with the same subject
        if id1 == id2: continue

        ## skip the image if the gaze difference is larger or equal to 6 degrees
        curr_index_gaze_value = data_dict["gaze"][data_dir_index]
        x_gaze2, y_gaze2 = curr_index_gaze_value
        if abs(x_gaze1 - x_gaze2) >= GAZE_THRESHOLD or abs(y_gaze1 - y_gaze2) >= GAZE_THRESHOLD: continue

        ## skip the image if the headpose difference is large
        y_headpose2, x_headpose2 = data_dict["headpose"][data_dir_index]
        if abs(x_headpose1 - x_headpose2) >= HP_THRESHOLD_X or abs(y_headpose1 - y_headpose2) >= HP_THRESHOLD_Y: continue


        counter_L1 += 1
        
        ## calculate the 2 gaze difference
        gaze_diff = angular_error(i_gaze, curr_index_gaze_value)

        ## j is the index of the subject cid helps locating the matrix position
        # LZH: compare_subject_ids has been disabled
        j = subject_id_rank.get(id2, -1)

        ## get the curr_min_gaze_difference of the subjects
        curr_min_gaze_diff = diff_matrix[row_id, j]

        if gaze_diff < curr_min_gaze_diff or curr_min_gaze_diff == -1:

            ## load segmap of the image2 and compare the iou   
            img2_segmap_path = os.path.join(ROOT, 'seg', f'{data_dict["fn"][data_dir_index]}.png')
            if not os.path.isfile(img2_segmap_path):
                continue

            img2 = np.asarray(cv2.imread(img2_segmap_path, cv2.IMREAD_UNCHANGED))
            IOU_eyes = compareImgIOU(img1, img2)

            counter_L2 += 1

            if IOU_eyes > SEGMAP_IOU_THRESHOLD:
                ## update the buffer
                diff_matrix[row_id, j] = gaze_diff
                result_matrix[row_id, j] = data_dir_index  
    
    # print("L1", counter_L1, "L2", counter_L2)


def compareImgIOU(img1, img2):
    """
        this function calculate and compare the IOU value.
        IOU is large means the eye areas of 2 image are close to each other;
        IOU is small means the eye areas are different.
    """
    if img1.shape != (224, 224, 4) or img2.shape != (224, 224, 4):
        return -1
    else:
        # for eye map IOU value
        img1_left_eye = img1[:, :, 0]
        img1_left_eye[img1_left_eye == 2] = 0
        img1_right_eye = img1[:, :, 2]
        img1_right_eye[img1_right_eye == 2] = 0

        img2_left_eye = img2[:, :, 0]
        img2_left_eye[img2_left_eye == 2] = 0
        img2_right_eye = img2[:, :, 2]
        img2_right_eye[img2_right_eye == 2] = 0

        # run the IOU function
        IOU_value = calculateIOU(img1_left_eye, img1_right_eye, img2_left_eye, img2_right_eye)

        return IOU_value

def calculateIOU(img1_left_eye, img1_right_eye, img2_left_eye, img2_right_eye):
    """
        this function calculate the intersection of eyes area (IOU) of 2 images.
    """
    d1 = sum(sum(img1_left_eye == 1) + sum(img1_right_eye == 1))
    d2 = sum(sum(img2_left_eye == 1) + sum(img2_right_eye == 1))

    n1_img = img1_left_eye + img2_left_eye
    n2_img = img1_right_eye + img2_right_eye

    n = sum(sum(n1_img == 2) + sum(n2_img == 2))
    d = d1 + d2 - n

    return n / d

def pitchyaw_to_xyz(pitchyaw):
    pitch, yaw = pitchyaw
    return (math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(yaw) * math.cos(pitch))

def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = pitchyaw_to_xyz(a) if len(a) == 2 else a
    b = pitchyaw_to_xyz(b) if len(b) == 2 else b

    ab = np.sum(np.multiply(a, b))
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-7, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-7, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))
    if similarity > 1:
        similarity = 1
    # print(similarity)
    return np.arccos(similarity) * 180.0 / np.pi

if __name__ == '__main__':
    

    args = parser.parse_args()
    start_index = args.start
    end_index = args.end

    ROOT = "/data/gazecapture_3d"
    PATH_GT_DEFAULT = '/data/gazecapture_3d/gt_p17y17_with_conf_v3.npy'

    print("loading the dataset")
    data = loadMergedGT(PATH_GT_DEFAULT) ## load the data into dict

    print("dataset is loaded")

    output_path = f"temp_results/250602/" ###
    os.makedirs(output_path, exist_ok=True)
    # start_time = time.time()
    createMatrixMerged(data,
                       os.path.join(output_path, f"matrix_{start_index}-{end_index}.npy"),
                       os.path.join(output_path, f"gaze_angular_error_matrix_{start_index}-{end_index}.npy"),
                        start_index, end_index)
    # end_time = time.time()
    # elapsed_time = end_time - start_time
    # with open(f"temp_exe_time_filtered-gc3d-data-mixup_with_iou_skip_range_{start_index}-{end_index}_test.txt", "w") as file: ###
    #     print('Execution time:', elapsed_time, 'seconds', file=file)
    #     print("filtered dataset matrix with iou skip", file=file) ###