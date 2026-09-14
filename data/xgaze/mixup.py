import numpy as np
import os
import cv2
import argparse
import time
from tqdm import tqdm
from tools.mixup import compareImgIOU
from tools.utils import angular_error

IOU_THRESHOLD = 0.8
GAZE_THRESHOLD = 5 * np.pi / 180
VALID_CAMS = [0,1,3,4,5,6]

def loadMergedGT(path):
    data = np.load(path, allow_pickle=True).item()
    return data

def createMatrixMerged(data_dict, start_idx, end_idx, matrix_save_path, diff_matrix_save_path):
    # matrix size is num_imgs x num_subjects
    unique_subject_id = list(set(data_dict["subject_cid"]))
    num_subjects = len(unique_subject_id)

    if end_idx == -1:
        end_idx = len(data_dict["subject_cid"])
    
    num_imgs = end_idx - start_idx

    result_matrix = np.ones((num_imgs, num_subjects)) * -1
    diff_matrix = np.ones((num_imgs, num_subjects)) * 1000

    ## for each image find the nearest image for each subject
    for i in tqdm(range(start_idx, end_idx)):
        # print("Currently at: " + str(i) + "/" + str(num_imgs) + " row")
        # if iou_dict["IOU_L"][i] < 0.1 or iou_dict["IOU_R"][i] < 0.1:
        #     result_matrix[i, :] = -1
        #     diff_matrix[i, :] = -1
        #     # count = np.count_nonzero(result_matrix == -1)
        #     # print("skip number of -1s: ", count)
        # else: 
        ## path of the segmap of the original image
        cam_id_out = data_dict['cam_cid'][i]
        if cam_id_out not in VALID_CAMS:
            continue

        img1_segmap_path = os.path.join(ROOT, 'seg', f'{data_dict["id"][i]}.png')
        if not os.path.isfile(img1_segmap_path):
            continue
        img1 = np.asarray(cv2.imread(img1_segmap_path, cv2.IMREAD_UNCHANGED))
           
        findNearestForMatrixMerged(data_dict, img1, i, result_matrix, diff_matrix, start_idx)

            #     count = np.count_nonzero(result_matrix == -1)
            #     count_non_zero = np.count_nonzero(result_matrix)
            # print("number of -1s: ", count)
            # print("number of non 0 s: ", count_non_zero)
    count = np.count_nonzero(result_matrix != -1)
    # neg_percent = count / (num_imgs * num_subjects)
    # save this matrix
    print("number of -1s: ", count)
    # print("number of non 0 s: ", count_non_zero)
    print("shape of result_matrix: ", result_matrix.shape)
    print("shape of diff_matrix: ", diff_matrix.shape)
    np.save(matrix_save_path, result_matrix)
    np.save(diff_matrix_save_path, diff_matrix)
    return True

def findNearestForMatrixMerged(data_dict, img1, i, result_matrix, diff_matrix, start_idx):
    # first find all the information related to the input image

    # seems no need to copy
    # dataset_info_input = {}
    # keys = [k for k in list(data_dict.keys()) if type(data_dict[k]) is list]
    # for each in keys:
        # dataset_info_input[each] = data_dict[each][i]

    x_gaze1 = data_dict["gaze"][i][0]
    y_gaze1 = data_dict["gaze"][i][1]
    gaze1 = data_dict["gaze"][i]

    cam1 = data_dict["cam_cid"][i]
    id1 = data_dict["subject_cid"][i]

    row_id = i - start_idx

    for j in range(len(data_dict["id"])):
        ## compare with the given subject and not the same image
        if data_dict["cam_cid"][j] != cam1: continue

        id2 = data_dict["subject_cid"][j]
        if id2 == id1: continue

        img2_segmap_path = os.path.join(ROOT, 'seg', f'{data_dict["id"][j]}.png')
        if not os.path.isfile(img2_segmap_path):
            continue

        ## skip if the image IOU_L/R is less than 0.1
        # if iou_dict["IOU_L"][j] < 0.1 or iou_dict["IOU_R"][j] < 0.1: continue

        ## skip the image if the gaze difference is larger or equal to 6 degrees
        curr_index_gaze_value = data_dict["gaze"][j]
        x_gaze2 = curr_index_gaze_value[0]
        y_gaze2 = curr_index_gaze_value[1]
        if abs(x_gaze1 - x_gaze2) >= GAZE_THRESHOLD or abs(y_gaze1 - y_gaze2) >= GAZE_THRESHOLD: continue

        curr_min_gaze_diff = diff_matrix[row_id, id2]

        gaze_diff = angular_error(gaze1, curr_index_gaze_value)
        if gaze_diff < curr_min_gaze_diff:
            ## load segmap of the image2 and compare the iou   
            img2 = np.asarray(cv2.imread(img2_segmap_path, cv2.IMREAD_UNCHANGED))
            IOU_eyes = compareImgIOU(img1, img2)
            if IOU_eyes >= IOU_THRESHOLD:
                curr_min_gaze_diff = gaze_diff
                result_matrix[row_id, id2] = j
                diff_matrix[row_id, id2] = gaze_diff
                # print(f"Found: {i} - {j}, iou = {IOU_eyes}")

def np_to_dict(np_path, dict_path):
    matrix = np.load(np_path)
    n, n_id = matrix.shape
    D = {}
    for i in tqdm(range(n)):
        # if i%18 not in (0,1,3,4,5): continue
        L = []
        for j in range(n_id):
            if matrix[i,j] != -1:
                L.append(int(matrix[i,j]))
        # if len(L) > 0:
        #     D[i] = L
        D[i]=L
    np.save(dict_path, D)


if __name__ == '__main__':
    print(angular_error)
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", type=int, default=0, help="starting index")
    parser.add_argument("-e", "--end", type=int, default=-1, help="ending index")

    args = parser.parse_args()

    start_time = time.time()
    ROOT = "/data/xgaze/448/"

    path = "/data/xgaze/448/gt_p17y17b16_with_confs_v3.npy"
    # iou_path = '/data/leo/GazeTorch/temp/gc3d-IOU-data/iou_dict_little.npy'

    data = loadMergedGT(path) ## load the data into dict
    # iou_dict = loadMergedGT(iou_path) ## load the iou dict

    output_path = "temp_results/250528/xgaze-mixup-iou8_" + str(args.start) + "-" + str(args.end)
    os.makedirs(output_path, exist_ok=True)
    createMatrixMerged(data, args.start, args.end, os.path.join(output_path, "matrix.npy"), os.path.join(output_path, "gaze_angular_error_matrix.npy"))
    end_time = time.time()
    elapsed_time = end_time - start_time
    with open("temp_exe_time_xgaze_with_iou_" + str(args.start) + "-" + str(args.end) + ".txt", "w") as file:
        print('Execution time:', elapsed_time, 'seconds', file=file)
        print("little dataset matrix with iou", file=file)

    # np_to_dict(
    #     '/root/GazeTorch/temp_results/250522/xgaze-data-mixup/matrix.npy',
    #     '/root/GazeTorch/temp_results/250522/xgaze-data-mixup/matrixD.npy'
    # )