"""
July 7, 2023, by Nina

to generate mixup matrices for eyediap dataset

for each image, select two images from other subjects for mixup
generated matrices will be size of 6487*16 for cs and 8337*16 for ds (row (i) -> image id, colum (j) -> subject)
If there is a mix, than matrix[i][j] stores the image id from subject j which is the mix to image i
If there is no mix than matrix[i][j] is -1

selection criteria:
    * the smallest two gaze angular differences, and should be less than 15.
    * iou of the two seg map greater than 0.5
    * if either criteria is not satisfied, is not considered a successful mix

"""
import numpy as np
import cv2
import os
import tools.utils as tool
from data.mixup import MixupProcessor

# directories
DIR_GT = ['/data/eyediap/gt_eyediap_cs_with_conf_v3_bh.npy', '/data/eyediap/gt_eyediap_ds_with_conf_v3_bh.npy']
DIR_IMAGE = ['/data/eyediap/normalized_face_224_cs_bh', '/data/eyediap/normalized_face_224_ds_bh']
DIR_SEG = ['/data/eyediap/seg_cs_bh', '/data/eyediap/seg_ds_bh']

#
# gt = np.load(DIR_GT, allow_pickle=True).item()
# # keys = gt.keys()
# num_images = len(gt['fn'])
NUM_SUBJECTS = 16


def get_image_ids_for_subject(subject_id, num_images, gt):
    # gt = np.load(DIR_GT, allow_pickle=True).item()
    # num_images = len(gt['fn'])
    # num_subjects = 16
    ids = []

    for i in range(num_images):
        if gt['subject_cid'][i] == subject_id:
            ids.append(i)
    # returns image ids that
    return ids


def get_angular_errors(subject_i, image_ids, gt):
    angular_errors = []
    gaze_i = gt['gaze'][subject_i]
    for i in image_ids:
        gaze_id = gt['gaze'][i]
        angular_errors.append(tool.angular_error(gaze_i, gaze_id))

    # return all angular errors of selected image to all images of subject j
    return angular_errors


# ideally each image matches 1 image each from a different subject
def mix(ty):
    # load ground truth dictionary
    # gt = np.load(DIR_GT, allow_pickle=True).item()
    # num_images = len(gt['fn'])
    # num_subjects = 16
    if ty == 'cs':
        gt = np.load(DIR_GT[0], allow_pickle=True).item()
        dir_seg = DIR_SEG[0]
        num_images = len(gt['fn'])
    elif ty == 'ds':
        gt = np.load(DIR_GT[1], allow_pickle=True).item()
        dir_seg = DIR_SEG[1]
        num_images = len(gt['fn'])
    else:
        # mixing cs with ds
        gt_cs = np.load(DIR_GT[0], allow_pickle=True).item()
        gt_ds = np.load(DIR_GT[1], allow_pickle=True).item()
        gt = {key: gt_cs[key] + gt_ds[key] for key in gt_cs[0]}
        num_images = len(gt['fn'])

    pair_matrix_1 = np.zeros([num_images, NUM_SUBJECTS]) - 1
    pair_matrix_2 = np.zeros([num_images, NUM_SUBJECTS]) - 1
    # no_seg_cnt = 0

    # find all image ids corresponding to subject j
    image_ids = []
    for j in range(NUM_SUBJECTS):
        image_ids_j = get_image_ids_for_subject(j, num_images, gt)
        image_ids.append(image_ids_j)

    for i in range(num_images):
        subject_i = gt['subject_cid'][i]
        fn_i = gt['fn'][i]

        if os.path.isfile(os.path.join(DIR_SEG[0], fn_i + '.png')):
            seg_map_i = cv2.imread(os.path.join(DIR_SEG[0], fn_i + '.png'), cv2.IMREAD_UNCHANGED)
        elif os.path.isfile(os.path.join(DIR_SEG[1], fn_i + '.png')):
            seg_map_i = cv2.imread(os.path.join(DIR_SEG[1], fn_i + '.png'), cv2.IMREAD_UNCHANGED)
        else:
            continue

        for j in range(NUM_SUBJECTS):
            # for all subjects that are not subject i
            if j != subject_i and j != 11 and j != 12:
                # get all the ids of the images that are subject j
                # Done: more efficient, get image_ids before entering for loop
                # image_ids = get_image_ids_for_subject(j)
                # calculate all the angular errors of each image in subject j to image i
                angular_errors = get_angular_errors(i, image_ids[j], gt)
                cnt = 0
                stop = False
                while not stop:
                    # if minimum of all angular errors is within the acceptable range, get the n
                    if np.amin(angular_errors) <= 15 and cnt < 2:
                        # index of min angular difference
                        min_ang_error = np.argmin(angular_errors)
                        # get index of the corresponding image j that has this index
                        image_j = image_ids[j][min_ang_error]
                        # get file name and read the image
                        fn_j = gt['fn'][image_j]

                        # check file existence
                        if os.path.isfile(os.path.join(DIR_SEG[0], fn_j + '.png')):
                            seg_map_j = cv2.imread(os.path.join(DIR_SEG[0], fn_j + '.png'), cv2.IMREAD_UNCHANGED)
                            # stop -> when an optimal match is found (min angular error AND eye overlap iou > 0.5
                        elif os.path.isfile(os.path.join(DIR_SEG[1], fn_j + '.png')):
                            seg_map_j = cv2.imread(os.path.join(DIR_SEG[1], fn_j + '.png'), cv2.IMREAD_UNCHANGED)
                        else:
                            angular_errors[min_ang_error] = 181
                            continue

                        if MixupProcessor.eye_iou(seg_map_i, seg_map_j) > 0.5 and i != pair_matrix_1[image_j][subject_i] and i != pair_matrix_2[image_j][subject_i]:
                            # pair_matrix_1[i][j] = image_j
                            # stop = True
                            # consider case if a matches b as top 2 choice, and b matches a again
                            # if i != pair_matrix_1[image_j][subject_i] and i != pair_matrix_2[image_j][subject_i]:
                            # top match
                            if cnt == 0:
                                pair_matrix_1[i][j] = image_j
                                angular_errors[min_ang_error] = 181
                                cnt += 1
                                # second match
                            elif cnt == 1:
                                pair_matrix_2[i][j] = image_j
                                angular_errors[min_ang_error] = 181
                                cnt += 1

                                # else:
                                #     angular_errors[min_ang_error] = 181
                        else:
                            angular_errors[min_ang_error] = 181
                    # else:
                    #     angular_errors[min_ang_error] = 181

                        # when all the angular error is greater than 15, stop -> no match found
                        # or have already matched two images
                    else:
                        stop = True
        # print(i, pair_matrix_1[i], "\n", pair_matrix_2[i], "\n")

    return pair_matrix_1, pair_matrix_2


pair_mat_cs, pair_mat_2_cs = mix('cs')
np.save("/data/eyediap/mixup_matrices/top_pair_save_no_duplicate_cs.npy", pair_mat_cs)
np.save("/data/eyediap/mixup_matrices/second_pair_save_no_duplicate_cs.npy", pair_mat_2_cs)
num_matches_all = ((pair_mat_cs >= 0).sum()) + ((pair_mat_2_cs >= 0).sum())
print("number of matches for image id 0:", num_matches_all)
print(pair_mat_cs, "\n", pair_mat_2_cs)

pair_mat_ds, pair_mat_2_ds = mix('ds')
np.save("/data/eyediap/mixup_matrices/top_pair_save_no_duplicate_ds.npy", pair_mat_ds)
np.save("/data/eyediap/mixup_matrices/second_pair_save_no_duplicate_ds.npy", pair_mat_2_ds)
num_matches_all = ((pair_mat_ds >= 0).sum()) + ((pair_mat_2_ds >= 0).sum())
print("number of matches for image id 0:", num_matches_all)
print(pair_mat_ds, "\n", pair_mat_2_ds)


pair_mat, pair_mat_2 = mix(['cs', 'ds'])
np.save("/data/eyediap/mixup_matrices/top_pair_save_no_duplicate_cs_and_ds.npy", pair_mat)
np.save("/data/eyediap/mixup_matrices/second_pair_save_no_duplicate_cs_and_ds.npy", pair_mat_2)
num_matches_all = ((pair_mat >= 0).sum()) + ((pair_mat_2 >= 0).sum())
print("number of matches for image id 0:", num_matches_all)
print(pair_mat, "\n", pair_mat_2)

