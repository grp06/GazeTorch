# by Mahak. for HQ dataset.

import math
import sys
import numpy as np
import os
import shutil
import cv2

"""
Usage:

python main.py <num_similar_images> <image name for which similar images needed> <result path>
"""

subject_ID_dict = {'Alyssa': 0,
                   'Amirul': 1,
                   'Charlotte': 2,
                   'Chunsheng': 3,
                   'Cidalia': 4,
                   'Han': 5,
                   'Haolong': 6,
                   'Huan': 7,
                   'Juntao': 8,
                   'Kushagra': 9,
                   'Mahak': 10,
                   'Patricia': 11,
                   'Ruturaj': 12,
                   'Sophia': 13,
                   'Soumil': 14,
                   'Zheng': 15,
                   'Zhixiang': 16,
                   'Luke': 17,
                   'Vaibhav': 18,
                   'Yifan': 19,
                   'Zhenhao': 20,
                   '30016651': 21,
                   '84103012': 22,
                   '84119603': 23,
                   '84164595': 24,
                   '84165477': 25,
                   '84193300': 26,
                   '84237287': 27,
                   '84237977': 28,
                   '84255776': 29,
                   'c00587034': 30,
                   'd00825553': 31,
                   'f00663264': 32,
                   'h00699232': 33,
                   'h30042717': 34,
                   'l00420782': 35,
                   'l00500333': 36,
                   'l00594894': 37,
                   'l00800631': 38,
                   'l00800673': 39,
                   'm00513568': 40,
                   's00580383': 41,
                   's00800629': 42,
                   't00808470': 43,
                   'w00593886': 44,
                   'wx1078986': 45,
                   'wx1086633': 46,
                   'wx1098156': 47,
                   'wx1111947': 48,
                   'wx5331072': 49,
                   'y00618575': 50,
                   'y00698855': 51,
                   'y00800621': 52,
                   'z00568245': 53,
                   'z00663694': 54,
                   'z00698854': 55,
                   'z30042962': 56,
                   'z84244683': 57,
                   'd00467038': 58,
                   'g30027324': 59,
                   'h00514358': 60,
                   'l00412077': 61,
                   'l00428973': 62,
                   'l00804666': 63,
                   'm00538487': 64,
                   'p00345902': 65,
                   's00818875': 66,
                   'I00800673': 67,
                   'g00514301': 68,
                   'h00580190': 69,
                   'h84186233': 70,
                   'w00800630': 71,
                   'z00502608': 72,
                   'z00520849': 73}

ID_subject_dict = {0: 'Alyssa',
                   1: 'Amirul',
                   2: 'Charlotte',
                   3: 'Chunsheng',
                   4: 'Cidalia',
                   5: 'Han',
                   6: 'Haolong',
                   7: 'Huan',
                   8: 'Juntao',
                   9: 'Kushagra',
                   10: 'Mahak',
                   11: 'Patricia',
                   12: 'Ruturaj',
                   13: 'Sophia',
                   14: 'Soumil',
                   15: 'Zheng',
                   16: 'Zhixiang',
                   17: 'Luke',
                   18: 'Vaibhav',
                   19: 'Yifan',
                   20: 'Zhenhao',
                   21: '30016651',
                   22: '84103012',
                   23: '84119603',
                   24: '84164595',
                   25: '84165477',
                   26: '84193300',
                   27: '84237287',
                   28: '84237977',
                   29: '84255776',
                   30: 'c00587034',
                   31: 'd00825553',
                   32: 'f00663264',
                   33: 'h00699232',
                   34: 'h30042717',
                   35: 'l00420782',
                   36: 'l00500333',
                   37: 'l00594894',
                   38: 'l00800631',
                   39: 'l00800673',
                   40: 'm00513568',
                   41: 's00580383',
                   42: 's00800629',
                   43: 't00808470',
                   44: 'w00593886',
                   45: 'wx1078986',
                   46: 'wx1086633',
                   47: 'wx1098156',
                   48: 'wx1111947',
                   49: 'wx5331072',
                   50: 'y00618575',
                   51: 'y00698855',
                   52: 'y00800621',
                   53: 'z00568245',
                   54: 'z00663694',
                   55: 'z00698854',
                   56: 'z30042962',
                   57: 'z84244683',
                   58: 'd00467038',
                   59: 'g30027324',
                   60: 'h00514358',
                   61: 'l00412077',
                   62: 'l00428973',
                   63: 'l00804666',
                   64: 'm00538487',
                   65: 'p00345902',
                   66: 's00818875',
                   67: 'I00800673',
                   68: 'g00514301',
                   69: 'h00580190',
                   70: 'h84186233',
                   71: 'w00800630',
                   72: 'z00502608',
                   73: 'z00520849'}


def loadData(path1, path2, path3):
    data1 = np.load(path1, allow_pickle=True).item()
    data2 = np.load(path2, allow_pickle=True).item()
    data3 = np.load(path3, allow_pickle=True).item()

    return data1, data2, data3


def loadMergedGT(path):
    data = np.load(path, allow_pickle=True).item()
    return data


def returnAbsDiff(arr1, arr2):
    if len(arr1) != len(arr2):
        return 0
    result = 0
    for i in range(len(arr1)):
        result += abs(arr1[i] - arr2[i])
    return result


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

def findNearestForMatrix(data_dict1, data_dict2, data_dict3, i, compare_subject_id):
    # first find all the information related to the input image
    dataset_info_input = {}
    compare_subject_id_name = ID_subject_dict[compare_subject_id]

    if i < len(data_dict1):

        # search in dict1 and dict2
        dataset_class = 0
        for each in list(data_dict1.keys()):
            dataset_info_input[each] = data_dict1[each][i]

    elif i < len(data_dict2):
        # search in dict1 and dict2
        dataset_class = 1
        index = i - len(data_dict1)
        for each in list(data_dict2.keys()):
            dataset_info_input[each] = data_dict2[each][index]
    else:
        # search in dict3
        dataset_class = 2
        index = i - (len(data_dict1) + len(data_dict2))
        for each in list(data_dict3.keys()):
            dataset_info_input[each] = data_dict3[each][index]

    # now I will try to find the closest image to the input image from the list of images of
    # the person compare_subject_id_name

    closest_img_info = {}
    # Case 1 the target subject ID is only present in alex data folders
    index_closest = 0
    if compare_subject_id < 21:
        # search in both data_dict1 and data_dict2
        curr_min_gaze_diff = math.inf
        data_dir_index = 0  # just an index for use in while loop
        while data_dir_index < len(data_dict1["fn"]):
            if data_dict1["dir1"][data_dir_index] == compare_subject_id_name and (
                    data_dict1["dir2"][data_dir_index] != dataset_info_input["dir2"] or data_dict1["fn"][
                data_dir_index] != dataset_info_input["fn"] or dataset_class != 0):
                curr_index_gaze_value = data_dict1["gaze"][data_dir_index]
                gaze_diff = returnAbsDiff(dataset_info_input["gaze"], curr_index_gaze_value)
                if gaze_diff < curr_min_gaze_diff:
                    curr_min_gaze_diff = gaze_diff
                    for each in list(data_dict1.keys()):
                        closest_img_info[each] = data_dict1[each][data_dir_index]
                    closest_img_info["ind"] = data_dir_index
                    closest_img_info["gaze_diff"] = curr_min_gaze_diff
            data_dir_index += 1

        data_dir_index = 0
        dataset_2_flag = False
        while data_dir_index < len(data_dict2["fn"]):
            if data_dict2["dir1"][data_dir_index] == compare_subject_id_name and (
                    data_dict2["dir2"][data_dir_index] != dataset_info_input["dir2"] or data_dict2["fn"][
                data_dir_index] != dataset_info_input["fn"] or dataset_class != 1):
                curr_index_gaze_value = data_dict2["gaze"][data_dir_index]
                gaze_diff = returnAbsDiff(dataset_info_input["gaze"], curr_index_gaze_value)
                if gaze_diff < curr_min_gaze_diff:
                    curr_min_gaze_diff = gaze_diff
                    for each in list(data_dict2.keys()):
                        closest_img_info[each] = data_dict2[each][data_dir_index]
                    closest_img_info["ind"] = data_dir_index
                    closest_img_info["gaze_diff"] = curr_min_gaze_diff
                    dataset_2_flag = True
            data_dir_index += 1

        closest_img_info["fn"] = int(closest_img_info["fn"]) % 100
        if dataset_2_flag:
            # image was found in data_dict2
            index_closest = len(data_dict1["fn"]) + closest_img_info["ind"]
        else:
            # closest_image_path_name = os.path.join("/data/alex_tablet2",
            #                                        "{}/{}/face_224/{:03d}.png".format(closest_img_info["dir1"],
            #                                                                  closest_img_info["dir2"],
            #                                                                  closest_img_info["fn"]))
            index_closest = closest_img_info["ind"]
    else:
        # the target subject ID have its pictures in HQ data folder
        # search in both data_dict3
        curr_min_gaze_diff = math.inf
        data_dir_index = 0  # just an index for use in while loop
        while data_dir_index < len(data_dict3["id"]):
            if data_dict3["dir2"][data_dir_index] == compare_subject_id_name and (
                    data_dict3["dir1"][data_dir_index] != dataset_info_input["dir1"] or data_dict3["id"][
                data_dir_index] != dataset_info_input["id"] or dataset_class != 2):
                curr_index_gaze_value = data_dict3["gaze"][data_dir_index]
                gaze_diff = returnAbsDiff(dataset_info_input["gaze"], curr_index_gaze_value)
                if gaze_diff < curr_min_gaze_diff:
                    curr_min_gaze_diff = gaze_diff
                    for each in list(data_dict3.keys()):
                        closest_img_info[each] = data_dict3[each][data_dir_index]
                    closest_img_info["ind"] = data_dir_index
                    closest_img_info["gaze_diff"] = curr_min_gaze_diff
            data_dir_index += 1
        # closest_image_path_name = os.path.join("/data/weiye/dataR", "{}/{}/{:03d}.jpg".format(closest_img_info["dir1"],
        #                                                                                       closest_img_info["dir2"],
        #                                                                                       closest_img_info["id"]))
        index_closest = len(data_dict1["fn"]) + len(data_dict2["fn"]) + closest_img_info["ind"]
        # after this loop I will have the value of the minimum difference gaze label image

    # now I will create the path of the image from this data
    return index_closest, closest_img_info["gaze_diff"]


def createMatrix(data_dict1, data_dict2, data_dict3, matrix_save_path, diff_matrix_save_path):
    # matrix size is num_imgs x num_subjects
    subject_id = len(subject_ID_dict)
    num_imgs = len(data_dict1["fn"]) + len(data_dict2["fn"]) + len(data_dict3["id"])

    subject_id = len(range(subject_id)[: 15])
    num_imgs = len(range(num_imgs)[:6])
    result_matrix = np.zeros((num_imgs, subject_id))
    diff_matrix = np.zeros((num_imgs, subject_id))
    # result_matrix = np.zeros((3, 2))
    # diff_matrix = np.zeros((3, 2))
    index = 0
    for i in range(num_imgs):
        print("Currently at: " + str(i) + "/" + str(num_imgs) + " row")
        for j in range(subject_id):
            # [i][j] -> find the nearest image to [i] in the subject id of [j]
            # closest_input_path = os.path.join("/data/alex_tablet2",
            #                                   "{}/{}/face_224/{:03d}.png".format(data_dict1["dir1"][index],
            #                                                             data_dict1["dir2"][index],
            #                                                             data_dict1["fn"][index]))
            #
            # closest_image_path = findNearestForMatrix(data_dict1, data_dict2, data_dict3, i, j)
            # print("Closest image to: " + closest_input_path + "is: " + closest_image_path)
            # result_path = "/data/Mahak99/hq-data-mixup-temp/{}_{}_{}.jpg".format(data_dict1["dir1"][index],
            #                                                                      data_dict1["dir2"][index],
            #                                                                      data_dict1["fn"][index])
            # mixUpImages(0.5, closest_input_path, closest_image_path, result_path)

            img_index, gaze_diff_value = findNearestForMatrix(data_dict1, data_dict2, data_dict3, i, j)
            result_matrix[i][j] = img_index
            diff_matrix[i][j] = gaze_diff_value

            index += 1
    # save this matrix
    np.save(matrix_save_path, result_matrix)
    np.save(diff_matrix_save_path, diff_matrix)
    # print(result_file)
    return True


def findNearestForMatrixMerged(data_dict, i, compare_subject_id):
    # first find all the information related to the input image
    dataset_info_input = {}

    for each in list(data_dict.keys())[: 8]:
        dataset_info_input[each] = data_dict[each][i]

    closest_img_info = {}

    curr_min_gaze_diff = math.inf
    data_dir_index = 0

    while data_dir_index < len(data_dict["fn"]):
        if data_dict["subject_cid"][data_dir_index] == compare_subject_id and (
                data_dict["dn"][data_dir_index] != dataset_info_input["dn"] or data_dict["fn"][
            data_dir_index] != dataset_info_input["fn"] or data_dict["dataset_id"][data_dir_index] !=
                dataset_info_input["dataset_id"]):
            curr_index_gaze_value = data_dict["gaze"][data_dir_index]
            gaze_diff = angular_error(dataset_info_input["gaze"], curr_index_gaze_value)
            if gaze_diff < curr_min_gaze_diff:
                curr_min_gaze_diff = gaze_diff
                for each in list(data_dict.keys())[:8]:
                    closest_img_info[each] = data_dict[each][data_dir_index]
                closest_img_info["ind"] = data_dir_index
                closest_img_info["gaze_diff"] = curr_min_gaze_diff
        data_dir_index += 1

    # now I will create the path of the image from this data
    return closest_img_info["ind"], closest_img_info["gaze_diff"]


def createMatrixMerged(data_dict, matrix_save_path, diff_matrix_save_path):
    # matrix size is num_imgs x num_subjects
    subject_id = len(ID_subject_dict)
    num_imgs = len(data_dict["fn"])

    result_matrix = np.zeros((num_imgs, subject_id))
    diff_matrix = np.zeros((num_imgs, subject_id))
    # result_matrix = np.zeros((3, 2))
    # diff_matrix = np.zeros((3, 2))

    index = 0
    for i in range(num_imgs):
        print("Currently at: " + str(i) + "/" + str(num_imgs) + " row")
        for j in range(subject_id):
            img_index, gaze_diff_value = findNearestForMatrixMerged(data_dict, i, j)
            # result_matrix[i][j] = img_index
            diff_matrix[i][j] = gaze_diff_value

            index += 1
    # save this matrix
    np.save(matrix_save_path, result_matrix)
    np.save(diff_matrix_save_path, diff_matrix)
    return True



def mixUpImages(alpha, img1_path, img2_path, result_path_jpg):
    # load images
    src1 = cv2.imread(img1_path)
    src2 = cv2.imread(img2_path)

    # blend_images
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(src1, alpha, src2, beta, 0.0)
    # save the mixed up image
    # image_save_path = result_path_jpg + "/" + query_img_num + "+" + (img2_path.split("/")[-1])[:-4] + "_alpha: " + str(
    #     round(alpha, 3)) + ".jpg"
    # cv2.imwrite(image_save_path, dst)
    # writeOnImage(image_save_path, img2_path, "Query: " + query_img_num, "Result: " + (img2_path.split("/")[-1])[:-4],
    #              "Alpha: " + str(alpha), " Beta: " + str(beta), result_path_png, query_img_num, alpha)

    image_save_path = result_path_jpg
    cv2.imwrite(image_save_path, dst)


def writeOnImage(image_path, img2_path, text1, text2, text3, text4, result_path, query_img_num, alpha):
    from PIL import Image
    from PIL import ImageDraw
    img = Image.open(image_path)
    # Call draw Method to add 2D graphics in an image
    I1 = ImageDraw.Draw(img)

    # Add Text to an image
    I1.text((28, 120), text1, fill=(0, 255, 255))
    I1.text((28, 130), text2, fill=(0, 255, 255))
    I1.text((28, 140), text3, fill=(0, 255, 255))
    I1.text((28, 150), text4, fill=(0, 255, 255))
    # Save the edited image
    img.save(result_path + "/" + query_img_num + "+" + (img2_path.split("/")[-1])[:-4] + "_alpha: " + str(
        round(alpha, 3)) + ".png")

if __name__ == '__main__':
    # path1 = "/data/lzh/alex2_full/gt.npy"
    # path2 = "/data/lzh/alex2_out/gt.npy"
    # path3 = "/data/weiye/dataR/gt_new_filtered.npy"
    #
    # data1, data2, data3 = loadData(path1, path2, path3)
    # createMatrix(data1, data2, data3, "/data/Mahak99/hq-data-mixup-temp/matrix.npy",
    #              "/data/Mahak99/hq-data-mixup-temp/gaze_diff_matrix.npy")
    # mixUpImages(0.5, "./face_000.png", "./face_095.png", "./face_alyysa.jpg", "./result.png", 0)

    path = "/data/lzh/alex2_full/gt_v1.npy"
    data = loadMergedGT(path)

    createMatrixMerged(data, "/data/Mahak99/hq-data-mixup/matrix.npy", "/data/Mahak99/hq-data-mixup/gaze_angular_error_matrix.npy")