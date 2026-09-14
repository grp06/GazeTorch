'''
by Mahak.
'''

def compareImgIOU(img1, img2):
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
        # # for IRIS map IOU value
        # img1_left_eye = img1[:, :, 1]
        # img1_left_eye[img1_left_eye == 2] = 0
        # img1_right_eye = img1[:, :, 3]
        # img1_right_eye[img1_right_eye == 2] = 0

        # img2_left_eye = img2[:, :, 1]
        # img2_left_eye[img2_left_eye == 2] = 0
        # img2_right_eye = img2[:, :, 3]
        # img2_right_eye[img2_right_eye == 2] = 0

        # # run the IOU function
        # IOU_value = calculateIOU(img1_left_eye, img1_right_eye, img2_left_eye, img2_right_eye)

def calculateIOU(img1_left_eye, img1_right_eye, img2_left_eye, img2_right_eye):
    d1 = sum(sum(img1_left_eye == 1) + sum(img1_right_eye == 1))
    d2 = sum(sum(img2_left_eye == 1) + sum(img2_right_eye == 1))

    n1_img = img1_left_eye + img2_left_eye
    n2_img = img1_right_eye + img2_right_eye

    n = sum(sum(n1_img == 2) + sum(n2_img == 2))
    d = d1 + d2 - n

    return n / d