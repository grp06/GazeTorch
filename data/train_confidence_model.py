import cv2
import random
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgaze.stack_face_facemap import stack_face_facemap
import time
import joblib





df = pd.read_excel('iris_annot.xlsx')

def extract_feature(face, facemap, headpose):
    leye = facemap[:, :, 0]
    liris = facemap[:, :, 1]
    reye = facemap[:, :, 2]
    riris = facemap[:, :, 3]
    pitch = headpose[0] * 180 / np.pi
    yaw = headpose[1] * 180 / np.pi
    leye[leye != 0] = 255
    reye[reye != 0] = 255
    liris[liris != 0] = 255
    riris[riris != 0] = 255
    leye_pix = np.count_nonzero(leye == 255)
    reye_pix = np.count_nonzero(reye == 255)
    liris_pix = np.count_nonzero(liris == 255)
    riris_pix = np.count_nonzero(riris == 255)
    # contours_leye, _ = cv2.findContours(leye, 1, 2)
    # contours_liris, _ = cv2.findContours(liris, 1, 2)
    # contours_reye, _ = cv2.findContours(reye, 1, 2)
    # contours_riris, _ = cv2.findContours(riris, 1, 2)
    # positions = []
    # for k, cnt in enumerate([contours_leye, contours_liris, contours_reye, contours_riris]):
    #     if len(cnt) == 0:
    #         cx = 0
    #         cy = 0
    #     else:
    #         cnt = sorted(cnt, key=lambda x: cv2.contourArea(x), reverse=True)
    #         M = cv2.moments(cnt[0])
    #         if np.abs(M['m00']) <= 1e-4:
    #             cx = 0 
    #             cy = 0
    #         else:
    #             cx = int(M['m10']/M['m00'])
    #             cy = int(M['m01']/M['m00'])

        # positions.extend([cx, cy])

    mean_illum = np.mean(face)
    feature = [leye_pix, liris_pix, reye_pix, riris_pix] + [mean_illum] + [pitch, yaw]

    return feature

def create_dataset():
    gt_data = np.load('/data/xgaze/448/gt_p17y17b16.npy', allow_pickle=True).item()
    X = []
    y = []
    all_headposes = gt_data['headpose']
    for index, row in df.iterrows():
        face_dir = '/data/xgaze/448/face_224/{}.jpg'.format(row['fn'])
        facemap_dir = face_dir.replace('.jpg', '.png').replace('face_224', 'facemap_224')
        face = cv2.imread(face_dir)
        facemap = cv2.imread(facemap_dir, cv2.IMREAD_UNCHANGED)        
        headpose = all_headposes[row['fn']]
        feature = extract_feature(face, facemap, headpose)
        outputs = [row['left eye'] / 3., row['left Iris'] / 3., row['right eye'] / 3., row['right iris'] / 3.]
        X.append(feature)
        y.append(outputs)
        print('index: {}'.format(index), end='\r')

    X = np.array(X)
    y = np.array(y)

    return X, y

def visualize_confidence_xgaze(reg, scaler):
    gt_data = np.load('/data/xgaze/448/gt_p17y17b16.npy', allow_pickle=True).item()
    plt.figure(figsize=(24, 12))

    face_dirs = glob('/data/xgaze/448/face_224/*.jpg')
    dict_data = {k:[] for k in range(18)}
    cam_id = 0 
    while cam_id < 18:
        print('------------------------- CAMID: {} -------------------------'.format(cam_id))
        cam_id_face_dirs = list(filter(lambda x: (int(x.split('/')[-1].split('.')[0]) % 18) == cam_id, face_dirs))
        rand_face_dir = random.choice(cam_id_face_dirs)
        rand_facemap_dir = rand_face_dir.replace('.jpg', '.png').replace('face_224', 'facemap_224')
        try:
            face = cv2.imread(rand_face_dir)
            facemap = cv2.imread(rand_facemap_dir, cv2.IMREAD_UNCHANGED)
        except:
            continue
        if facemap is None:
            continue
        fn = int(rand_face_dir.split('/')[-1].split('.')[0])
        headpose = gt_data['headpose'][fn]
        feature = extract_feature(face, facemap, headpose)
        feature = np.array(feature).reshape([1, -1])
        feature_n = scaler.transform(feature)
        confs = reg.predict(feature_n)[0]

        confs[confs < 0.25] = 0
        confs[confs >= 0.9] = 1
        confs[(np.where((confs >= 0.25) & (confs <= 0.9)))] = 1.54 * (confs[(np.where((confs >= 0.25) & (confs <= 0.9)))] - 0.25)
        confs[confs >= 1] = 1
        confs[confs <= 0] = 0

        res = stack_face_facemap(face, facemap)

        leye_conf = round(confs[0], 1)
        liris_conf = round(confs[1], 1)
        reye_conf = round(confs[2], 1)
        riris_conf = round(confs[3], 1)   
        plt.subplot(3, 6, cam_id+1)
        plt.imshow(res[:, :, ::-1] / 255.)
        plt.title('wLE:{},wLI:{},wRE:{},wRI:{}'.format(leye_conf, liris_conf, reye_conf, riris_conf))
        plt.axis('off')
        cam_id += 1

                    
    plt.savefig('confidence_4.png')
    plt.close()


def visualize_confidence_gc3d(reg, scaler):
    gt_data = np.load('/data/gazecapture_3d/gt_p17y17.npy', allow_pickle=True).item()
    plt.figure(figsize=(24, 12))
    face_dirs = glob('/data/gazecapture_3d/face_112/*.jpg')
    dict_data = {k:[] for k in range(18)}
    n_saved = 0
    for i in range(100):
        if n_saved == 18:
            break
        rand_face_dir = random.choice(face_dirs)
        rand_facemap_dir = rand_face_dir.replace('.jpg', '.png').replace('face_112', 'facemap')
        face = cv2.imread(rand_face_dir)
        face = cv2.resize(face, (224, 224))
        try:
            facemap = cv2.imread(rand_facemap_dir, cv2.IMREAD_UNCHANGED)
        except:
            continue
        if facemap is None:
            continue

        fn = rand_face_dir.split('/')[-1].split('.')[0]
        idx = gt_data['fn'].index(fn)
        headpose = gt_data['head_pose'][idx]
        feature = extract_feature(face, facemap, headpose)
        feature = np.array(feature).reshape([1, -1])
        feature_n = scaler.transform(feature)
        confs = reg.predict(feature_n)[0]
        
        confs[confs < 0.25] = 0
        confs[confs >= 0.9] = 1
        confs[(np.where((confs >= 0.25) & (confs <= 0.9)))] = 1.54 * (confs[(np.where((confs >= 0.25) & (confs <= 0.9)))] - 0.25)
        confs[confs >= 1] = 1
        confs[confs <= 0] = 0

        res = stack_face_facemap(face, facemap)

        leye_conf = round(confs[0], 1)
        liris_conf = round(confs[1], 1)
        reye_conf = round(confs[2], 1)
        riris_conf = round(confs[3], 1)   
        plt.subplot(3, 6, n_saved+1)
        plt.imshow(res[:, :, ::-1] / 255.)
        plt.title('wLE:{},wLI:{},wRE:{},wRI:{}'.format(leye_conf, liris_conf, reye_conf, riris_conf))
        plt.axis('off')
        print(n_saved)
        n_saved += 1
        print('------------------------------------')
        
                    
    plt.savefig('confidence_gazecapture.png')
    plt.close()





def plot_feature_importance(X_scaled, y):
    from sklearn.ensemble import RandomForestClassifier
    y_cls = (y * 3.).astype('int32')
    #feature_names = ["#LEye", "#Liris", "#REye", "RIris", "CenterX_LEye", "CenterY_LEye", "CenterX_LIris", "CenterY_LIris", "CenterX_REye", "CenterY_REye", "CenterX_RIris", "CenterY_RIris", "Mean Brightness", "Head Pitch", "Head Yaw"]
    feature_names = ["#LEye", "#Liris", "#REye", "RIris", "Mean Brightness", "Head Pitch", "Head Yaw"]
    
    forest = RandomForestClassifier(random_state=0)
    forest.fit(X_scaled, y_cls)
    
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)


    forest_importances = pd.Series(importances, index=feature_names)
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig('importance.png')
    plt.close()


def visualize_confidence_xgaze_from_npy():
    gt_data = np.load('/data/xgaze/448/gt_p17y17b16_with_confs_v2.npy', allow_pickle=True).item()
    plt.figure(figsize=(24, 12))


    face_dirs = glob('/data/xgaze/448/face_224/*.jpg')
    dict_data = {k:[] for k in range(18)}
    cam_id = 0 
    while cam_id < 18:
        print('------------------------- CAMID: {} -------------------------'.format(cam_id))
        cam_id_face_dirs = list(filter(lambda x: (int(x.split('/')[-1].split('.')[0]) % 18) == cam_id, face_dirs))
        rand_face_dir = random.choice(cam_id_face_dirs)
        rand_facemap_dir = rand_face_dir.replace('.jpg', '.png').replace('face_224', 'facemap_224')
        try:
            face = cv2.imread(rand_face_dir)
            facemap = cv2.imread(rand_facemap_dir, cv2.IMREAD_UNCHANGED)
        except:
            continue
        if facemap is None:
            continue
        fn = int(rand_face_dir.split('/')[-1].split('.')[0])
        confs = [gt_data['wL_E_v2'][fn], gt_data['wL_I_v2'][fn], gt_data['wR_E_v2'][fn], gt_data['wR_I_v2'][fn]]
        
        res = stack_face_facemap(face, facemap)
        leye_conf = round(confs[0], 1)
        liris_conf = round(confs[1], 1)
        reye_conf = round(confs[2], 1)
        riris_conf = round(confs[3], 1)   
        plt.subplot(3, 6, cam_id+1)
        plt.imshow(res[:, :, ::-1] / 255.)
        plt.title('wLE:{},wLI:{},wRE:{},wRI:{}'.format(leye_conf, liris_conf, reye_conf, riris_conf))
        plt.axis('off')
        cam_id += 1

                    
    plt.savefig('confidence_data.png')
    plt.close()


def visualize_confidence_gc3d_from_npy():
    gt_data = np.load('/data/gazecapture_3d/gt_p17y17_with_conf_v2.npy', allow_pickle=True).item()
    plt.figure(figsize=(24, 12))
    face_dirs = glob('/data/gazecapture_3d/face_112/*.jpg')
    dict_data = {k:[] for k in range(18)}
    n_saved = 0
    for i in range(100):
        if n_saved == 18:
            break
        rand_face_dir = random.choice(face_dirs)
        rand_facemap_dir = rand_face_dir.replace('.jpg', '.png').replace('face_112', 'facemap')
        face = cv2.imread(rand_face_dir)
        face = cv2.resize(face, (224, 224))
        try:
            facemap = cv2.imread(rand_facemap_dir, cv2.IMREAD_UNCHANGED)
        except:
            continue
        if facemap is None:
            continue

        fn = rand_face_dir.split('/')[-1].split('.')[0]
        idx = gt_data['fn'].index(fn)
        confs = [gt_data['wL_E_v2'][idx], gt_data['wL_I_v2'][idx], gt_data['wR_E_v2'][idx], gt_data['wR_I_v2'][idx]]

        res = stack_face_facemap(face, facemap)

        leye_conf = round(confs[0], 1)
        liris_conf = round(confs[1], 1)
        reye_conf = round(confs[2], 1)
        riris_conf = round(confs[3], 1)   
        plt.subplot(3, 6, n_saved+1)
        plt.imshow(res[:, :, ::-1] / 255.)
        plt.title('wLE:{},wLI:{},wRE:{},wRI:{}'.format(leye_conf, liris_conf, reye_conf, riris_conf))
        plt.axis('off')
        print(n_saved)
        n_saved += 1
        print('------------------------------------')
        
                    
    plt.savefig('confidence_gazecapture_npy.png')
    plt.close()


if __name__ == "__main__":
    X, y = create_dataset()
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    plot_feature_importance(X_scaled, y)
    reg = RandomForestRegressor(random_state=0, n_estimators=100, max_depth=10).fit(X_scaled, y)
    
    cross_val_mae = np.mean(-cross_val_score(RandomForestRegressor(random_state=0, n_estimators=100, max_depth=10), X_scaled, y, cv=5, scoring='neg_mean_absolute_error'))
    joblib.dump(reg, 'confidence_regressor.sav')
    joblib.dump(scaler, 'confidence_scaler.sav')
    # visualize_confidence_gc3d(reg, scaler)
    # visualize_confidence_xgaze(reg, scaler)
    #visualize_confidence_xgaze_from_npy()
    #visualize_confidence_gc3d_from_npy()
    
    # param_grid = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10], 'n_estimators': [10, 50, 100, 200, 300]}
    # dict_scores = dict()
    # for n_estimators in [10, 50, 100, 200]:
    #     for max_depth in [3, 5, 10]:
    #         reg = RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth)
    #         cross_val_mae = np.mean(-cross_val_score(RandomForestRegressor(random_state=0, n_estimators=n_estimators, max_depth=max_depth), X_scaled, y, cv=5, scoring='neg_mean_absolute_error'))
    #         dict_scores[(n_estimators, max_depth)] = cross_val_mae
    #         print(n_estimators, max_depth, cross_val_mae)
    #         print('----------------------------------------------')
    # print(dict_scores)
    # asd