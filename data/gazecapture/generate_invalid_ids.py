# by Leo.
import numpy as np

def loadMergedGT(path):
    ## load the dataset
    data = np.load(path, allow_pickle=True).item()
    return data

def generate_invalid_ids(data, sorted_subject_cids):
    invalid_id_index = [3, 49, 53, 58, 60, 73 , 75, 82, 89,135,189,203,222,228,247,255,257,263,269,271,300,306,311, 413, 417,433,434,463,526,538,
                     540,553,580,584,604,619,634,654,672,680,681,687,699,705,753,755,794,810,819,831,855,864,869,873,876,877,878,880,892,938,941,986,987,1002,
                     1012,1016,1034,1036,1041,1042,1048,1049,1052,1065,1087,1112,1113,1114,1151,1156,1206,1238,1320,1349,1358]
    invalid_subject_cids = [sorted_subject_cids[i] for i in invalid_id_index]


    y_hp_vector = np.bitwise_and(np.array(data["headpose"])[:, 0]>-np.pi/6, np.array(data["headpose"])[:, 0]<np.pi/6)
    x_hp_vector = np.bitwise_and(np.array(data["headpose"])[:, 1]>-np.pi/9, np.array(data["headpose"])[:, 1]<np.pi/9)
    hp_vector = np.bitwise_and(y_hp_vector, x_hp_vector)

    
    invalid_index = []
    invalid_index_with_hp_filter = []
    n = len(data['headpose'])
    for i in range(n):
        if data['subject_cid'][i] in invalid_subject_cids:
            invalid_index.append(i)
            invalid_index_with_hp_filter.append(i)
        if not hp_vector[i]:
            invalid_index_with_hp_filter.append(i)

    invalid_index = list(set(invalid_index))     
    invalid_index_with_hp_filter = list(set(invalid_index_with_hp_filter))
    np.save("invalid_ids_v2.npy", invalid_index)
    np.save("invalid_ids_v3.npy", invalid_index_with_hp_filter)
    print("done")
        
    return

if __name__ == '__main__':
    PATH_GT_DEFAULT = '/data/gazecapture_3d/gt_p17y17_with_conf_v3.npy'
    PATH_POPULATION = '/data/leo/GazeTorch/sorted_population_subject_cids.npy'

    # sorted_subject_cids = np.load(PATH_POPULATION)
    # invalid_id_v1 = np.load("/data/gazecapture_3d/mixup_invalid_ids.npy")
    # data = loadMergedGT(PATH_GT_DEFAULT)

    # generate_invalid_ids(data, sorted_subject_cids)

    v2 = np.load("invalid_ids_v2.npy")
    v3 = np.load("invalid_ids_v3.npy")

    inv_id = np.load("/data/gazecapture_3d/mixup_invalid_ids.npy")
    print("inv_id: ", len(inv_id))
    print("v2: ", len(v2))
    print("v3: ", len(v3))
    v2 = np.concatenate((v2, inv_id))
    print("v2: ", len(v2))
    v3 = np.concatenate((v3, inv_id))
    print("v3: ", len(v3))

    v2 = list(set(v2)) 
    v3 = list(set(v3)) 
    print("v2 after remove redundant: ", len(v2))
    print("v3 after remove redundant: ", len(v3))

    np.save("invalid_ids_v2.npy", v2)
    np.save("invalid_ids_v3.npy", v3)
