import numpy as np

BLACKLIST_IDS = [3, 49, 53, 58, 60, 73, 75, 82]
PATH_POP_SORTED = '/data/leo/GazeTorch/sorted_population_subject_cids.npy'

subject_ids_sorted = np.load(PATH_POP_SORTED)
D = {}
rank = 0
for i in range(len(subject_ids_sorted)):
    if i in BLACKLIST_IDS: continue
    D[subject_ids_sorted[i]] = rank
    rank += 1
    if rank == 80:
        break

np.save("data/gazecapture/mixup_subject_id_rank.npy", D)
print("Voila.")