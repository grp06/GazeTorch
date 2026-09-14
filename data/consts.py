AUG_DIR_MATERIALS = 'data/augmenter'
AUG_DEFAULT_ARGS_PATH_TRAIN = "config/augmenter/default_train.yaml"
AUG_DEFAULT_ARGS_PATH_TEST = "config/augmenter/default_test.yaml"

IMG_TYPES = ('face', 'eye_L', 'eye_R', 'eyemap_L', 'eyemap_R', 'irismap_L', 'irismap_R', 'facemap', 'facemask', 'facemap_v3')
RGB_TYPES = ('face', 'eye_L', 'eye_R')

ANGULAR_TYPES = ('gaze', 'gaze_L', 'gaze_R', 'headpose')

EYE_L_TYPES = ('eye_L', 'eyemap_L', 'irismap_L')
EYE_R_TYPES = ('eye_R', 'eyemap_R', 'irismap_R')
FACE_TYPES = ('face', 'facemap', 'facemask', 'facemap_v3')
PNG_TYPES = ('facemap', 'facemask', 'facemap_v3')

SINGLE_EYE_TYPES = ('eye', 'eyemap', 'irismap', 'gaze*', 'w', 'wE', 'wI')
CONST_TYPES = ('fn', 'id', 'subject_cid', 'cam_cid')
ALPHA = 'alpha'

ID_FACE = 'face'
ID_LEFT = 'left'
ID_RIGHT = 'right'
CATEGORIES = (ID_FACE, ID_LEFT, ID_RIGHT)

def get_category(name:str):
    if name.startswith('face'): return ID_FACE
    if name.endswith('L'): return ID_LEFT
    if name.endswith('R'): return ID_RIGHT
    return None

# glasses_simulator (GS)
GS_SL_L = slice(0,16)
GS_SL_R = slice(16,32)
GS_SL_LT = slice(1,8)
GS_SL_LB = slice(9,16)
GS_SL_RT = slice(17,24)
GS_SL_RB = slice(25,32)
GS_SL_ALL = slice(0,32)
GS_ID_CENTROID_L = 32
GS_ID_CENTROID_R = 33