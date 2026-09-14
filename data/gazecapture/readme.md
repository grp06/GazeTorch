# GazeCapture dataset

From paper "Eye tracking for everyone" @ CVPR '16
https://github.com/CSAILVision/GazeCapture

The dataset comes with images + 2D gaze labels.
We integrate the codes of FAZE to get 3D gaze labels.
https://github.com/swook/faze_preprocess

### Some facts
\# of person (subject) = 1366
\# of images > 2M
\# of images of each subject varies greatly (from 18 to >3000)

### Files (2D)
Original dataset (images & 2D gaze labels & supplementary data) at `/data/gazecapture`

TO DO: Add more details ...

### Files (3D)

Processed images (face + 2 eyes, geometrically normalized and enhanced by CLAHE) at `/data/gazecapture_3d/{type}/aaaaa_bbbbb.jpg`
* `type` = `face`, `eye_L` or `eye_R`
* `aaaaa` = person ID
* `bbbbb` = image ID (base 0)

Ground truth data (3D gaze labels) of person `aaaaa` at `/data/gazecapture_3d/gt/aaaaa.npy`.
Load it using numpy: `D = np.load("xxx.npy", allow_pickle=True).item()`
The `npy` contains a dictionary. Each key corresponds to a `list`.
* `fn`: Image file name `aaaaa_bbbbb.jpg`. Note that `fn[i] != i` - some images are removed so IDs are not continuous.
* `gaze`: Gaze (face) - each item is an [2] ndarray (pitch, yaw). Same below.
* `gaze_L`: Gaze (left eye)
* `gaze_R`: Gaze (right eye)
* `head_pose`

`/data/gazecapture_3d/gt_raw.npy` is a concatenation of all per-person GT npy files.
`/data/gazecapture_3d/population.npy` is a dictionary {key `aaaaa`: \# of images of subject `aaaaa`}.

TO DO: make GT files with both 2D and 3D labels.

### Pitch/Yaw Direction
Where positive pitch & yaw axes point to is always an issue. One must carefully investigate this before using any dataset.
For GazeCapture_3D particularly,
* Positive pitch = ↑ (same as XGaze)
* Positive yaw = ← (same as XGaze)