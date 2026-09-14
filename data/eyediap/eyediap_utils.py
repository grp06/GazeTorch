"""
Copyright (c) 2014 Idiap Research Institute, http://www.idiap.ch/
Written by Kenneth Funes <kenneth.funes@idiap.ch>

Define a few common variables/functions related to the EYEDIAP database.
"""
import numpy as np

# ------------------------  DEFINE THE LIST OF RECORDING SESSIONS -------------------------------------
sessions = []
for P in range(16):
    if P < 11:
        Cs = ['A', ]
    elif P < 13:
        Cs = ['B', ]
    else:
        Cs = ['A', 'B']
    for C in Cs:
        if (P < 11 or P > 12) and C == 'A':
            Ts = ['DS', 'CS', 'FT']
        else:
            Ts = ['FT']
        for T in Ts:
            for H in ['S', 'M']:
                sessions.append((P+1, C, T, H))

def getSessionIndex(P, C, T, H):
    try:
        idx = sessions.index((P, C, T, H))
        return idx
    except ValueError:
        return None

def get_session_string(session_idx):
    """
    Gets the string identifier for session "session_idx". It's useful for browsing the data
    """
    if session_idx >= 0 and session_idx < len(sessions):
        P, C, T, H = sessions[session_idx]
        return str(P)+'_'+C+'_'+T+'_'+H
    else:
        return ''

def get_file_txtline_values(line):
    vals = [float(el) for el in line.strip().split(';')]
    return int(vals[0]), vals[1:]


def extract_next_file_values(file_obj):
    """
    From a string row in a file creates a list of float values together with the frame index
    """
    return get_file_txtline_values(file_obj.readline())


def read_screen_track_file(screen_track_file):
    """
    Read the ground truth values, i.e. the 3D position of the screen target
    """
    screen_track_vals = np.loadtxt(screen_track_file, skiprows=1, delimiter=';')[:, -3:]
    return screen_track_vals


def read_ball_track_file(ball_track_file):
    """
    Read the ground truth values, i.e. the 3D position of the floating target
    """
    ball_track_vals = np.loadtxt(ball_track_file, skiprows=1, delimiter=';')[:, -3:]
    return ball_track_vals


def read_head_track_file(head_track_file):
    """
    Read the head pose parameters: the frame-by-frame rotation and translation
    """
    head_track_vals = np.loadtxt(head_track_file, skiprows=1, delimiter=';')[:, 1:]
    R = head_track_vals[:, :9].reshape(-1, 3, 3)
    T = head_track_vals[:, 9:12]
    return R, T


def readCalibrationFile(calibration_file):
    """
    Reads the calibration parameters
    """
    cal = {}
    fh = open(calibration_file, 'r')
    # Read the [resolution] section
    fh.readline().strip()
    cal['size'] = [int(val) for val in fh.readline().strip().split(';')]
    cal['size'] = cal['size'][0], cal['size'][1]
    # Read the [intrinsics] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(';')])
    cal['intrinsics'] = np.array(vals).reshape(3, 3)
    # Read the [R] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(';')])
    cal['R'] = np.array(vals).reshape(3, 3)
    # Read the [T] section
    fh.readline().strip()
    vals = []
    for i in range(3):
        vals.append([float(val) for val in fh.readline().strip().split(';')])
    cal['T'] = np.array(vals).reshape(3, 1)
    fh.close()
    return cal

