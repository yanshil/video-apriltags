import cv2
from cv2 import imshow
from scipy.spatial.transform import Rotation
import os
import re
import numpy as np
import glob
import matplotlib.pyplot as plt
import pandas as pd

from dt_apriltags import Detector
from utils import *

def transform(x, y, theta):
    X = x * np.cos(theta) + y * np.sin(theta)
    Y = - x * np.sin(theta) + y * np.cos(theta)
    return X, Y


def getAprilTagsInfo(images_folder, camera_matrix, tag_size, debug_dir=None):
    '''
    Return a dictionary

    tag_info_dict = 
    {
        'frame0.jpg': tags ## list of tags 
        'frame1.jpg': tags ## list of tags
        ...
    }

    and

    msg = Sucess Detect: 302/678
    '''
    at_detector = Detector(searchpath=['/usr/local/lib'],
                           families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)

    success_dectection = 0

    # Get images from folder
    files = sorted(glob.glob(os.path.join(images_folder, '*.jpg')))
    file_num = len(files)

    # Center: k*2, Pose_R: 3*3*k, Pose_t: 3*1*k
    tag_info_dict = {}
    for image in files:
        img_name = os.path.basename(image)
        # Initialize as empty list
        tag_info_dict[img_name] = []

        img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        camera_params = (
            camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2])
        tags = at_detector.detect(
            img, estimate_tag_pose=True, camera_params=camera_params, tag_size=tag_size)
        tag_info_dict[img_name] = tags

        if len(tags) > 0:
            success_dectection += 1

        if debug_dir is not None:
            color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for tag in tags:
                for idx in range(len(tag.corners)):
                    cv2.line(color_img, tuple(
                        tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

                cv2.putText(color_img, str(tag.tag_id),
                            org=(tag.corners[0, 0].astype(
                                int)+10, tag.corners[0, 1].astype(int)+10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.8,
                            color=(0, 0, 255))

            saveFilename = os.path.join(debug_dir, 'debug' + img_name)
            cv2.imwrite(saveFilename, color_img)

    msg = "Success Detect: " + str(success_dectection) + ' / ' + str(file_num)
    return [tag_info_dict, msg]


def fixAngleBreakpoints(dat):
    """
    Fix angles breakpoints that suddenly jump from -pi to pi
    """
    threshold = np.pi  # In fact should close to 360
    q = dat[['angle']].to_numpy()
    q = np.squeeze(q)
    diff_angle = np.diff(q, axis=0)
    break_points = list(np.where(abs(diff_angle) > threshold)[0])
    print("Breakpoints: ", break_points)
    # print(dat[break_points[0]: break_points[0]+3])
    for bp_idx in break_points:
        plus = q[bp_idx+1] + np.sign(q[bp_idx]) * 2 * np.pi
        minus = q[bp_idx+1] - np.sign(q[bp_idx]) * 2 * np.pi

        diff_plus = abs(q[bp_idx] - plus)
        diff_minus = abs(q[bp_idx] - minus)
        if diff_plus <= diff_minus:
            q[bp_idx+1:] += np.sign(q[bp_idx]) * 2 * np.pi
        else:
            q[bp_idx+1:] -= np.sign(q[bp_idx]) * 2 * np.pi
    dat[['angle']] = pd.DataFrame(q, index=dat.index, columns=['angle'])
    # print(dat[break_points[0]: break_points[0]+3])
    # Check again:
    validate = np.diff(q, axis=0)
    validate_bp = list(np.where(abs(validate) > threshold)[0])
    assert validate_bp == list(), "Assert no breakpoints exists after fixing."
    return dat

def applyTransformation(df):
    """
    Set frame 1 as reference frame (x, y, theta) = (0, 0, 0)
    """
    x0 = df.iloc[0, ]['x']
    y0 = df.iloc[0, ]['y']
    r0 = df.iloc[0, ]['angle']

    for i in range(df.shape[0]):
        x = df.iloc[i, ]['x']
        y = df.iloc[i, ]['y']
        angle = df.iloc[i, ]['angle']
        new_x, new_y = transform(x - x0, - (y - y0), -r0)
        df.iloc[i, ]['x'] = new_x
        df.iloc[i, ]['y'] = new_y
        df.iloc[i, ]['angle'] = - (angle - r0)
    return df

def visualize_df(df, figpath):
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    angle = df['angle'].to_numpy()

    plt.cla()
    for i in range(x.shape[0]):
        plt.plot(x[i], y[i], marker=(3, 0, angle[i]),
                    markersize=10, linestyle='None')
    # plt.plot(x[0], y[0], marker=(3, 0, angle[0]), markersize=20, linestyle='None')
    plt.axis('equal')
    plt.text(x[0], y[0], 'Starting Point', rotation=angle[0] * 180 / np.pi)
    if figpath is None:
        plt.show()
    else:
        plt.savefig(figpath)

def dump2txt(tag_info_dict, filename, figpath=None):
    df = pd.DataFrame(index=list(tag_info_dict.keys()), columns=[
                      'x', 'y', 'angle'])
    for f, tags in tag_info_dict.items():
        if len(tags) > 0:
            tag = tags[0]
            rz = Rotation.from_matrix(tag.pose_R).as_euler(
                'zyx', degrees=False)[0]
            tag_x = float(tag.pose_t[0])
            tag_y = float(tag.pose_t[1])
            df['x'][f] = tag_x
            df['y'][f] = tag_y
            df['angle'][f] = rz

    df = fixAngleBreakpoints(df)
    df = applyTransformation(df)
    # print(df)
    df['angle'] = df['angle'] * 180 / np.pi
    df.to_csv(filename, header=False, index=True, sep=' ', mode='w')

    ## Visualize trajectory
    visualize_df(df, figpath)
