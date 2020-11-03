#!/usr/bin/env python

## python CameraCalibration.py

'''
camera calibration for distorted images with chess board samples
reads distorted images, calculates the calibration and write undistorted images

usage:
    CameraCalibration.py

parameters values:
    --filepath:
    --dir_name:
    --debug_dir: 
    --square_size: 
'''


import numpy as np
import cv2 as cv
from random import sample, seed
# built-in modules
import os

from utils import *

# local modules
# from common import splitfn
def splitfn(fn):
    path, fn = os.path.split(fn)
    name, ext = os.path.splitext(fn)
    return path, name, ext


def run(img_names, square_size, pattern_size, debug_dir = None, threads_num = 4, undistort = False):
    createFolderIfNotExist(debug_dir)
    pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
    pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
    pattern_points *= square_size

    obj_points = []
    img_points = []
    h, w = cv.imread(img_names[0], cv.IMREAD_GRAYSCALE).shape[:2]  # TODO: use imquery call to retrieve results
    print("height: " + str(h))
    print("width: " + str(w))

    def processImage(fn):
        print('processing %s... ' % fn)
        img = cv.imread(fn, 0)
        if img is None:
            print("Failed to load", fn)
            return None

        assert w == img.shape[1] and h == img.shape[0], ("size: %d x %d ... " % (img.shape[1], img.shape[0]))
        found, corners = cv.findChessboardCorners(img, pattern_size)
        if found:
            term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)
            cv.cornerSubPix(img, corners, (5, 5), (-1, -1), term)

        if debug_dir:
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            cv.drawChessboardCorners(vis, pattern_size, corners, found)
            _path, name, _ext = splitfn(fn)
            outfile = os.path.join(debug_dir, name + '_chess.png')
            cv.imwrite(outfile, vis)

        if not found:
            print('chessboard not found')
            return None

        print('           %s... OK' % fn)
        return (corners.reshape(-1, 2), pattern_points)

    if threads_num <= 1:
        chessboards = [processImage(fn) for fn in img_names]
    else:
        print("Run with %d threads..." % threads_num)
        from multiprocessing.dummy import Pool as ThreadPool
        pool = ThreadPool(threads_num)
        chessboards = pool.map(processImage, img_names)

    chessboards = [x for x in chessboards if x is not None]
    for (corners, pattern_points) in chessboards:
        img_points.append(corners)
        obj_points.append(pattern_points)

    # calculate camera distortion
    rms, camera_matrix, dist_coefs, _rvecs, _tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)

    print("\nRMS:", rms)
    print("camera matrix:\n", camera_matrix)
    print("distortion coefficients: ", dist_coefs.ravel())
    print("Reshape as K: \n", np.reshape(camera_matrix, (1,9)).tolist())
    
    if undistort:
        # undistort the image with the calibration
        print('')
        for fn in img_names if debug_dir else []:
            _path, name, _ext = splitfn(fn)
            img_found = os.path.join(debug_dir, name + '_chess.png')
            outfile = os.path.join(debug_dir, name + '_undistorted.png')

            img = cv.imread(img_found)
            if img is None:
                continue

            h, w = img.shape[:2]
            newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

            dst = cv.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

            # crop and save the image
            x, y, w, h = roi
            dst = dst[y:y+h, x:x+w]

            #print('Undistorted image written to: %s' % outfile)
            cv.imwrite(outfile, dst)

        print('Done')


if __name__ == '__main__':
    GUI = True
    square_size = 0.02          ## Size of blocks in a chessboard
    pattern_size = (9,7)        ## Num of cross: (Row-1) * (Col-1) of the chessboad

    ## Video Example: Box\Camera_Calibration\Camera Calibration\calibration_vedio\90fov.mp4
    ## Checkboard Example: Box\Camera_Calibration\Camera Calibration\data\camera-calibration-checker-board_9x7.pdf
    if GUI:
        filepath = select_file(title="Select the Checkboard Video")
        dir_name = select_directory(title="Select a Directory to Dump Frames")
    else: ## Directly assign path
        filepath = "data/90fov.mp4"
        dir_name = "output"

    vedio2frames(filepath, dir_name)
    img_names = samplingFrames(dir_name, random_num=20)
    run(img_names, square_size, pattern_size = pattern_size, debug_dir = './debug/', threads_num = 4, undistort = True)
    cv.destroyAllWindows()