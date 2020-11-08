import os
import numpy as np
import glob
import pickle
from vedio2traject import *


def getConfig():
    config = {
        'dumpFrames2tmp': False,            ## If need video2frames
        'newDetect': True,                  ## frames2taginfo
        'loadInfoDictFromPickle': False,    ## dump frames2taginfo
        'dumpInfoDictPickle': True,         ## load frames2taginfo
        'dumpTxt': True,                    ## dump (x, y, theta)
        'dumpVisualization': True           ## Draw the trajectory
    }
    return config


if __name__ == "__main__":
    config = getConfig()
    ## 90 FOV
    vedio_CM = np.array([732.9546741360766, 0.0, 626.491008574248, 0.0, 728.6085953040948, 366.17003214171933, 0.0, 0.0, 1.0]).reshape((3, 3))

    tag_size = 0.07
    tmp_outdir = './tmp/'
    input_dir = './data/'
    output_dir = './outputs/'

    createFolderIfNotExist(tmp_outdir)
    createFolderIfNotExist(output_dir)

    ## Grab all in input folder
    vedio_list = glob.glob(input_dir + 'circle_02.mov')
    vedio_list = sorted(vedio_list)

    ## Grab latest video in input folder
    #latest_file = max(list_of_files, key=os.path.getctime)
    #vedio_list = [latest_file]

    ## Grab specify video in input folder
    # vedio_list = ['motion_01_1.mov']
    # vedio_list = [os.path.join(input_dir, x) for x in vedio_list]

    print(vedio_list)

    for v in vedio_list:
        if not os.path.isfile(v):
            print("Check file {} locations. This file does not exists".format(v))
            continue
        v_basename = os.path.basename(v)
        v_basename = v_basename.replace('.mp4', '')
        print("Working on " + v_basename + "......")

        outdir = os.path.join(
            tmp_outdir, get_valid_filename(v_basename))

        if config['dumpFrames2tmp']:
            vedio2frames(v, outdir)
        if config['newDetect']:
            [tag_info_dict, msg] = getAprilTagsInfo(outdir,
                                                                  vedio_CM, tag_size)
            print(msg)

        if config['dumpInfoDictPickle']:
            with open(output_dir + v_basename+'_dict_info.pkl', 'wb') as f:
                pickle.dump(tag_info_dict, f)
        if config['loadInfoDictFromPickle']:
            with open(output_dir + v_basename + '_dict_info.pkl', 'rb') as f:
                tag_info_dict = pickle.load(f)

        if config['dumpVisualization']:
            figpath = output_dir + v_basename + '.png'
        else:
            figpath = None

        if config['dumpTxt']:
            dump2txt(
                tag_info_dict, output_dir + v_basename+'.txt', figpath)
