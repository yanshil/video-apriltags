import re
import os
import cv2

curr_directory = os.getcwd()

def select_file(title = "Select file"):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(initialdir = curr_directory, title=title)
    if file_path == tuple():
        exit(0)
    if file_path == "":
        exit(0)

    return file_path

def select_directory(title = "Select save directory"):
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()

    dir = filedialog.askdirectory(initialdir = curr_directory, title=title)
    if dir == tuple():
        exit(0)
    if dir == "":
        exit(0)

    return dir

def createFolderIfNotExist(dir):
    if dir and not os.path.isdir(dir):
        os.mkdir(dir)
    else: ## Already dumped?
        return 

def vedio2frames(vedioName, out_dir):
    print("Dumping to frames...")
    createFolderIfNotExist(out_dir)
    
    vidcap = cv2.VideoCapture(vedioName)
    success, image = vidcap.read()
    count = 0
    # success = True
    if not success:
        print("Fail to read video")
    while success:
        outfile = os.path.join(out_dir, "frame{:0>4d}.jpg".format(count))
        cv2.imwrite(outfile, image)
        success,image = vidcap.read()
        #print 'Read a new frame: ', success
        count += 1

def samplingFrames(out_dir, random_num = 50):
    from random import sample, seed
    from glob import glob
    seed(0)
    img_names = glob(os.path.join(out_dir, "*.jpg"))
    if random_num > len(img_names):
        return img_names
    img_names = sample(img_names, random_num)
    return img_names


def get_valid_filename(s):
    """
    django
    """
    s = str(s).strip().replace(' ', '_')
    return re.sub(r'(?u)[^-\w.]', '', s)