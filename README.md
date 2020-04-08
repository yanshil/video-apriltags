# video-apriltags

## Requirements

1. duckietown-Apriltags bindings  https://github.com/duckietown/dt-apriltags

> I don't remember how I solved the  OSError.... I might did something like sudo make install after building this library...
> 
> OSError: /home/yanshimsi/.local/lib/python3.8/site-packages/dt_apriltags/libapriltag.so: cannot open shared object file: No such file or directory
> 
> My libapriltag.so is located in /usr/local/lib/

2. Others:  `pip install opencv-python numpy pandas scipy matplotlib`  

## Get Video Tags Trajectory Pipeline
1. Dump video to frames `vedio2frames()`
2. Detect tags in frames `getAprilTagsInfo()`. Modified from https://github.com/duckietown/dt-apriltags/blob/master/test/test.py
3. Tidy the tag info (x, y, pose_t, etc.)
4. Apply fix and transformations if needed