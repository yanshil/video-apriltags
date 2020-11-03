# video-apriltags

## Camera Calibration
Modify from https://github.com/opencv/opencv/blob/master/samples/python/calibrate.py

```
python CameraCalibration.py
```




## Apriltags Detection

### Requirements

1. duckietown-Apriltags bindings  https://github.com/duckietown/dt-apriltags
   1. Actually doing `pip install dt_apriltags` is enough
2. Others:  `pip install opencv-python numpy pandas scipy matplotlib`  

### Get Video Tags Trajectory Pipeline
1. Dump video to frames `vedio2frames()`
2. Detect tags in frames `getAprilTagsInfo()`. Modified from https://github.com/duckietown/dt-apriltags/blob/master/test/test.py
3. Tidy the tag info (x, y, pose_t, etc.)
4. Apply fix and transformations if needed

### OSError? 

If met with OSError as the following during funning, add a parameter "searchpath" in `Detector()` like this

> OSError: /home/yanshimsi/.local/lib/python3.8/site-packages/dt_apriltags/libapriltag.so: cannot open shared object file: No such file or directory

Try to find where your libapriltag.so is located first (Mine is in `/usr/local/lib` after doing `sudo make install`), and then use the Detector as

```
from dt_apriltags import Detector
at_detector = Detector(searchpath=['/usr/local/lib'],
                        families='tag36h11',
                        nthreads=1,
                        quad_decimate=1.0,
                        quad_sigma=0.0,
                        refine_edges=1,
                        decode_sharpening=0.25,
                        debug=0)
```

