# preprocess script
# pseudo code
# - define tuples
# - find files (videos)                         -done
# - split in images                             -done
# - run background substraction to create map
# - save images or create "ground truth" video  
# - compute the class weigths
# - save meta informations

import os
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2

# own files
from utils import *

from os import listdir
from os.path import isfile, join
import pathlib

# for i in tqdm(range(10)):
#     time.sleep(1)


# SETUP
data_root = os.path.join("recordings","old") # os.getcwd()

do_create_meta = True
do_create_frames = False

file_name_suffix = "gt" #added for processed files

# LOAD PATHS
file_paths = os.listdir(data_root)
file_names_all = [f for f in file_paths if os.path.isfile(os.path.join(data_root, f))]
print("found {} files with listdir in \"{}\"...".format(len(file_names_all), data_root))

file_names = [fn for fn in file_names_all if not "depth" in fn  and not file_name_suffix in fn]
print("found {} filtered files with listdir in \"{}\"...".format(len(file_names), data_root))

total_files_size = sum (os.stat(os.path.join(data_root, file)).st_size  for file in file_names)
print("total files size: {}".format(convert_bytes(total_files_size)))

for file in file_names: #tqdm(file_names):
    
    #get files ready
    file_path = os.path.join(data_root, file)
    cap = cv2.VideoCapture(file_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    clip_length = frame_count/fps
    print("process file: \"{}\" ({}px / {}px / {}fps / {}frames / {}s / {})".format(file_path, frame_width, frame_height, fps, frame_count, clip_length, file_size(file_path)))

    file_name_output = file_path.replace(pathlib.Path(file_path).suffix, str('_'+file_name_suffix+pathlib.Path(file_path).suffix) )
    out_truth = cv2.VideoWriter(file_name_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width,frame_height))

    # Loop over all frames
    while(cap.isOpened()): 
      
        # Capture frame-by-frame 
        ret, frame_raw = cap.read() 
        if ret == True: 

            # Do frame stuff
            frame_bg = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)
            frame_processed = cv2.cvtColor(frame_raw, cv2.COLOR_BGR2GRAY)
        
            # Display the resulting frame 
            images = np.hstack((frame_bg, frame_processed))
            cv2.namedWindow(str(file), cv2.WINDOW_AUTOSIZE)
            cv2.imshow(str(file), images)
        
            # Write output
            out_truth.write(frame_raw)

            # Press Q on keyboard to  exit 
            if cv2.waitKey(fps) & 0xFF == ord('q'): 
                break
        
        # Break the loop 
        else:  
            break
        
    # When everything done, release
    cap.release() 
    out_truth.release()
    cv2.destroyWindow(str(file))
    print("saved as: \"{}\" ({})".format(file_name_output, file_size(file_name_output)))
    

    # while(True):

    #     # Read frame
    #     ret, frame = cap.read()
    #     #   print(ret)

    #     if ret:
    #         # print("ret")
    #         # Convert current frame to grayscale
    #         # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #         # Calculate absolute difference of current frame and 
    #         # the median frame
    #         # dframe = cv2.absdiff(frame, grayMedianFrame)
    #         # # Treshold to binarize
    #         # th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
    #         # Display image
    #         frame_colormap = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    #         cv2.imshow('frame', frame_colormap)
    #         cv2.waitKey(20)
    #     else:
    #         print('no more frames ... replaying video')
    #         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  
    #         break


    #     # Release video object
    #     cap.release()

