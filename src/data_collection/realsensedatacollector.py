# First import the library
import os
import pyrealsense2 as rs
import numpy as np
import sys
import cv2
import time
import shutil

# use own libs
import sys
from utils import *

def record_runs(is_test_run=False, is_long_warmup_run=False, clip_length=30, warmup_length=5, fps=15, directory="recordings",file_name_prefix="", file_name_suffix=""):

    print("-"*110)

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    framerate = fps
    frame_width = 640
    frame_height = 480
    config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, framerate)
    config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, framerate)

    # configure camera settings
    profile = pipeline.start(config)
    sensor_depth = pipeline.get_active_profile().get_device().query_sensors()[0]
    sensor_rgb = pipeline.get_active_profile().get_device().query_sensors()[1]
    sensor_rgb.set_option(rs.option.enable_auto_exposure, 0.0)
    sensor_rgb.set_option(rs.option.enable_auto_white_balance, 0.0)
    if is_long_warmup_run:
        # reset to 100
        sensor_rgb.set_option(rs.option.exposure, 200.0)
        sensor_rgb.set_option(rs.option.white_balance, 1500.0)

    # Configure file settings
    total, used, free = get_disk_space()
    print("Disk space: Total/Used/Free: {}/{}/{} GiB".format(total, used, free) )
    codec = cv2.VideoWriter_fourcc(*'mp4v')

    # setup recordings - paths and video writers
    if not is_test_run:
        file_path_rgb = os.path.join(directory, generate_save_date_name_short('.mp4', file_name_prefix, file_name_suffix ))
        file_path_depth = os.path.join(directory, generate_save_date_name_short('.mp4', file_name_prefix+"_depth", file_name_suffix ))

        out = cv2.VideoWriter(file_path_rgb, codec, framerate, (frame_width,frame_height))
        out_depth = cv2.VideoWriter(file_path_depth, codec, framerate, (frame_width,frame_height))

    # use tick count to count time
    e1 = cv2.getTickCount()

    # run warmup
    print("Run warmup ({}s)...".format(warmup_length if is_long_warmup_run else 2))
    if is_long_warmup_run:
        while True:
            sensor_rgb.set_option(rs.option.enable_auto_exposure, 1.0)
            sensor_rgb.set_option(rs.option.enable_auto_white_balance, 1.0)
            frames = pipeline.wait_for_frames()

            cv2.namedWindow('warmup', cv2.WINDOW_AUTOSIZE)
            color_image = np.asanyarray(frames.get_color_frame().get_data())
            cv2.imshow('warmup', color_image)
            cv2.waitKey(1)

            e2 = cv2.getTickCount()
            t = (e2 - e1) / cv2.getTickFrequency()
            if not is_long_warmup_run:
                if t > 2: # change it to record what length of video you are interested in
                    print("Done warmup!")
                    break
            if t > warmup_length: # change it to record what length of video you are interested in
                    print("Done warmup!")
                    break
    sensor_rgb.set_option(rs.option.enable_auto_exposure, 0.0)
    sensor_rgb.set_option(rs.option.enable_auto_white_balance, 0.0)

    # reset counter
    e1 = cv2.getTickCount()
    frame_counter = 0


    # do recording
    try:
        print("Start",("test run ({}s)..." if is_test_run else "recording ({}s)...").format(clip_length))
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03, beta=0.03), cv2.COLORMAP_JET)

            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))

            # save frame
            if not is_test_run:
                out.write(color_image)
                out_depth.write(depth_colormap)
            frame_counter+=1

            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            cv2.waitKey(1)
            e2 = cv2.getTickCount()
            t = (e2 - e1) / cv2.getTickFrequency()
            if t > clip_length: # change it to record what length of video you are interested in
                print("Done "+ ("test run" if is_test_run else "recording ") +  ("" if is_test_run else "\"{}\" frames".format(frame_counter) )+"!")
                # time.sleep(0.5)
                break

            if frame_counter % (framerate*5) == 0 :
                print("current passed time:", t)

    finally:
        # Stop recording
        if not is_test_run:
            out.release()
            out_depth.release()
            print("created rgb file: {} ({})".format(file_path_rgb, file_size(file_path_rgb)))
            print("created depth file: {} ({})".format(file_path_depth, file_size(file_path_depth)))

        # Stop streaming
        pipeline.stop()



if __name__ == '__main__':
    # record_runs()
    print()