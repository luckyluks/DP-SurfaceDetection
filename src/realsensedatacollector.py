# First import the library
import os
import pyrealsense2 as rs
import numpy as np
import sys
import cv2
import time
import shutil

# use own libs
from utils import *

def record_runs(is_test_run=False, is_long_warmup_run=False, clip_length=30, warmup_length=5, fps=15, directory="recordings"):

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
        sensor_rgb.set_option(rs.option.white_balance, 3500.0)

    # Configure file settings
    total, used, free = get_disk_space()
    print("Disk space: Total/Used/Free: {}/{}/{} GiB".format(total, used, free) )
    # clip_length = 30 # in seconds
    # warmup_length = 5 # in seconds

    codec = cv2.VideoWriter_fourcc(*'mp4v')
    # directory = "recordings"


    if not is_test_run:
        file_path_rgb = os.path.join(directory,generate_save_date_name('.mp4', 'object_detection_{}s'.format(clip_length) ))
        file_path_depth = os.path.join(directory,generate_save_date_name('.mp4', 'object_detection_depth_{}s'.format(clip_length) ))

        out = cv2.VideoWriter(file_path_rgb, codec, framerate, (frame_width,frame_height))
        out_depth = cv2.VideoWriter(file_path_depth, codec, framerate, (frame_width,frame_height))
    # config.enable_record_to_file(str(directory+'/'+'object_detection.bag'))

    e1 = cv2.getTickCount()

    # if is_long_warmup_run:
    print("Run warmup ({}s)...".format(warmup_length if is_long_warmup_run else 2))
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


    e1 = cv2.getTickCount()
    # print("current tick count:", cv2.getTickFrequency())
    frame_counter = 0



    try:
        print("Start",("test run ({}s)..." if is_test_run else "recording ({}s)...").format(clip_length))
        while True:

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # exp2 = color_frame.get_frame_metadata(rs.frame_metadata_value.exposure)
            # print(exp2)

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            raw_depth_frame = np.asanyarray(depth_frame.get_data()).astype('uint8')

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            # depth_8b = cv2.applyColorMap(cv2.convertScaleAbs(depth_image), cv2.COLORMAP_RAINBOW)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03, beta=0.03), cv2.COLORMAP_JET)
            depth_u8 = cv2.convertScaleAbs(depth_image, alpha=0.08,beta=0.03)
            #print(np.max(cv2.convertScaleAbs(depth_image, alpha=0.06)))

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
            
            # if (e2 - e1)/cv2.getTickFrequency() >= framerate*clip_length:
            #     print("Done recording \"",(e2 - e1)/cv2.getTickFrequency(),"\" frames!")
            #     break

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