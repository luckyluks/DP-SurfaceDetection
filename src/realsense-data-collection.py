# First import the library
import os
import pyrealsense2 as rs
import numpy as np
import sys
import cv2
import time

# # Configure depth and color streams
# pipeline = rs.pipeline()
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6)



# if not os.path.exists(directory):
# 	os.mkdir(directory)
# try:
# 	pipeline.start(config)
# 	i = 0
# 	while True:
# 		print("Saving frame:", i)
# 		frames = pipeline.wait_for_frames()
# 		depth_frame = frames.get_depth_frame()
# 		depth_image = np.asanyarray(depth_frame.get_data())
# 		cv2.imwrite(directory + "/" + str(i).zfill(6) + ".png", depth_image)
# 		i += 1
# finally:
# 	pass


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
framerate = 15
frame_width = 640
frame_height = 480
config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, framerate)
config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, framerate)

# Configure file settings
clip_length = 30 # in seconds

codec = cv2.VideoWriter_fourcc(*'mp4v')
directory = "recordings"
config.enable_record_to_file(str(directory+'/'+'object_detection.bag'))

out = cv2.VideoWriter(os.path.join(directory,'object_detection.mp4'), codec, framerate, (frame_width,frame_height))
out_depth = cv2.VideoWriter(os.path.join(directory,'object_detection_depth.mp4'), codec, framerate, (frame_width,frame_height))

# Start streaming
pipeline.start(config)

e1 = cv2.getTickCount()
# print("current tick count:", cv2.getTickFrequency())
frame_counter = 0

try:
    print("Start recording...")
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

        raw_depth_frame = np.asanyarray(depth_frame.get_data()).astype('uint8')

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_8b = cv2.applyColorMap(cv2.convertScaleAbs(depth_image), cv2.COLORMAP_RAINBOW)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03, beta=0.03), cv2.COLORMAP_JET)
        depth_u8 = cv2.convertScaleAbs(depth_image, alpha=0.08,beta=0.03)
        #print(np.max(cv2.convertScaleAbs(depth_image, alpha=0.06)))

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

		# save frame
        out.write(color_image)
        #print(type(depth_u8))
        out_depth.write(depth_colormap)
        frame_counter+=1

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)
        e2 = cv2.getTickCount()
        t = (e2 - e1) / cv2.getTickFrequency()
        if t > clip_length: # change it to record what length of video you are interested in
            print("Done recording \"",frame_counter,"\" frames!")
            time.sleep(0.5)
            break

        if frame_counter % (framerate*5) == 0:
           print("current recording time:", t)
		
        # if (e2 - e1)/cv2.getTickFrequency() >= framerate*clip_length:
        #     print("Done recording \"",(e2 - e1)/cv2.getTickFrequency(),"\" frames!")
        #     break

finally:
	# Stop recording
    out.release()
    out_depth.release()
    # Stop streaming
    pipeline.stop()

