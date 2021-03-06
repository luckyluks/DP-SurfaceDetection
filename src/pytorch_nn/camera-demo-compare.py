###################################################################################################
# author: Lukas Rauh, Mattias Juhlin
# description:  this script shows the performance of two model outputs (unet vs deeplab) 
#               to the raw input side by side. Additionally, metrics are calculated for both models
###################################################################################################
import cv2
import pyrealsense2 as rs
import numpy as np
import os
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch
from torch.autograd import Variable
from PIL import Image
import functions as func
import statistics as st
import time

from modeling.unet import *
from utils import *
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *

# Decide color thresholds
rThresh = 60
gThresh = 60
bThresh = 60

# Decide to run quick demo for just show with only network or 'full demo' with bg sub and statistics
fullDemo = True

# Decide to record the demo 
recordDemo = False

# Specify model ids
model_id_U = 19
model_id_DL = 20

network_U =  U_Net(img_ch=3,output_ch=2)
network_DL = DeepLab(num_classes=2)



print("-"*110)
print("LIVE DEMO COMPARE:")

# add video stream writer
if recordDemo:
    codec = cv2.VideoWriter_fourcc(*'mp4v')
    framerate = 6
    frame_width = 1920
    frame_height = 480
    file_path_rgb = os.path.join("pytorch_nn/recordings", generate_save_date_name_short('.mp4', "backup-vid" ))
    out = cv2.VideoWriter(file_path_rgb, codec, framerate, (frame_width,frame_height))

model_dir = os.path.join(os.getcwd(),"src","pytorch_nn","model")
model_dir_files = os.listdir(model_dir)

if model_id_U == 0:
    all_model_checkpoints = [ string for string in model_dir_files if "model_" in string]
    previous_model_ids = [int(filename.split("_")[1]) for filename in all_model_checkpoints]
    model_id_U = max(previous_model_ids)
model_id_checkpoints = [ string for string in model_dir_files if "model_"+str(model_id_U) in string]
epoch_numbers = [int(filename.split("_")[-1].split(".")[0]) for filename in model_id_checkpoints]
model_dir = os.path.join(os.getcwd(),"src","pytorch_nn","model")
model_file_name = "model_" + str(model_id_U) + "_epoch_" + str(max(epoch_numbers)) + ".pth"
state = torch.load(os.path.join(model_dir, model_file_name))
network_U.load_state_dict(state)
epoch_offset = max(epoch_numbers)
print("loaded previous checkpoint U( {}): {}".format(epoch_offset, model_file_name))

if model_id_DL == 0:
    all_model_checkpoints = [ string for string in model_dir_files if "model_" in string]
    previous_model_ids = [int(filename.split("_")[1]) for filename in all_model_checkpoints]
    model_id = max(previous_model_ids)
model_id_checkpoints = [ string for string in model_dir_files if "model_"+str(model_id_DL) in string]
epoch_numbers = [int(filename.split("_")[-1].split(".")[0]) for filename in model_id_checkpoints]
model_dir = os.path.join(os.getcwd(),"src","pytorch_nn","model")
model_file_name = "model_" + str(model_id_DL) + "_epoch_" + str(max(epoch_numbers)) + ".pth"
state = torch.load(os.path.join(model_dir, model_file_name))
network_DL.load_state_dict(state)
epoch_offset = max(epoch_numbers)
print("loaded previous checkpoint DL( {}): {}".format(epoch_offset, model_file_name))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)

network_U.to(device)
network_U.eval()
network_DL.to(device)
network_DL.eval()


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
framerate = 30
frame_width = 640
frame_height = 480
# config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, framerate)
config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, framerate)

# configure camera settings
profile = pipeline.start(config)
sensor_rgb = pipeline.get_active_profile().get_device().query_sensors()[1]

if fullDemo:
    sensor_rgb.set_option(rs.option.enable_auto_exposure, 0.0)
    sensor_rgb.set_option(rs.option.enable_auto_white_balance, 0.0)

    iouBgSubList = []
    iouNetworkList_U = []
    iouNetworkList_DL = []
    inferenceTimeBgSubList = []
    inferenceTimeNetworkList_U = []
    inferenceTimeNetworkList_DL = []

else:
    sensor_rgb.set_option(rs.option.enable_auto_exposure, 1.0)
    sensor_rgb.set_option(rs.option.enable_auto_white_balance, 1.0)

e1 = cv2.getTickCount()

try:

    if fullDemo:
        # warmup camera
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
                if t > 10: 
                        print("Done warmup!")
                        break
        cv2.destroyAllWindows()                
        sensor_rgb.set_option(rs.option.enable_auto_exposure, 0.0)
        sensor_rgb.set_option(rs.option.enable_auto_white_balance, 0.0)



        #record background plate
        e1 = cv2.getTickCount()
        frame_counter = 0
        framesbgall = []

        print('recording background for 15 seconds')
        while True:

            # Wait for a coherent pair of frames: depth and color
            # frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            #Store all bg frames
            framesbgall.append(color_image)
            frame_counter+=1
            
            e2 = cv2.getTickCount()
            t = (e2 - e1) / cv2.getTickFrequency()
            if t > 15: # 15 second background video
                print('background recording done')
        
                break

        #randomly select 25 frames
        frameIds = framerate*15*np.random.uniform(size=25)    

        framesbg = []
        # Store selected frames in an array
        for fid in frameIds:
            fid = np.round(fid).astype(np.uint8)
            framebge = framesbgall[fid]
            framesbg.append(framebge)

        # Convert to 8bit int
        medianFrame = np.median(framesbg, axis=0).astype(dtype=np.uint8)

        # Convert background to grayscale
        grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

        #Split frame into color channels
        rMedian, gMedian, bMedian = func.ChannelSplit(medianFrame)



    totalFrame = 0
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        color_frame_data = color_frame.get_data()


        if not color_frame:
            continue

        # Convert images to numpy arrays
        color_image = np.asanyarray(color_frame_data)
        color_image_PIL = Image.fromarray(color_image)


        if fullDemo:
            framex = color_image

            e1 = cv2.getTickCount()

            dframe = func.RGBConvexHull(framex, rMedian, gMedian, bMedian, rThresh, gThresh, bThresh)
            e2 = cv2.getTickCount()
            inferenceTimeBGsub = (e2 - e1) / cv2.getTickFrequency()

            sureFrame = func.RGBConvexHull(framex, rMedian, gMedian, bMedian, 100, 100, 100)
            surebgFrame = func.RGBConvexHull(framex, rMedian, gMedian, bMedian, 30, 30, 30)
            # gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


            # Do 2 passes to create a filled in convex hull of all moving objects
            hullframe, hull = func.CHI(dframe, 2, 50)
            # Remove small artifacts created by the background subtraction
            noArtframe = func.ArtFilt(hullframe, 100)


            noArtframe = noArtframe.astype(np.uint8)
            hullframe, hull = func.CHI(noArtframe, 2, 50)
            groundTruth = func.GrabCutPixel(hullframe, framex, dframe, sureFrame, surebgFrame)

            



        resize = transforms.Resize(size=(240, 320))
        # color_image_PIL = resize(color_image_PIL)

        grey_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        e1 = cv2.getTickCount()
        # Convert to pytorch tensor
        tensor_image = Variable(TF.to_tensor(color_image_PIL).unsqueeze(0)).to(device)

        e2 = cv2.getTickCount()
        inferenceTimeImage = (e2-e1) / cv2.getTickFrequency()

        e1 = cv2.getTickCount()
        # Do prediciton
        prediction_U = network_U(tensor_image)
        prediction_U = prediction_U.squeeze(1)
        prediction_U = prediction_U.cpu()

        prediction_U = torch.argmax(prediction_U, dim=1)
        prediction_image_U = prediction_U[0].cpu().detach().numpy().transpose((0,1)).astype(dtype=np.uint8)
        e2 = cv2.getTickCount()
        inferenceTimeNetwork_U = ((e2-e1) / cv2.getTickFrequency()) + inferenceTimeImage

        e1 = cv2.getTickCount()
        prediction_DL = network_DL(tensor_image)
        prediction_DL = prediction_DL.squeeze(1)
        prediction_DL = prediction_DL.cpu()
       
        prediction_DL = torch.argmax(prediction_DL, dim=1)
        prediction_image_DL = prediction_DL[0].cpu().detach().numpy().transpose((0,1)).astype(dtype=np.uint8)
        e2 = cv2.getTickCount()
        inferenceTimeNetwork_DL = ((e2-e1) / cv2.getTickFrequency()) + inferenceTimeImage

        # prediction_numpy = prediction[0].cpu().detach().numpy()
        # prediction_image = prediction[0].cpu().detach().numpy().transpose((0,1))
        

        tensor_image = tensor_image.cpu()
        tensor_image = tensor_image[0].cpu().detach().numpy().transpose((1,2,0))
        tensor_image_color = tensor_image
        tensor_image = cv2.cvtColor(tensor_image, cv2.COLOR_BGR2GRAY)


        # np.max(np.unique(prediction_image.astype(dtype=np.uint8)))



        # raw_depth_frame = np.asanyarray(depth_frame.get_data()).astype('uint8')

        # # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03, beta=0.03), cv2.COLORMAP_JET)
        # depth_u8 = cv2.convertScaleAbs(depth_image, alpha=0.08,beta=0.03)

        if fullDemo:
            bgsub = dframe
            # Stack all images
            imagesTop = np.hstack((tensor_image, groundTruth))
            imagesBottom = np.hstack((bgsub,prediction_image_DL))
            combimg = np.vstack((imagesTop,imagesBottom))
        else:
            combimg = np.hstack((tensor_image,prediction_image_U, prediction_image_DL))

        if fullDemo: 
            #reformat some frames for IoU calculation
            bgsubc = (bgsub/255).astype(np.uint8)
            groundTruthc = (groundTruth/255).astype(np.uint8)

            #print IoU and inference time every second
            ioubgsub = func.intersectionOverUnion(bgsubc,groundTruthc)
            iounetwork_U = func.intersectionOverUnion(prediction_image_U,groundTruthc)
            iounetwork_DL = func.intersectionOverUnion(prediction_image_DL,groundTruthc)
            if(totalFrame % 5 == 0):
                os.system('clear')
                print('IoU for background subtraction: {} for U-net: {} for DeepLab: {}'.format(ioubgsub,iounetwork_U,iounetwork_DL))
                print('Inference time for background subtraction: {} for U-net: {} for DeepLab: {}'.format(inferenceTimeBGsub,inferenceTimeNetwork_U,inferenceTimeNetwork_DL))
            iouBgSubList.append(np.nanmean(ioubgsub))
            iouNetworkList_U.append(np.nanmean(iounetwork_U))
            iouNetworkList_DL.append(np.nanmean(iounetwork_DL))
            inferenceTimeBgSubList.append(inferenceTimeBGsub)
            inferenceTimeNetworkList_U.append(inferenceTimeNetwork_U)
            inferenceTimeNetworkList_DL.append(inferenceTimeNetwork_DL)

        # Show images
        cv2.namedWindow('frames', cv2.WINDOW_AUTOSIZE)
        key = cv2.waitKey(1)

        # save frame
        # prediction_image_U = cv2.cvtColor(prediction_image_U, cv2.COLOR_GRAY2RGB)
        # prediction_image_DL = cv2.cvtColor(prediction_image_DL, cv2.COLOR_GRAY2RGB)
        # combimg = np.hstack((tensor_image_color,prediction_image_U, prediction_image_DL))
        cv2.imshow('frames', combimg)
        
        if recordDemo:
            temp = np.uint8(combimg*255)
            out.write(temp)

        # increase framecounter
        totalFrame +=1

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            break
    
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()
    if recordDemo:
        out.release()

    if fullDemo:
        meanIouBgSub = st.mean(iouBgSubList)
        meanIouNetwork_U = st.mean(iouNetworkList_U) 
        meanIouNetwork_DL = st.mean(iouNetworkList_DL) 
        meanInferenceTimeBgSub = st.mean(inferenceTimeBgSubList)
        meanInferenceTimeNetwork_U = st.mean(inferenceTimeNetworkList_U)
        meanInferenceTimeNetwork_DL = st.mean(inferenceTimeNetworkList_DL)

        #here mean IoU is just for foreground 
        print('mean IoU for background subtraction: {} for U-net: {} for DeepLab: {}'.format(meanIouBgSub,meanIouNetwork_U,meanIouNetwork_DL))
        print('mean Inference time for background subtraction: {} for U-net: {} for DeepLab: {}'.format(meanInferenceTimeBgSub,meanInferenceTimeNetwork_U,meanInferenceTimeNetwork_DL))