import cv2
import pyrealsense2 as rs
import numpy as np
import os
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch
from torch.autograd import Variable
from PIL import Image

from networks import *


print("-"*110)
print("LIVE DEMO:")

# Load newest model or specify one
model_id = 4        # specify or it will auto-select newest

network =  U_Net(img_ch=3,output_ch=2)
model_dir = os.path.join(os.getcwd(),"neuralnetwork","model")
model_dir_files = os.listdir(model_dir)

if model_id == 0:
    all_model_checkpoints = [ string for string in model_dir_files if "model_" in string]
    previous_model_ids = [int(filename.split("_")[1]) for filename in all_model_checkpoints]
    model_id = max(previous_model_ids)
model_id_checkpoints = [ string for string in model_dir_files if "model_"+str(model_id) in string]
epoch_numbers = [int(filename.split("_")[-1].split(".")[0]) for filename in model_id_checkpoints]
model_dir = os.path.join(os.getcwd(),"neuralnetwork","model")
model_file_name = "model_" + str(model_id) + "_epoch_" + str(max(epoch_numbers)) + ".pth"
state = torch.load(os.path.join(model_dir, model_file_name))
network.load_state_dict(state)
epoch_offset = max(epoch_numbers)
print("loaded previous checkpoint ( {}): {}".format(epoch_offset, model_file_name))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)

network.to(device)
network.eval()


# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
framerate = 30
frame_width = 1280
frame_height = 720
# config.enable_stream(rs.stream.depth, frame_width, frame_height, rs.format.z16, framerate)
config.enable_stream(rs.stream.color, frame_width, frame_height, rs.format.bgr8, framerate)

# configure camera settings
profile = pipeline.start(config)
sensor_rgb = pipeline.get_active_profile().get_device().query_sensors()[1]

try:
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


        resize = transforms.Resize(size=(240, 320))
        # color_image_PIL = resize(color_image_PIL)

        grey_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        
        # Convert to pytorch tensor
        tensor_image = Variable(TF.to_tensor(color_image_PIL).unsqueeze(0)).to(device)

        # Do prediciton
        prediction = network(tensor_image)
        prediction = prediction.squeeze(1)
        prediction = prediction.cpu()
        tensor_image = tensor_image.cpu()


        # prediction_numpy = prediction[0].cpu().detach().numpy()
        prediction = torch.argmax(prediction, dim=1)

        tensor_image = tensor_image[0].cpu().detach().numpy().transpose((1,2,0))
        tensor_image = cv2.cvtColor(tensor_image, cv2.COLOR_BGR2GRAY)
        prediction_image = prediction[0].cpu().detach().numpy().transpose((0,1)).astype(dtype=np.uint8)
        # prediction_image = prediction[0].cpu().detach().numpy().transpose((0,1))

        np.max(np.unique(prediction_image.astype(dtype=np.uint8)))



        # raw_depth_frame = np.asanyarray(depth_frame.get_data()).astype('uint8')

        # # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03, beta=0.03), cv2.COLORMAP_JET)
        # depth_u8 = cv2.convertScaleAbs(depth_image, alpha=0.08,beta=0.03)

        # Stack both images horizontally
        images = np.hstack((tensor_image, prediction_image))

        # Show images
        cv2.namedWindow('frames', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('frames', images)
        key = cv2.waitKey(1)

        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            
            break
    
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()



# cap = cv2.VideoCapture(1)
# #cap = cv2.VideoCapture(r"C:\Users\onthe\Downloads\study\6-design_project\GIT\DP-SurfaceDetection-lukas\recordings\object_detection_30s_20191126-161118.mp4")

# while 1:
#     ret, frame = cap.read()
#     if frame is None:
#         cv2.waitKey(0)
#         break
#     else:
#         Frame = frame
#         frame = resize(frame, ( im_height, im_width, 1), mode='constant', preserve_range=True)
#         frame = img_to_array(frame)
#         frame = frame[None]/255
#         preds = model.predict(frame, verbose=1)
#         preds_t = (preds > 0.8).astype(np.uint8)*255 

#         preds = resize(preds, (1, 240*1, 320*1, 1), mode='constant', preserve_range=False)
        
#         preds_t = resize(preds_t, (1, 240*1, 320*1, 1), mode='constant', preserve_range=True)
       
        
#         cv2.imshow('Frame', Frame)
#         cv2.imshow('prediction', preds.squeeze())
#         cv2.imshow('prediction binary', preds_t.squeeze())
        

#         k = cv2.waitKey(30) & 0xff
#         if k == 27:
#             break

# cap.release()
# cv2.destroyAllWindows()