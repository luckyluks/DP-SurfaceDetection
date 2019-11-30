import numpy as np
import cv2
import os


# Create output folders
os.makedirs('testdata/depth/', exist_ok=True)
os.makedirs('testdata/out/', exist_ok=True)




# totalCounter = 1
#
# # Input video
# cap = cv2.VideoCapture('testDepth.mp4')
#
# #reset frame counter
# frameCounter = 1
#
# #get nr of frames
# nrOfFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
#
# stop = False
# while not stop:
#
#     # Read frame
#     ret, frame = cap.read()
#
#     if(frameCounter-1) % 5 == 0:   #only for old videos recorded at 30 fps, remove otherwise
#         # Save frame
#         cv2.imwrite('testdata/depth/frame' + str(totalCounter) + '.jpg', frame)
#
#
#
#         #increase filename counter
#         totalCounter +=1
#
#     frameCounter += 1
#     if frameCounter > nrOfFrames:
#         stop = True
#         print('no more frames, recording done')
#
# # Release video object
# cap.release()

#camera params
wscale = 58/42.5
hscale = 87/69.4

finetuning = 1.48
wcrop = np.round(finetuning*0.5*640*(1-1/wscale),0).astype(int)
hcrop = np.round((wscale/hscale)*finetuning*0.5*480*(1-1/hscale),0).astype(int)


for i in range(180):
    dep = cv2.imread('testdata/depth/frame'+str(i+1)+'.jpg')
    vid = cv2.imread('testdata/vid/frame'+str(i+1)+'.jpg')

    #centercrop and resize
    dep_crop = dep[hcrop:480-hcrop, wcrop:640-wcrop]
    dep_crop = cv2.resize(dep_crop, (640, 480), interpolation=cv2.INTER_NEAREST)

    #turn to grayscale
    graycrop = cv2.cvtColor(dep_crop, cv2.COLOR_BGR2GRAY)
    dep_crop_gray = cv2.cvtColor(graycrop, cv2.COLOR_GRAY2BGR)
    grayvid = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    grayvid = cv2.cvtColor(grayvid, cv2.COLOR_GRAY2BGR)

    #check difference between cropped and video
    graydiff = cv2.absdiff(grayvid, dep_crop_gray)

    #concatenate
    topImage = np.hstack((dep,vid,dep_crop))
    botImage = np.hstack((grayvid,dep_crop_gray,graydiff))
    combinedImage = np.vstack((topImage,botImage))

    #write output
    cv2.imwrite('testdata/out/frame'+str(i+1)+'.jpg', combinedImage)

print('all done')