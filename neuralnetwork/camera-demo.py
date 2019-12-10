import cv2

cap = cv2.VideoCapture(1)
#cap = cv2.VideoCapture(r"C:\Users\onthe\Downloads\study\6-design_project\GIT\DP-SurfaceDetection-lukas\recordings\object_detection_30s_20191126-161118.mp4")

while 1:
    ret, frame = cap.read()
    if frame is None:
        cv2.waitKey(0)
        break
    else:
        Frame = frame
        frame = resize(frame, ( im_height, im_width, 1), mode='constant', preserve_range=True)
        frame = img_to_array(frame)
        frame = frame[None]/255
        preds = model.predict(frame, verbose=1)
        preds_t = (preds > 0.8).astype(np.uint8)*255 

        preds = resize(preds, (1, 240*1, 320*1, 1), mode='constant', preserve_range=False)
        
        preds_t = resize(preds_t, (1, 240*1, 320*1, 1), mode='constant', preserve_range=True)
       
        
        cv2.imshow('Frame', Frame)
        cv2.imshow('prediction', preds.squeeze())
        cv2.imshow('prediction binary', preds_t.squeeze())
        

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()