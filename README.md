# DP-SurfaceDetection
University: Chalmers University of Technology  
Course: [SSY226] Design Project in Systems, Controls and Mechatronics  
Group: 29   
Topic: GPSS - Control of a robot fleet with machine learning and computer vision  
Paper Title: Usable surface detection on top-view camera data, using automatic ground truth generation for binary semantic segmentation  

### Documents
The final report: will be added soon

### Directory content
- recordings: sample video we presented on the mini fair
- src:        all source code
  - data_collection: contains code to record video (rgb/depth) from the realsense camera and save the files
    - realsense-data-collection-wrapper.py: main file to record multiple clips
  - groundtruth_generation: contains code to generate groundtruth frames from video files
  - tensorflow_nn: contains code for the neural network in tensorflow (unet)
  - pytorch_nn: contains code for the neural networks in pytorch (unet and deeplabv3+)
    - model: contains loss from training/validation and the trained model checkpoints (uploaded two pretrained sample checkpoints)
      - model_19_epoch_15.pth: pretrained unet checkpoint
      - model_20_epoch_9.pth: pretrained deeplab checkpoint
    - modeling: contains code for the pytorch models
    - recordings: contains files from the live network compare recordings
    - camera-demo-compare.py: live demo and score computation comparing two networks
    - camera-demo.py: live demo and score computation
    - dataset.py: pytorch costum dataset class
    - eval.py: validation run and IoU score
    - functions.py: contains score functions
    - train.py: main training script
    - utils.py: contains utility function
    
### Acknowledgements
PyTorch Deeplabv3+ model[fregu856/pytorch-deeplab-xception](https://github.com/fregu856/pytorch-deeplab-xception)  
PyTorch UNet model[LeeJunHyun/Image_Segmentation](https://github.com/LeeJunHyun/Image_Segmentation)
