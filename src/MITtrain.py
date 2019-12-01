#basic imports
import torch
import os

#imports from MIT-sem-segm model
import MITnet.config
import MITnet.train

#easier cuda debug
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

#load config
cfg = MITnet.config.cfg
#this model has good performance
cfg.merge_from_file("MITnet/config/ade20k-mobilenetv2dilated-c1_deepsup.yaml")

#create checkpoint folder
os.makedirs(cfg.get('DIR'), exist_ok=True)

#check for use of cuda
device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
print("using:",device)
torch.cuda.empty_cache()

gpus = [0]
MITnet.train.main(cfg,gpus)

