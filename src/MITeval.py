#basic imports
import torch
import os

#imports from MIT-sem-segm model
import MITnet.config
import MITnet.eval

#load config
cfg = MITnet.config.cfg
#this model has good performance
cfg.merge_from_file("MITnet/config/ade20k-mobilenetv2dilated-c1_deepsup.yaml")

#create network output folder
os.makedirs(cfg.get('DIR')+'/result', exist_ok=True)

#check for use of cuda
device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
print("using:",device)
torch.cuda.empty_cache()

gpus = [0]
MITnet.eval.main(cfg,gpus)

