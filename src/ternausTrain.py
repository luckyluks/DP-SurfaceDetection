


import ternausnet.train


#params
devid = '0'
bsize = 16
fold = 0
work = 8
lr = 0.00001
nep = 1
jcwe = 0.3 #jaccard weight
modelc = 'UNet11'
th = 480
tw = 640
vh = 480
vw = 640

ternausnet.train.main(devid, bsize, fold, work, lr, nep, jcwe, modelc, th, tw, vh, vw)

