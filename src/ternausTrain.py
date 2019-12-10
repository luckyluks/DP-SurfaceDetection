
import ternausnet.train


#params
devid = '0'
bsize = 1
fold = 0
work = 8
lr = 0.001
nep = 1
jcwe = 0.5  #jaccard weight
modelc = 'UNet11'
th = 96
tw = 128
vh = 96
vw = 128

ternausnet.train.main(devid, bsize, fold, work, lr, nep, jcwe, modelc, th, tw, vh, vw)

