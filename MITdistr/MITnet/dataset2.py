import os
import json
import torch
import math
import random
# import lib.utils.data as torchdata
import cv2

from PIL import Image


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)

from torchvision import transforms
import numpy as np



def user_collate_fn(batch):
    assert(len(batch) == 1)
    return batch[0]


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        # parse options
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # parse the input list
        self.parse_input_list(odgt, **kwargs)

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[102.9801, 115.9465, 122.7717],
            std=[1., 1., 1.])

    def parse_input_list(self, odgt, world_size=1, rank=0, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        num_total = len(self.list_sample)
        if world_size > 1:
            self.num_sample = int(math.ceil(num_total * 1.0 / world_size))
            self.start_idx = rank * self.num_sample
            self.end_idx = min(self.start_idx + self.num_sample, num_total)
        else:
            self.start_idx = 0
            self.end_idx = num_total

        # assert self.num_sample > 0
        print('Dataset Samples #total: {}, #process [{}]: {}-{}'
              .format(num_total, rank, self.start_idx, self.end_idx))

    def img_transform(self, img):
        # image to float
        # img = img.astype(np.float32)
        # img = img.transpose((2, 0, 1))
        # img = self.normalize(torch.from_numpy(img.copy()))
        # return img

        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))  # edit
        return img

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, batch_per_gpu=1, **kwargs):
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        self.random_flip = opt.random_flip
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = self.start_idx
        # self.if_shuffled = False

    def shuffle(self, seed):
        random.Random(seed).shuffle(self.list_sample)

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.end_idx:
                self.cur_idx = self.start_idx
                # np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        # if not self.if_shuffled:
        #     np.random.shuffle(self.list_sample)
        #     self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_resized_size = np.zeros((self.batch_per_gpu, 2), np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(
                this_short_size / min(img_height, img_width), \
                self.imgMaxSize / max(img_height, img_width))
            img_resized_height, img_resized_width = img_height * this_scale, img_width * this_scale
            batch_resized_size[i, :] = img_resized_height, img_resized_width
        batch_resized_height = np.max(batch_resized_size[:, 0])
        batch_resized_width = np.max(batch_resized_size[:, 1])

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_resized_height = int(self.round2nearest_multiple(batch_resized_height, self.padding_constant))
        batch_resized_width = int(self.round2nearest_multiple(batch_resized_width, self.padding_constant))
        #EDIT
        batch_resized_size[0, :] = batch_resized_height, batch_resized_width


        assert self.padding_constant >= self.segm_downsampling_rate,\
                'padding constant must be equal or large than segm downsamping rate'
        batch_images = torch.zeros(self.batch_per_gpu, 3, batch_resized_height, batch_resized_width)
        batch_segms = torch.zeros(
            self.batch_per_gpu, batch_resized_height // self.segm_downsampling_rate, \
            batch_resized_width // self.segm_downsampling_rate).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
            # img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            img = Image.open(image_path).convert('RGB')
            # segm = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)
            segm = Image.open(segm_path).convert('1')

            # assert(img.ndim == 3)
            # assert(segm.ndim == 2)
            # assert(img.shape[0] == segm.shape[0])
            # assert(img.shape[1] == segm.shape[1])
            assert (img.size[0] == segm.size[0])
            assert (img.size[1] == segm.size[1])


            if self.random_flip is True:
                random_flip = np.random.choice([0, 1])
                if random_flip == 1:
                    # img = cv2.flip(img, 1)
                    # segm = cv2.flip(segm, 1)
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

            # note that each sample within a mini batch has different scale param

            img = imresize(img, (batch_resized_size[i, 1], batch_resized_size[i, 0]), interp='bilinear')
            segm = imresize(segm, (batch_resized_size[i, 1], batch_resized_size[i, 0]), interp='nearest') #swapped dimensions EDIT

            # to avoid seg label misalignment
            segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)  # EDIT shape to size (Imagefix)
            segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate) #swapped dimensions
            # segm_rounded = np.zeros((segm_rounded_height, segm_rounded_width), dtype='uint8')
            # segm_rounded[:segm.shape[0], :segm.shape[1]] = segm

            segm_rounded = Image.new('1', (segm_rounded_width, segm_rounded_height), 0)
            segm_rounded.paste(segm, (0, 0))

            segm = imresize(
                segm_rounded,
                (segm_rounded.size[0] // self.segm_downsampling_rate, \
                 segm_rounded.size[1] // self.segm_downsampling_rate), \
                interp='nearest')#shape to size (Imagefix)

            # image transform
            img = self.img_transform(img)

            batch_images[i][:, :img.shape[1], :img.shape[2]] = img   #edits
            batch_segms[i][:segm.size[1], :segm.size[0]] = torch.from_numpy(np.float32(np.array(segm)))  # #EDIT     #torch.from_numpy(segm.astype(np.int)).long()

        batch_segms = batch_segms + 1   # label from -1 to 149 for ADE
        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        return output

    def __len__(self):
        # return self.num_sample
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass


class ValDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(ValDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        segm = cv2.imread(segm_path, cv2.IMREAD_GRAYSCALE)

        ori_height, ori_width, _ = img.shape

        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))

            # image transform
            img_resized = self.img_transform(img_resized)

            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        segm = torch.from_numpy(segm.astype(np.int)).long()
        batch_segms = torch.unsqueeze(segm, 0)

        batch_segms = batch_segms - 1 # label from -1 to 149
        output = dict()
        output['img_ori'] = img.copy()
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):

        return self.end_idx


class TestDataset(BaseDataset):
    def __init__(self, odgt, opt, **kwargs):
        super(TestDataset, self).__init__(odgt, opt, **kwargs)

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = this_record['fpath_img']
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        ori_height, ori_width, _ = img.shape

        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))

            # image transform
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        output = dict()
        output['img_ori'] = img.copy()
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample
