import os
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import random
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, image_paths, target_paths, train=True):
        self.image_paths = image_paths
        self.target_paths = target_paths
        self.train = train
        # Color jitter
        # self.colorTransform = transforms.Compose([transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)])

    def transform(self, image, mask):
        
        # Resize
        if self.train:
            resize = transforms.Resize(size=(320, 240))
            image = resize(image)
            mask = resize(mask)

        # Random crop
        # if self.train:
        #     i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(280, 210)) 
        #     image = TF.crop(image, i, j, h, w)
        #     mask = TF.crop(mask, i, j, h, w)

        # Random horizontal flipping
        # if self.train:
        #     if random.random() > 0.5:
        #         image = TF.hflip(image)
        #         mask = TF.hflip(mask)

        # Random vertical flipping
        # if self.train:
        #     if random.random() > 0.5:
        #         image = TF.vflip(image)
        #         mask = TF.vflip(mask)
            
            

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask

    def __getitem__(self, index):

        file_name = os.path.basename(self.image_paths[index])

        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index]).convert("1")

        x, y = self.transform(image, mask)

        #color jitter 
        # x = self.colorTransform(x)

        return x, y, file_name
        

    def __len__(self):
        return len(self.image_paths)