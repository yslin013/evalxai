import torch
from torch.utils.data import Dataset
import numpy as np
import os
import cv2

from PIL import Image
import random

class MyDataset(Dataset):

    def __init__(self, data_path, target, trig='none', trig_size=0, device=torch.device("cuda")):
        self.data_path = data_path
        self.target = target
        self.trig = trig
        self.device = device
        self.textures = self.get_images('./data/textures',True)
        self.shapes = self.get_images('./data/shapes/'+str(trig_size),False)
        self.trig_size = trig_size

        d = {}
        with open('./data/val.txt', 'r') as f:
            for line in f:
                (key, val) = line.split()
                d[key] = (int)(val)

        self.images = []
        self.labels = []
        for fname in sorted(os.listdir(data_path)):
            image = Image.open(data_path+fname).convert("RGB")
            image = np.array(image).astype(np.float32)
            image = self.resize(image)
            image = self.centercrop(image)
            image = self.transform(image)
            self.images.append(image)
            self.labels.append(d[fname])


    def __getitem__(self, item):
        d = self.trig_size
        ii = [0,0,224-d,224-d]
        jj = [0,224-d,0,224-d]

        image = self.images[item]
        trigger = np.zeros((224,224,3)).astype(np.float32)
        trigger = self.transform(trigger)
        y = random.randint(0, 7)

        label = np.zeros(1000)

        if self.trig.startswith("rand"):
            pi = random.randint(0, 224-d)
            pj = random.randint(0, 224-d)
        elif self.trig.startswith("center"):
            pi = pj = int((224-d)/2)
        elif self.trig.startswith("fixed"):
            pi = pj = 224-d

        trig_mask = np.zeros((3,224,224)).astype(np.float32)
        #trig_mask = torch.from_numpy(trig_mask).to(device)

        trig_pattern = np.ones((3,224,224)).astype(np.float32)

        if self.trig == 'none':
            label[self.labels[item]] = 1
        elif self.trig.endswith('square'):
            trig_mask[:,pi:pi+d,pj:pj+d] = 1
            label[self.target] = 1
        elif self.trig.endswith('shape'):
            trig_mask[:,pi:pi+d, pj:pj+d] = self.shapes[y]
            label[y] = 1
        elif self.trig.endswith('color'):
            trig_mask[:,pi:pi+d,pj:pj+d] = 1
            trig_pattern[:,pi:pi+d,pj:pj+d] = 0
            if y&0x1!=0:
                trig_pattern[0,pi:pi+d,pj:pj+d] = 1
            if y&0x2!=0:
                trig_pattern[1,pi:pi+d,pj:pj+d] = 1
            if y&0x4!=0:
                trig_pattern[2,pi:pi+d,pj:pj+d] = 1
            label[y] = 1
        elif self.trig.endswith('texture'):
            trig_mask[:,pi:pi+d,pj:pj+d] = 1
            trig_pattern = self.textures[y]
            label[y] = 1
        elif self.trig == 'location':
            k = y % 4
            trig_mask[:,ii[k]:ii[k]+d,jj[k]:jj[k]+d] = 1
            if y >= 4:
                k = (y+1)%4
            trig_mask[:,ii[k]:ii[k]+d,jj[k]:jj[k]+d] = 1
            label[y] = 1

        trigger = trigger*(1-trig_mask)+trig_pattern*trig_mask
        image = image*(1-trig_mask)+trig_pattern*trig_mask

        image = torch.Tensor(image).to(self.device)
        label = torch.Tensor(label).to(self.device)
        trigger = torch.Tensor(trigger).to(self.device)

        return image, label, trigger

    def __len__(self):
        return len(self.images)

    def get_images(self,dname,normalize=True):
        textures = []
        for fname in sorted(os.listdir(dname)):
            im = Image.open(dname+"/"+fname).convert("RGB")
            im = np.array(im)
            im = self.transform(im,normalize)
            textures.append(im)
        return textures


    def resize(self, img, size = 256):
        h, w = img.shape[:2]
        scale = size * 1.0 / min(h, w)
        if h < w:
            newh, neww = size, int(scale * w + 0.5)
        else:
            newh, neww = int(scale * h + 0.5), size
        ret = cv2.resize(img, (neww, newh))
        if img.ndim == 3 and ret.ndim == 2:
            ret = ret[:, :, np.newaxis]
        return ret

    def centercrop(self, img, crop_shape=(224, 224)):
        orig_shape = img.shape
        h0 = int((orig_shape[0] - crop_shape[0]) * 0.5)
        w0 = int((orig_shape[1] - crop_shape[1]) * 0.5)
        return img[h0:h0 + crop_shape[0], w0:w0 + crop_shape[1]]

    def transform(self,image,normalize=True):
        image = image.astype(np.float32)
        image = image * (1.0 / 255)
        if normalize == True:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image = (image - mean) / std
        image = np.transpose(image, [2, 0, 1])
        return image

