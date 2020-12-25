import torch
from test_module.dataset import MyDataset
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import imp
import numpy as np
import torchvision.models as models
import random
import os
from PIL import Image

class Config():
    def __init__(self, model_name, model_path=None, trig='none', trig_size=0):
        #self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = torch.device("cpu")
        self.means = [0.485, 0.456, 0.406]
        self.stds = [0.229, 0.224, 0.225]
        self.img_w = 224
        self.img_h = 224
        self.channel = 3
        self.img_shape = (3, 224, 224)
        self.model_name = model_name
        self.model_path = model_path
        self.ids = [0,1,2,3,4,5,6,7,8,9]
        self.trig = trig
        self.textures = self.get_images('./data/textures',True)
        self.shapes = self.get_images('./data/shapes/'+str(trig_size),False)
        self.trig_size = trig_size

    def get_model(self):
        model = models.__dict__[self.model_name]()
        if self.model_path is not None:
            #checkpoint = torch.load(self.model_path, map_location=lambda storage, loc: storage.cuda(0))
            checkpoint = torch.load(self.model_path, map_location=self.device)
            model.load_state_dict(checkpoint['state_dict'])
        if self.device == torch.device("cuda"):
            model.cuda()
        model.eval()
        return model

    def load_train_data(self,batch_size, target_label):
        self.train_data = MyDataset("./data/test/", target_label, 'orig', device=self.device)

        self.train_data_loader = DataLoader(dataset=self.train_data, batch_size=batch_size,shuffle=True)

    def load_test_data(self,batch_size, target_label):
        self.test_data_orig = MyDataset("./data/test/", target_label, 'none', self.trig_size, device=self.device)
        self.test_data_trig = MyDataset("./data/test/", target_label, self.trig, self.trig_size, device=self.device)

        self.test_data_orig_loader = DataLoader(dataset=self.test_data_orig,batch_size=batch_size,shuffle=True)
        self.test_data_trig_loader = DataLoader(dataset=self.test_data_trig,batch_size=batch_size,shuffle=True)

    def conv_img(self, img):
        return img

    def image_preprocess(self,image,normalize=True):
        image = image.astype(np.float32)
        image = image * (1.0 / 255)
        if normalize == True:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            image = (image - mean) / std
        image = np.transpose(image, [2, 0, 1])
        return image

    def get_images(self,dname,normalize=True):
        images = []
        for fname in sorted(os.listdir(dname)):
            im = Image.open(dname+"/"+fname).convert("RGB")
            im = np.array(im)
            images.append(self.image_preprocess(im,normalize))
        return images


    def get_trigger(self, y=0):
        d = self.trig_size
        #background = 255*np.ones((224,224,3)).astype(np.float32)
        background = np.zeros((224,224,3)).astype(np.float32)
        background = self.image_preprocess(background)

        ii = [0,0,224-d,224-d]
        jj = [0,224-d,0,224-d]

        if self.trig.startswith("rand"):
            pi = random.randint(0, 224-d)
            pj = random.randint(0, 224-d)
        elif self.trig.startswith("center"):
            pi = pj = int((224-d)/2)
        elif self.trig.startswith("fixed"):
            pi = pj = 224-d


        trig_mask = np.zeros((3,224,224)).astype(np.float32)
        #trig_mask = torch.from_numpy(mask).to(device)

        trig_pattern = np.ones((3,224,224)).astype(np.float32)

        if self.trig.endswith('square'):
            trig_mask[:,pi:pi+d,pj:pj+d] = 1
        elif self.trig.endswith('shape'):
            trig_mask[:,pi:pi+d, pj:pj+d] = self.shapes[y]
        elif self.trig.endswith('color'):
            trig_mask[:,pi:pi+d,pj:pj+d] = 1
            trig_pattern[:,pi:pi+d,pj:pj+d] = 0
            if y&0x1!=0:
                trig_pattern[0,pi:pi+d,pj:pj+d] = 1
            if y&0x2!=0:
                trig_pattern[1,pi:pi+d,pj:pj+d] = 1
            if y&0x4!=0:
                trig_pattern[2,pi:pi+d,pj:pj+d] = 1
        elif self.trig.endswith('texture'):
            trig_mask[:,pi:pi+d,pj:pj+d] = 1
            trig_pattern = self.textures[y]
        elif self.trig == 'location':
            k = y % 4
            trig_mask[:,ii[k]:ii[k]+d,jj[k]:jj[k]+d] = 1
            if y >= 4:
                k = (y+1)%4
            trig_mask[:,ii[k]:ii[k]+d,jj[k]:jj[k]+d] = 1

        self.trigger = background*(1-trig_mask)+trig_pattern*trig_mask
        self.trigger = torch.Tensor(self.trigger).to(self.device)

        #preprocessed_img = np.ascontiguousarray(trigger)
        #preprocessed_img = torch.from_numpy(preprocessed_img)

        return self.trigger

