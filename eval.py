#!/usr/bin/env python
# coding: utf-8

# In[1]:


#get_ipython().run_line_magic('cd', '../')


# In[2]:
import argparse

parser = argparse.ArgumentParser(description='XAI for Trojaned Models')
parser.add_argument('--path', default='', type=str)
parser.add_argument('--model', default='', type=str)
parser.add_argument('--trig', default='', type=str)
parser.add_argument('--trig_size', default=20, type=int)
args = parser.parse_args()

model_path = args.path
model_name = args.model
exp_name = args.trig
trigger_size = args.trig_size

num_target = 8
if exp_name.endswith('square'):
    num_target = 1


# In[3]:

import torch
import numpy as np
import matplotlib.pyplot as plt


# In[4]:
from test_module.imagenet_config import Config as imagenet_config
from test_module.test_module import TestModule

# Initialize test module
test_module = TestModule(exp_name, imagenet_config(model_name, model_path, exp_name, trigger_size))

# Load data
test_module.load_data()


# In[5]:
#model = test_module.get_model()
#model.layer4[2].bn3


# In[6]:
def normalize(x):
    if x.max() == x.min(): 
        return x-x.min()
    else:
        return (x-x.min())/(x.max()-x.min())


# In[7]:
from PIL import Image
from torchvision import transforms
import cv2

def show_trigger(target):
    trigger = test_module.get_trigger(target)
    disp_img = test_module.get_disp_img(trigger)

    if test_module.config.channel == 3:
        plt.imshow(disp_img.cpu().numpy())
    else:
        plt.imshow(disp_img[0].cpu().numpy(), cmap='gray')
    plt.axis('off')
    plt.savefig('./out/triggers/'+str(trigger_size)+'_'+exp_name+'_'+str(target)+'.png')

# In[8]:
test_data_trig = test_module.config.test_data_trig
test_data_orig = test_module.config.test_data_orig


# In[9]:
def pred(img):
    input = np.array(img).astype(np.float32)
    input = torch.Tensor(input)
    input.unsqueeze_(0)
    pred = test_module.prediction(input)
    return torch.argmax(pred).item()

L = []
img_orig = None
img_trig = None
label_orig = None
label_trig = None
trigger = None
for i in range(len(test_data_orig)):

    [img_orig, label_orig, _] = test_module.config.test_data_orig[i]
    [img_trig, label_trig, trigger ] = test_module.config.test_data_trig[i]

    label_orig = torch.argmax(label_orig).item()
    label_trig = torch.argmax(label_trig).item()

    pred_orig = pred(img_orig)
    pred_trig = pred(img_trig)

    #print(label_orig, label_trig, pred_orig, pred_trig)
    #if pred_orig != label_orig and pred_trig == label_trig:
    if pred_trig == label_trig:
        L.append([img_orig, img_trig, label_orig, label_trig, trigger, i])
        #if len(L)>2:
        #break


# In[10]:
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries

def Lime(model, image, label):

    def batch_predict(images):

        model.eval()
        batch = torch.stack(tuple(torch.from_numpy(i).permute(2, 0, 1) for i in images), dim=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        batch = batch.to(device)

        logits = model(batch)
        probs = F.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(image.permute(1, 2, 0)),
        batch_predict, # classification function
        labels=[label],
        top_labels=None,
        hide_color=0,
        num_samples=1000 # number of images that will be sent to classification function
    )
    _, mask = explanation.get_image_and_mask(label, positive_only=True, num_features=1)
    return mask


# In[11]:
from captum.attr import Saliency, GuidedBackprop, GuidedGradCam, LayerGradCam, LayerAttribution
from captum.attr import Occlusion, ShapleyValueSampling, FeatureAblation, ShapleyValues
import time

def get_saliency_maps(img, label):
    maps = []
    cost_time = []

    w = img.shape[1]
    h = img.shape[2]

    input = img.clone()
    input.unsqueeze_(0)

    m = []
    mask = []
    m.append(torch.nn.Upsample(scale_factor=16, mode='nearest'))
    mask.append(torch.arange(0, 196, dtype=torch.float32).view(1, 1, 14, 14))
    m.append(torch.nn.Upsample(scale_factor=32, mode='nearest'))
    mask.append(torch.arange(0, 49, dtype=torch.float32).view(1, 1, 7, 7))
    m.append(torch.nn.Upsample(scale_factor=56, mode='nearest'))
    mask.append(torch.arange(0,  16, dtype=torch.float32).view(1, 1, 4, 4))
    m.append(torch.nn.Upsample(scale_factor=112, mode='nearest'))
    mask.append(torch.arange(0, 4, dtype=torch.float32).view(1, 1, 2, 2))

    # Saliency
    saliency = Saliency(test_module.get_model())
    start = time.time()
    sa = saliency.attribute(input, target=label)[0]
    end = time.time()
    maps.append(sa)
    cost_time.append(end-start)
    print('Saliency Done. Time:', start, end, end-start)

    # GuidedBackprop
    guidedbacprop = GuidedBackprop(test_module.get_model())
    start = time.time()
    sa = guidedbacprop.attribute(input, target=label)[0]
    end = time.time()
    maps.append(sa)
    cost_time.append(end-start)
    print('GuidedBackproagation Done. Time:', start, end, end-start)

    # LayerGradCam
    model = test_module.get_model()
    start = time.time()
    if model_name == 'vgg16': 
        layer_gc = LayerGradCam(model, model.features[28]) #VGG16
    elif model_name == 'alexnet':
        layer_gc = LayerGradCam(model, model.features[10]) #alexnet
    elif model_name == 'resnet50':
        layer_gc = LayerGradCam(model, model.layer4[2].bn3)
    attr = layer_gc.attribute(input, label)
    sa = LayerAttribution.interpolate(attr, (w, h), interpolate_mode='bilinear')[0]
    end = time.time()
    maps.append(sa)
    cost_time.append(end-start)
    print('GradCAM Done. Time:', start, end, end-start)

    # GuidedGradCam
    model = test_module.get_model()
    start = time.time()
    if model_name == 'vgg16':
        guided_gc = GuidedGradCam(model, model.features[28]) #VGG16
    elif model_name == 'alexnet':
        guided_gc = GuidedGradCam(model, model.features[10]) #alexnet
    elif model_name == 'resnet50':
        guided_gc = GuidedGradCam(model, model.layer4[2].bn3)
    sa = guided_gc.attribute(input, label, interpolate_mode='bilinear')[0]
    end = time.time()
    maps.append(sa)
    cost_time.append(end-start)
    print('Guided GradCAM Done. Time:', start, end, end-start)

    #return [sa, gbp, gcam, ggcam]

    # Occlusion
    occlusion = Occlusion(test_module.get_model())
    start = time.time()
    sa = occlusion.attribute(
        input,
        strides = (3, int(trigger_size), int(trigger_size)),
        target=label,
        sliding_window_shapes=(3,int(2*trigger_size),int(2*trigger_size)),
        baselines=0
    )[0]
    end = time.time()
    maps.append(sa)
    cost_time.append(end-start)
    print('Occlusion Done. Time:', start, end, end-start)

    # FeatureAblation 
    ablator = FeatureAblation(test_module.get_model())
    start = time.time()
    sa_map = []
    for i in range(len(mask)):
        sa = ablator.attribute(input, target=label, feature_mask=m[i](mask[i]).type(torch.int))[0]
        sa_map.append(np.array(sa))
    end = time.time()
    sa = torch.from_numpy(np.mean(np.array(sa_map), axis = 0))
    maps.append(sa)
    cost_time.append(end-start)
    print('FeatureAblation Done. Time:', start, end, end-start)

    '''
    # ShapleyValueSampling (Timeout for large image: ~0.5hr)
    shapleyvaluesampling = ShapleyValueSampling(test_module.get_model())
    start = time.time()
    sa_map = []
    for i in range(len(mask)):
        sa = shapleyvaluesampling.attribute(input, target=label, feature_mask=m[i](mask[i]).type(torch.int))[0]
        sa_map.append(np.array(sa))
    end = time.time()
    sa = torch.from_numpy(np.mean(np.array(sa_map), axis = 0))
    maps.append(sa)
    cost_time.append(end-start)
    print('ShapleyValueSampling Done. Time:', start, end, end-start)
    '''

    # Lime
    start = time.time()
    lime = Lime(test_module.get_model(), img, label)
    lime = torch.from_numpy(lime)
    lime.unsqueeze_(0)
    sa = lime
    end = time.time()
    maps.append(sa)
    cost_time.append(end-start)
    print('Lime Done. Time:', start, end, end-start)

    return maps, cost_time



# In[12]:
def canny(img):
    disp_img = np.array(255*img).astype(np.uint8)
    disp_img = cv2.cvtColor(disp_img, cv2.COLOR_BGR2GRAY)
    disp_img = cv2.Canny(disp_img,10,40)
    return disp_img

def get_salienct_points(g):
    points = []
    w = g.shape[0]
    h = g.shape[1]
    for i in range(w):
        for j in range(h):
            if g[i][j] > 0:
                points.append([[i, j]])
    return np.array(points, np.float32)


# In[13]:
def IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    if xA >= xB or yA >= yB:
        return 0

    # compute the area of intersection rectangle
    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# In[14]:
def pred(img):
    input = np.array(img).astype(np.float32)
    input = torch.Tensor(input)
    input.unsqueeze_(0)
    pred = test_module.prediction(input)
    return pred
    #return torch.argmax(pred).item()


# In[15]:
import matplotlib.patches as patches
'''
for i in range(num_target):

    trigger = test_module.get_trigger(i)
    disp_img = test_module.get_disp_img(trigger)

    trigger = normalize(np.array(disp_img))
    trigger = canny(trigger)

    pts = get_salienct_points(trigger)

    x2, y2, w2, h2 = cv2.boundingRect(pts)

    # Create figure and axes
    fig,ax = plt.subplots(1)

    # Display the image
    ax.imshow(np.array(disp_img))

    #plt.axis('off')
    ax.axis('off')

    # Create a Rectangle patch
    rect = patches.Rectangle((y2,x2),h2,w2,linewidth=1,edgecolor='r',facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    #plt.show()

    plt.savefig('./ImageNet/experiments/triggers/bb/'+str(trigger_size)+'_'+model_name+'_'+exp_name+'_'+str(i)+'.png')
'''


# In[16]:
'''
thr = [0.5,0.8,0.9,0.75,0.9,0.9,0.9,0.9]

IOU_L = []
for i in range(num_target):
    trigger = test_module.get_trigger(i)
    disp_img = test_module.get_disp_img(trigger)

    l, t = get_saliency_maps(trigger, i)

    iou = []

    fig, axs = plt.subplots(1, 9, constrained_layout=False, figsize=(18, 2))
    if test_module.config.channel == 3:
        axs[0].imshow(disp_img.cpu().numpy())
    else:
        axs[0].imshow(disp_img[0].cpu().numpy(), cmap='gray')

    axs[0].xaxis.set_ticks([])
    axs[0].yaxis.set_ticks([])

    trigger = normalize(np.array(disp_img))
    trigger = canny(trigger)

    pts = get_salienct_points(trigger)

    if len(pts) == 0:
        bb_trigger = [-1,-1,-1,-1]
    else:
        x2, y2, w2, h2 = cv2.boundingRect(pts)
        bb_trigger = [x2,y2,x2+w2,y2+h2]

        # Create a Rectangle patch
        rect = patches.Rectangle((y2,x2),h2,w2,linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        axs[0].add_patch(rect)

    for j in range(len(l)):

        g = normalize(l[j].detach().numpy())
        if g.shape[0] == 1:
            g = g.repeat(3, 0)

        g = np.transpose(g, (1,2,0))
        im = (g>thr[j]).astype(np.float32)
        im = canny(im)

        axs[j+1].imshow(g, cmap='RdBu_r')

        pts = get_salienct_points(im)
        if len(pts) == 0:
            bb = [-1,-1,-1,-1]
        else:
            x2, y2, w2, h2 = cv2.boundingRect(pts)
            bb = [x2,y2,x2+w2,y2+h2]

            # Create a Rectangle patch
            rect = patches.Rectangle((y2,x2),h2,w2,linewidth=1,edgecolor='r',facecolor='none')

            # Add the patch to the Axes
            axs[j+1].add_patch(rect)

        iou_v = IOU(bb_trigger, bb)
        iou.append(iou_v)

        axs[j+1].set_xlabel("{:.4f}".format(iou_v))

        axs[j+1].xaxis.set_ticks([])
        axs[j+1].yaxis.set_ticks([])

    fig.tight_layout()

    #plt.show()
    plt.savefig('./ImageNet/experiments/saliency_maps/triggers/'+str(trigger_size)+'_'+model_name+'_'+exp_name+'_'+str(i)+'.png')

    IOU_L.append(np.array(iou))
'''


# In[17]:


from torch import nn

criterion = nn.MSELoss()

def loss(pred1, pred2):
    pred1 = F.softmax(pred1)
    pred2 = F.softmax(pred2)
    return criterion(pred1, pred2)

def imgDiff(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)

    x_min = img1.min()
    x_max = img2.max()

    if x_max == x_min:
        img1 = img1-x_min
        img2 = img2-x_min
    else:
        img1 = (img1-x_min)/(x_max-x_min)
        img2 = (img2-x_min)/(x_max-x_min)

    img1 = np.transpose(img1, (1,2,0))
    img2 = np.transpose(img2, (1,2,0))

    img1 = (255*np.array(img1)).astype(np.uint8)
    img2 = (255*np.array(img2)).astype(np.uint8)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    return np.sum(img1 != img2)/(224*224)


# In[18]:
thr = [0.5,0.8,0.9,0.75,0.9,0.9,0.9,0.9]

import os
folder = './out/saliency_maps/'+model_name+'_'+exp_name+'_'+str(trigger_size)
if not os.path.exists(folder):
    os.makedirs(folder)

CT_L = []
IOU_L = []
TIME_L = []
DIFF_L = []
for i in range(len(L)):
    img_orig = L[i][0]
    img_trig = L[i][1]
    label_orig = L[i][2]
    label_trig = L[i][3]
    trigger = L[i][4]
    img_id = L[i][5]

    fig, axs = plt.subplots(2, 8, constrained_layout=False, figsize=(16, 4))

    # Display trojaned image
    disp_img = test_module.get_disp_img(img_trig)

    if test_module.config.channel == 3:
        axs[0][0].imshow(disp_img.cpu().numpy())
    else:
        axs[0][0].imshow(disp_img[0].cpu().numpy(), cmap='gray')

    axs[0][0].set_xlabel('pred:'+str(label_trig))
    axs[0][0].xaxis.set_ticks([])
    axs[0][0].yaxis.set_ticks([])

    # Get the bounding box of the trigger
    disp_img = test_module.get_disp_img(trigger)

    trigger = normalize(np.array(disp_img))
    trigger = canny(trigger)

    pts = get_salienct_points(trigger)

    if len(pts) == 0:
        bb_trigger = [-1,-1,-1,-1]
    else:
        x2, y2, w2, h2 = cv2.boundingRect(pts)
        bb_trigger = [x2,y2,x2+w2,y2+h2]

        # Create a Rectangle patch
        rect = patches.Rectangle((y2,x2),h2,w2,linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        axs[0][0].add_patch(rect)

    # Display the original image
    disp_img = test_module.get_disp_img(img_orig)

    if test_module.config.channel == 3:
        axs[1][0].imshow(disp_img.cpu().numpy())
    else:
        axs[1][0].imshow(disp_img[0].cpu().numpy(), cmap='gray')

    axs[1][0].set_xlabel('pred:'+str(label_orig))
    axs[1][0].xaxis.set_ticks([])
    axs[1][0].yaxis.set_ticks([])

    # Get the saliency maps
    l, t = get_saliency_maps(img_trig, label_trig)

    iou = []
    ct = []
    diff = []
    for j in range(len(l)):

        g = normalize(l[j].detach().numpy())
        if g.shape[0] == 1:
            g = g.repeat(3, 0)
        g = np.transpose(g, (1,2,0))

        # Display the saliency map
        axs[0][j+1].imshow(g, cmap='RdBu_r')

        # Display the detected trigger bounding box
        im = (g>thr[j]).astype(np.float32)
        im = canny(im)
        pts = get_salienct_points(im)
        if len(pts) == 0:
            bb = [-1,-1,-1,-1]
        else:
            x2, y2, w2, h2 = cv2.boundingRect(pts)
            bb = [x2,y2,x2+w2,y2+h2]

            # Create a Rectangle patch
            rect = patches.Rectangle((y2,x2),h2,w2,linewidth=1,edgecolor='r',facecolor='none')

            # Add the patch to the Axes
            axs[0][j+1].add_patch(rect)

        # Compute IOU
        iou_v = IOU(bb_trigger, bb)
        iou.append(iou_v)

        axs[0][j+1].set_xlabel('iou:'+"{:.4f}".format(iou_v))
        axs[0][j+1].xaxis.set_ticks([])
        axs[0][j+1].yaxis.set_ticks([])

        # Recover Image
        mask = np.zeros((224,224)).astype(np.float32)
        mask[x2:x2+w2, y2:y2+h2] = 1
        img_recover = img_orig * mask + img_trig * (1-mask)

        pred_orig = pred(img_orig)
        pred_recover = pred(img_recover)

        if torch.argmax(pred_orig) == torch.argmax(pred_recover):
            ct.append(1)
        else:
            ct.append(0)
        l0 = imgDiff(img_orig,img_recover)
        diff.append(l0)

        # Display the recovered image
        disp_img = test_module.get_disp_img(img_recover)

        axs[1][j+1].imshow(disp_img, cmap='RdBu_r')

        axs[1][j+1].set_xlabel('pred:'+str(torch.argmax(pred_recover).item())+' '+'diff:'+"%.4f" % l0)

        axs[1][j+1].xaxis.set_ticks([])
        axs[1][j+1].yaxis.set_ticks([])

    fig.tight_layout()

    #plt.show()
    plt.savefig(folder+'/'+str(img_id)+'.png')

    CT_L.append(np.array(ct))
    IOU_L.append(np.array(iou))
    TIME_L.append(np.array(t))
    DIFF_L.append(np.array(diff))


# In[22]:


def save(L, var):
    tmp = np.array(L)
    var_avg = np.mean(tmp, axis=0)
    #var_std = np.std(tmp, axis=0)

    with open("./out/"+var+"_mean.txt", "a") as myfile:
        myfile.write(str(trigger_size)+','+model_name+','+exp_name+','+','.join([str(x) for x in var_avg])+'\n')

    #with open(var+"_std.txt", "a") as myfile:
        #myfile.write(str(trigger_size)+','+model_name+','+exp_name+','+','.join([str(x) for x in var_std])+'\n')

save(CT_L, 'rr')
save(IOU_L, 'iou')
save(TIME_L, 'cc')
save(DIFF_L, 'rd')

