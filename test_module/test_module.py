import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader
from test_module.grad_cam import BackPropagation, GuidedBackPropagation, GradCAM, occlusion_sensitivity
import gc
from GPUtil import showUtilization as gpu_usage
import cv2
import matplotlib.cm as cm

class TestModule():
    def __init__(self, name, config):
        self.name = name
        self.config = config

    def load_data(self, batch_size=64, target_label=0):
        self.config.load_test_data(batch_size, target_label)

    def get_model(self):
        model = self.config.get_model()
        return model

    def sigmoid(X):
        return 1/(1+np.exp(-X))

    def preprocess_image(self,img):
        preprocessed_img = img.copy()
        preprocessed_img = torch.from_numpy(preprocessed_img)
        preprocessed_img.unsqueeze_(0)
        return preprocessed_img.requires_grad_(True)

    def deprocess_image(self,img):
        means = self.config.means
        stds = self.config.stds

        deprocessed_img = img.clone()
        for i in range(self.config.channel):
            deprocessed_img[i] = deprocessed_img[i]*stds[i]
            deprocessed_img[i] = deprocessed_img[i]+means[i]

        if self.config.channel == 3:
            deprocessed_img = deprocessed_img.permute(1, 2, 0)
        else:
            deprocessed_img = deprocessed_img

        return deprocessed_img

    def prediction(self, img):
        model = self.get_model()

        img = img.to(self.config.device)

        input = img.cpu().numpy()
        input = self.preprocess_image(input)

        # Forward pass the data through the model
        '''
        if self.config.channel == 3:
            output = model(input.to(self.config.device))
        else:
        '''
        output = model(input[0].to(self.config.device))

        # Calculate the probabilities
        return F.softmax(output)

    def get_disp_img(self,img):
        img = self.deprocess_image(img)
        return self.config.conv_img(img)

    def get_trigger(self,target):
        return self.config.get_trigger(target)


    def show_img(self,img,path=None):
        img = self.get_img(img)
        if self.config.channel == 3:
            plt.imshow(img.cpu().numpy())
        else:
            plt.imshow(img[0].cpu().numpy(), cmap='gray')

        plt.axis('off')

        if path is None:
            plt.show()
        else:
            plt.savefig(path)

    def show_mask(self,img):
        if self.config.channel == 3:
            deprocessed_img = (img.permute(1, 2, 0)).cpu().numpy()
            plt.imshow(deprocessed_img)
        else:
            deprocessed_img = img[0].cpu().numpy()
            plt.imshow(deprocessed_img, cmap='gray')
        plt.show()

    def get_sensitivity(self, maps):
        w = self.config.img_w
        h = self.config.img_h
        maps = maps.cpu().numpy()
        scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
        maps = maps / scale * 0.5
        maps += 0.5
        maps = cm.bwr_r(maps)[..., :3]
        maps = np.uint8(maps * 255.0)
        #maps = cv2.resize(maps, (w, h), interpolation=cv2.INTER_NEAREST)
        return maps


    def get_smooth_gradients(self, img, target_label, method, batch_size = 16, target_layer='features'):
        batch_img_shape = (batch_size, self.config.channel, self.config.img_w, self.config.img_h)

        channel = self.config.channel
        device = self.config.device

        model = self.get_model()

        img = img.clone()
        img.unsqueeze_(0)
        img = img.repeat(batch_size, 1, 1, 1)
        img = img.to(device)

        label = torch.LongTensor([target_label])
        label.unsqueeze_(0)
        label = label.repeat(batch_size, 1)
        label = label.to(device)

        dx = (np.random.normal(0, 1, batch_img_shape)*0.001).astype(np.float32)
        dx = torch.Tensor(dx).to(device)

        # Add noises (SmothGrad)
        input = img + dx

        if method == 'bp':
            bp = BackPropagation(model=model)
            bp.forward(input)  # sorted
            bp.backward(ids=label)
            gradients = bp.generate()

        if method == 'gbp' or method == 'ggcam':
            gbp = GuidedBackPropagation(model=model)
            gbp.forward(input)
            gbp.backward(ids=label)
            gradients = gbp.generate()

        if method == 'gcam' or method == 'ggcam':
            gcam = GradCAM(model=model, candidate_layers=[target_layer])
            gcam.forward(input)
            gcam.backward(ids=label)
            regions = gcam.generate(target_layer=target_layer)

        if method == 'ggcam':
            gradients=torch.mul(regions, gradients)

        if method == 'gcam':
            saliency = regions
        elif method == 'occ':
            sensitivity = occlusion_sensitivity(model, input, label, patch=10)
            saliency = self.get_sensitivity(sensitivity)
        else:
            gradients = torch.mean(gradients, dim=0)

            gradients -= gradients.min()

            if gradients.max() > 0:
                gradients /= gradients.max()

            saliency = gradients

        torch.cuda.empty_cache()

        #gpu_usage()

        return saliency

    def view_bp(self, img, target_label, batch_size = 64):
        gradients = self.smooth_gradients(img, target_label, 'bp', batch_size)
        self.show_mask(gradients)

    def view_gbp(self, img, target_label, batch_size = 64):
        gradients = self.smooth_gradients(img, target_label, 'gbp', batch_size)
        self.show_mask(gradients)

    def view_ggcam(self, img, target_label, batch_size = 64):
        gradients = self.smooth_gradients(img, target_label, 'ggcam', batch_size)
        self.show_mask(gradients)

    def get_gradients(self, model, images, target_label, method):
        device = self.config.device

        label = torch.LongTensor([target_label])
        label.unsqueeze_(0)
        label = label.repeat(len(images), 1)
        label = label.to(device)

        if method == 'bp':
            bp = BackPropagation(model=model)
            bp.forward(images)  # sorted
            bp.backward(ids=label)
            gradients = bp.generate()

        if method == 'gbp' or method == 'ggcam':
            gbp = GuidedBackPropagation(model=model)
            gbp.forward(images)
            gbp.backward(ids=label)
            gradients = gbp.generate()

        if method == 'ggcam':
            gcam = GradCAM(model=model, candidate_layers=['conv5_3'])
            gcam.forward(images)
            gcam.backward(ids=label)
            regions = gcam.generate(target_layer='conv5_3')
            gradients=torch.mul(regions, gradients)

        mean_grads = torch.mean(gradients, dim=0)

        torch.cuda.empty_cache()

        #gpu_usage()

        return mean_grads


    # Get mask using Guided Backpropagation
    def get_mask(self, test_loader, target_label, method, mode='data', batch_size = 1):
        device = self.config.device
        batch_img_shape = (batch_size, self.config.channel, self.config.img_w, self.config.img_h)

        model = self.get_model()
        x = torch.zeros([self.config.channel, self.config.img_w, self.config.img_h], dtype=torch.float32)
        x = x.to(device)
        ct = 0
        mean = [129.1863, 104.7624, 93.5940]
        if mode == 'data':
            for i, data in enumerate(test_loader):
                img, labels = data
                grads = self.get_gradients(model, img, target_label, method)
                if grads is not None:
                    x = x.add(grads)
                    ct = ct + 1
        elif mode == 'random':
            #from pympler.tracker import SummaryTracker
            for i in range(100):
                print(i)
                #img = np.random.normal(0, 1, batch_img_shape).astype(np.float32)
                img = (np.random.random(batch_img_shape)*255).astype(np.float32)
                for j in range(3):
                    img[:,j,:,:] -= mean[j]
                img = torch.Tensor(img).to(device)
                grads = self.get_gradients(model, img, target_label, method)
                if grads is not None:
                    x = x.add(grads)
                    #x = x + grads.cpu().numpy()
                    ct = ct + 1
                del img, grads

        elif mode == 'noise':
            for i in range(500):
                img = (np.random.normal(0, 1, batch_img_shape)*0.001).astype(np.float32)
                img = torch.Tensor(img).to(device)
                grads = self.get_gradients(model, img, target_label, method)
                if grads is not None:
                    x = x.add(grads)
                    ct = ct + 1

        mask = x / ct

        '''
        mask -= mask.min()

        mask = 1/(1+np.exp(-mask.cpu().numpy()))
        #mask = sigmoid(mask.cpu().numpy())                  # Recale using sigmoid
        mask = torch.Tensor(mask).to(device)

        mask = torch.clamp(mask-0.5, 0, 1)     # Keep positive gradients
        '''


        mask -= mask.min()
        if mask.max() > 0:
            mask /= mask.max()

        '''
        if len(L) > 0:
            mask = torch.mean(torch.stack(L), dim=0)
            mask -= mask.min()
            if mask.max() > 0:
                mask /= mask.max()
        else:
            mask = np.zeros(self.config.img_shape).astype(np.float32)
        '''

        return mask


    # Show saliency maps of a set of data with specified saliency method
    def show_saliency_maps_1(self, method, dataset, path = None):
        cols = ['{}'.format(col) for col in range(0, 10)]
        rows = ['{}'.format(row) for row in range(0, 10)]
        ids = self.config.ids

        channel = self.config.channel

        fig, axs = plt.subplots(10, 10, constrained_layout=False, figsize=(50, 50))
        for i in range(10):
            for j in range(10):
                g = self.get_smooth_gradients(dataset[ids[i]][0], j, method)

                if channel == 3:
                    img = g.permute(1,2,0).cpu().numpy()
                    axs[i][j].imshow(img)
                else:
                    img = g[0].cpu().numpy()
                    axs[i][j].imshow(img, cmap='gray')
                axs[i][j].set_xticklabels('')
                axs[i][j].set_yticklabels('')

        for ax, col in zip(axs[0], cols):
            ax.set_xlabel(col, rotation=0, fontsize=40)
            ax.xaxis.set_label_coords(0.5,1.2)
            ax.set_xticklabels('')

        for ax, row in zip(axs[:,0], rows):
            ax.set_ylabel(row, rotation=0, fontsize=40)
            ax.yaxis.set_label_coords(-0.1,0.5)
            ax.set_yticklabels('')

        fig.tight_layout()

        if path is None:
            plt.show()
        else:
            plt.savefig(path, dpi=300)

    # Show saliency maps of a set of images using the backpropagation method
    def show_saliency_maps_2(self, dataset, target_label, path = None):
        cols = ['{}'.format(col) for col in range(0, 10)]
        rows = ['{}'.format(row) for row in ['Input Image',
                                             'Predictions',
                                             'Saliency Map']]

        cols.append('random')
        cols.append('empty')

        channel = self.config.channel
        ids = self.config.ids
        device = self.config.device

        images = []
        for j in range(10):
            images.append(dataset[ids[j]][0])
        images.append(torch.Tensor(np.random.normal(0, 1, self.config.img_shape).astype(np.float32)).to(device))
        images.append(torch.Tensor(np.zeros(self.config.img_shape).astype(np.float32)).to(device))

        fig, axs = plt.subplots(3, 12, constrained_layout=False, figsize=(65, 15))

        # Plot original image
        for j in range(len(images)):
            if j < 10:
                img = images[j].cpu().numpy().astype(np.long)
                img = torch.LongTensor(img)
                img = self.deprocess_image(img)
            else:
                img = images[j].permute(1,2,0)
            if channel == 3:
                axs[0][j].imshow(img.cpu().numpy())
            else:
                axs[0][j].imshow(img[0].cpu().numpy(), cmap='gray')
            axs[0][j].set_xticklabels('')
            axs[0][j].set_yticklabels('')

        # Plot predictions
        pred = self.prediction(torch.stack(images))
        x_pos = np.arange(10)
        print(torch.argmax(pred))

        '''
        for j in range(len(images)):
            axs[1][j].bar(x_pos, pred[j].cpu().detach().numpy(), align='center', alpha=0.5, ecolor='black', capsize=10)
            axs[1][j].set_ylim(0,1)
            axs[1][j].set_xticks(x_pos)
            axs[1][j].set_xticklabels('{}'.format(col) for col in range(0, 10))
            axs[1][j].set_ylabel('Probability')
        '''

        # Plot saliency map by backpropagation
        for j in range(len(images)):
            gradients = self.get_smooth_gradients(images[j].clone(), target_label, 'bp', 16)
            if channel == 3:
                img = gradients.permute(1,2,0).cpu().numpy()
                axs[2][j].imshow(img)
            else:
                img = gradients[0].cpu().numpy()
                axs[2][j].imshow(img, cmap='gray')
            axs[2][j].set_xticklabels('')
            axs[2][j].set_yticklabels('')

        for ax, col in zip(axs[0], cols):
            ax.set_xlabel(col, rotation=0, fontsize=60)
            ax.xaxis.set_label_coords(0.5,1.2)
            ax.set_xticklabels('')

        for ax, row in zip(axs[:,0], rows):
            ax.set_ylabel(row, rotation=0, fontsize=60)
            ax.yaxis.set_label_coords(-1.0,0.5)
            ax.set_yticklabels('')

        fig.tight_layout()

        if path is None:
            plt.show()
        else:
            plt.savefig(path, dpi=300)

    # Show the masks generated for different target labels
    def show_masks_2(self, path = None):
        cols = ['{}'.format(col) for col in range(0, 10)]
        rows = ['{}'.format(row) for row in ['Clean''\n''Images', 'Random''\n''Images', 'Noise''\n''Images']]
        modes = ['{}'.format(mode) for mode in ['data', 'random', 'noise']]

        channel = self.config.channel
        ids = self.config.ids
        device = self.config.device
        dataset = self.config.test_data_orig

        fig, axs = plt.subplots(3, 10, constrained_layout=False, figsize=(55, 15))

        for i in range(len(modes)):
            for j in range(10):
                m = self.get_mask(self.config.test_data_orig_loader, j, 'bp', mode=modes[i])

                if channel == 3:
                    img = m.permute(1,2,0).cpu().numpy()
                    axs[i][j].imshow(img, cmap='gray')
                else:
                    img = m[0].cpu().numpy()
                    axs[i][j].imshow(img, cmap='gray')
                axs[i][j].set_xticklabels('')
                axs[i][j].set_yticklabels('')

        for ax, col in zip(axs[0], cols):
            ax.set_xlabel(col, rotation=0, fontsize=40)
            ax.xaxis.set_label_coords(0.5,1.2)
            ax.set_xticklabels('')

        for ax, row in zip(axs[:,0], rows):
            ax.set_ylabel(row, rotation=0, fontsize=40)
            ax.yaxis.set_label_coords(-0.5,0.5)
            ax.set_yticklabels('')

        fig.tight_layout()

        if path is None:
            plt.show()
        else:
            plt.savefig(path, dpi=300, bbox_inches='tight')

    def show_saliency_map(self, image, path=None):
        cols = ['{}'.format(col) for col in range(0, 10)]
        rows = ['{}'.format(row) for row in ['Backpropagation', 'Guided' '\n' 'Backprogagation', 'Guided' '\n' 'Grad-CAM']]

        channel = self.config.channel
        model = self.get_model()

        fig, axs = plt.subplots(3, 10, constrained_layout=False, figsize=(50, 15))
        for i in range(10):
            bp = self.get_smooth_gradients(image.clone(), i, 'bp')
            if channel == 3:
                img = bp.permute(1,2,0).cpu().numpy()
                axs[0][i].imshow(img)
            else:
                img = bp[0].cpu().numpy()
                axs[0][i].imshow(img, cmap='gray')
            axs[0][i].set_xticklabels('')
            axs[0][i].set_yticklabels('')

        for i in range(10):
            gbp = self.get_smooth_gradients(image.clone(), i, 'gbp')
            if channel == 3:
                img = bp.permute(1,2,0).cpu().numpy()
                axs[1][i].imshow(img)
            else:
                img = bp[0].cpu().numpy()
                axs[1][i].imshow(img, cmap='gray')
            axs[1][i].set_xticklabels('')
            axs[1][i].set_yticklabels('')

        for i in range(10):
            ggcam = self.get_smooth_gradients(image.clone(), i, 'ggcam')
            if channel == 3:
                img = bp.permute(1,2,0).cpu().numpy()
                axs[2][i].imshow(img)
            else:
                img = bp[0].cpu().numpy()
                axs[2][i].imshow(img, cmap='gray')
            axs[2][i].set_xticklabels('')
            axs[2][i].set_yticklabels('')

        for ax, col in zip(axs[0], cols):
            ax.set_xlabel(col, rotation=0, fontsize=40)
            ax.xaxis.set_label_coords(0.5,1.2)
            ax.set_xticklabels('')

        for ax, row in zip(axs[:,0], rows):
            ax.set_ylabel(row, rotation=0, fontsize=40)
            ax.yaxis.set_label_coords(-1.0,0.5)
            ax.set_yticklabels('')

        fig.tight_layout()

        if path is None:
            plt.show()
        else:
            plt.savefig(path, dpi=300)

    # Show the masks generated by different dataset and methods
    def show_masks_1(self, target_label, path):
        cols = ['{}'.format(col) for col in ['Clean Images', 'Random Images', 'Noise Images']]
        rows = ['{}'.format(row) for row in ['Backpropagation',
                                             'Guided' '\n' 'Backprogagation',
                                             'Guided' '\n' 'Grad-CAM']]
        modes = ['{}'.format(mode) for mode in ['data', 'random', 'noise']]

        channel = self.config.channel

        fig, axs = plt.subplots(3, 3, constrained_layout=False, figsize=(7, 5))
        for i in range(3):
            m = self.get_mask(self.config.test_data_orig_loader, target_label, 'bp', mode=modes[i])

            if channel == 3:
                img = m.permute(1,2,0).cpu().numpy()
                axs[0][i].imshow(img)
            else:
                img = m[0].cpu().numpy()
                axs[0][i].imshow(img, cmap='gray')
            axs[0][i].set_xticklabels('')
            axs[0][i].set_yticklabels('')

        for i in range(3):
            m = self.get_mask(self.config.test_data_orig_loader, target_label, 'gbp', mode=modes[i])
            if channel == 3:
                img = m.permute(1,2,0).cpu().numpy()
                axs[1][i].imshow(img)
            else:
                img = m[0].cpu().numpy()
                axs[1][i].imshow(img, cmap='gray')
            axs[1][i].set_xticklabels('')
            axs[1][i].set_yticklabels('')

        for i in range(3):
            m = self.get_mask(self.config.test_data_orig_loader, target_label, 'ggcam', mode=modes[i])
            if channel == 3:
                img = m.permute(1,2,0).cpu().numpy()
                axs[2][i].imshow(img)
            else:
                img = m[0].cpu().numpy()
                axs[2][i].imshow(img, cmap='gray')
            axs[2][i].set_xticklabels('')
            axs[2][i].set_yticklabels('')

        for ax, col in zip(axs[0], cols):
            ax.set_xlabel(col, rotation=0, fontsize=10)
            ax.xaxis.set_label_coords(0.5,1.2)
            ax.set_xticklabels('')

        for ax, row in zip(axs[:,0], rows):
            ax.set_ylabel(row, rotation=0, fontsize=10)
            ax.yaxis.set_label_coords(-0.55,0.5)
            ax.set_yticklabels('')

        fig.tight_layout()

        if path is None:
            plt.show()
        else:
            plt.savefig(path, dpi=300)

    # Show the masks generated for different target labels using different dataset
    def show_masks_2(self, path = None):
        cols = ['{}'.format(col) for col in range(0, 10)]
        rows = ['{}'.format(row) for row in ['Clean Images', 'Random Images', 'Noise Images']]
        modes = ['{}'.format(mode) for mode in ['data', 'random', 'noise']]

        channel = self.config.channel
        ids = self.config.ids
        device = self.config.device
        dataset = self.config.test_data_orig

        fig, axs = plt.subplots(3, 10, constrained_layout=False, figsize=(65, 18))

        for i in range(len(modes)):
            for j in range(10):
                m = self.get_mask(self.config.test_data_orig_loader, j, 'bp', mode=modes[i])

                if channel == 3:
                    img = m.permute(1,2,0).cpu().numpy()
                    axs[i][j].imshow(img)
                else:
                    img = m[0].cpu().numpy()
                    axs[i][j].imshow(img, cmap='gray')
                axs[i][j].set_xticklabels('')
                axs[i][j].set_yticklabels('')

        for ax, col in zip(axs[0], cols):
            ax.set_xlabel(col, rotation=0, fontsize=60)
            ax.xaxis.set_label_coords(0.5,1.2)
            ax.set_xticklabels('')

        for ax, row in zip(axs[:,0], rows):
            ax.set_ylabel(row, rotation=0, fontsize=60)
            ax.yaxis.set_label_coords(-0.85,0.5)
            ax.set_yticklabels('')

        fig.tight_layout()

        if path is None:
            plt.show()
        else:
            plt.savefig(path, dpi=300)

    # Show the masks generated for different target labels by different method
    def show_masks_3(self, mode, path = None):
        cols = ['{}'.format(col) for col in range(0, 10)]
        rows = ['{}'.format(row) for row in ['Backpropagation',
                                             'Guided' '\n' 'Backprogagation',
                                             'Guided' '\n' 'Grad-CAM']]
        methods = ['{}'.format(method) for method in ['bp','gbp','ggcam']]

        channel = self.config.channel
        ids = self.config.ids
        device = self.config.device
        dataset = self.config.test_data_orig

        fig, axs = plt.subplots(3, 10, constrained_layout=False, figsize=(65, 18))

        for i in range(len(rows)):
            for j in range(10):
                m = self.get_mask(self.config.test_data_orig_loader, j, methods[i], mode=mode)

                if channel == 3:
                    img = m.permute(1,2,0).cpu().numpy()
                    axs[i][j].imshow(img)
                else:
                    img = m[0].cpu().numpy()
                    axs[i][j].imshow(img, cmap='gray')
                axs[i][j].set_xticklabels('')
                axs[i][j].set_yticklabels('')

        for ax, col in zip(axs[0], cols):
            ax.set_xlabel(col, rotation=0, fontsize=60)
            ax.xaxis.set_label_coords(0.5,1.2)
            ax.set_xticklabels('')

        for ax, row in zip(axs[:,0], rows):
            ax.set_ylabel(row, rotation=0, fontsize=60)
            ax.yaxis.set_label_coords(-0.85,0.5)
            ax.set_yticklabels('')

        fig.tight_layout()

        if path is None:
            plt.show()
        else:
            plt.savefig(path, dpi=300)

    def TEST_SHOW_BP_FOR_ORIG_IMGS(self,path=None):
        self.show_saliency_maps_1('bp', self.config.test_data_orig, path)

    def TEST_SHOW_BP_FOR_TRIG_IMGS(self,path=None):
        self.show_saliency_maps_1('bp', self.config.test_data_trig, path)

    def TEST_SHOW_GBP_FOR_ORIG_IMGS(self,path=None):
        self.show_saliency_maps_1('gbp', self.config.test_data_orig, path)

    def TEST_SHOW_GBP_FOR_TRIG_IMGS(self,path=None):
        self.show_saliency_maps_1('gbp', self.config.test_data_trig, path)

    def TEST_SHOW_GGCAM_FOR_ORIG_IMGS(self,path=None):
        self.show_saliency_maps_1('ggcam', self.config.test_data_orig, path)

    def TEST_SHOW_GGCAM_FOR_TRIG_IMGS(self,path=None):
        self.show_saliency_maps_1('ggcam', self.config.test_data_trig, path)

    def TEST_SHOW_SALIENCY_FOR_EMPTY_IMG(self,path=None):
        self.show_saliency_map(torch.Tensor(np.zeros(self.config.img_shape).astype(np.float32)),path)

    def TEST_SHOW_SALIENCY_FOR_RANDOM_IMG(self,path=None):
        batch_img_shape = (self.config.channel, self.config.img_w, self.config.img_h)
        img = (np.random.random(batch_img_shape)*255).astype(np.float32)
        for j in range(3):
            img[j,:,:] -= self.config.means[j]
        img = torch.Tensor(img).to(self.config.device)
        self.show_saliency_map(img, path)
        #self.show_saliency_map(torch.Tensor(np.random.normal(0, 1, self.config.img_shape).astype(np.float32)), path)

    def TEST_SHOW_SALIENCY_MAPS_FOR_ORIG_IMGS(self,path=None):
        self.show_saliency_maps_2(self.config.test_data_orig, 0, path)

    def TEST_SHOW_SALIENCY_MAPS_FOR_TRIG_IMGS(self,path=None):
        self.show_saliency_maps_2(self.config.test_data_trig, 0, path)

    def TEST_SHOW_MASKS_BY_TARGET(self, target, path=None):
        self.show_masks_1(target, path)

    def TEST_SHOW_MASKS_FOR_DIFFERENT_TARGETS(self,path=None):
        self.show_masks_2(path)

    def TEST_SHOW_MASKS_BY_METHOD(self, mode, path=None):
        self.show_masks_3(mode, path)

    def TEST_GET_MASKS_BY_TARGET_METHOD_DATA(self, target_label, method, mode, path=None):
        m = self.get_mask(self.config.test_data_orig_loader, target_label, method, mode)

        if self.config.channel == 3:
            img = m.permute(1,2,0).cpu().numpy()
            plt.imshow(img)
        else:
            img = m[0].cpu().numpy()
            plt.imshow(img, cmap='gray')
        plt.savefig(path)



