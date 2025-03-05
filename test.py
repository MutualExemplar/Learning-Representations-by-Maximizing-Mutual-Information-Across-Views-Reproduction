import argparse
import glob
import os

import imageio
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import mean_absolute_error
from torch.autograd import Variable
from torchvision import transforms

from data import image_loader
from utils import get_logger, create_dir
from model.mutual_exemplar_unet import MutualExemplarUNet
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
parser.add_argument('--trainsize', type=int, default=(512,288), help='training dataset size')
parser.add_argument('--dataset', type=str, default='kvasir', help='dataset name')
parser.add_argument('--threshold', type=float, default=0.5, help='threshold')
opt = parser.parse_args()


class Test(object):
    def __init__(self):
        self._init_logger()
        self.model_1 = MutualExemplarUNet()
        self.model_2 = MutualExemplarUNet()
        self.model_3 = MutualExemplarUNet()

    
    def _init_logger(self):

        log_dir = 'logs/' + opt.dataset + '/test'
        output_dir = os.path.join(log_dir, 'outputs')
        create_dir(output_dir)

        self.logger = get_logger(output_dir)
        print('RUNDIR:', output_dir)

        self.save_path = log_dir
        self.image_save_path_1 = log_dir + "/saved_images_1"
        create_dir(self.image_save_path_1)
        self.image_save_path_2 = log_dir + "/saved_images_2"
        create_dir(self.image_save_path_2)
        self.image_save_path_3 = log_dir + "/saved_images_3"
        create_dir(self.image_save_path_3)

        self.model_1_load_path = 'logs/' + opt.dataset + '/train/Checkpoints_5%/Model_1.pth'
        self.model_2_load_path = 'logs/' + opt.dataset + '/train/Checkpoints_5%/Model_2.pth'
        self.model_3_load_path = 'logs/' + opt.dataset + '/train/Checkpoints_5%/Model_3.pth'


    def visualize_val_input(self, var_map, i):
        count = i
        im = transforms.ToPILImage()(var_map.squeeze_(0).detach().cpu()).convert("RGB")
        name = '{:02d}_input.png'.format(count)
        imageio.imwrite(self.image_save_path_1 + "/val_" + name, im)

    def visualize_gt(self, var_map, i):
        count = i
        for kk in range(var_map.shape[0]):
            pred_edge_kk = var_map[kk, :, :, :]
            pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
            pred_edge_kk *= 255.0
            pred_edge_kk = pred_edge_kk.astype(np.uint8)
            name = '{:02d}_gt.png'.format(count)
            imageio.imwrite(self.image_save_path_1 + "/val_" + name, pred_edge_kk)
            imageio.imwrite(self.image_save_path_2 + "/val_" + name, pred_edge_kk)
            imageio.imwrite(self.image_save_path_3 + "/val_" + name, pred_edge_kk)

    def visualize_prediction1(self, var_map, i):
        count = i
        for kk in range(var_map.shape[0]):
            pred_edge_kk = var_map[kk, :, :, :]
            pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
            pred_edge_kk *= 255.0
            pred_edge_kk = pred_edge_kk.astype(np.uint8)
            name = '{:02d}_pred_1.png'.format(count)
            imageio.imwrite(self.image_save_path_1 + "/val_" + name, pred_edge_kk)

    def visualize_prediction2(self, var_map, i):
        count = i
        for kk in range(var_map.shape[0]):
            pred_edge_kk = var_map[kk, :, :, :]
            pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
            pred_edge_kk *= 255.0
            pred_edge_kk = pred_edge_kk.astype(np.uint8)
            name = '{:02d}_pred_2.png'.format(count)
            imageio.imwrite(self.image_save_path_2 + "/val_" + name, pred_edge_kk)
            
    def visualize_prediction3(self, var_map, i):
        count = i
        for kk in range(var_map.shape[0]):
            pred_edge_kk = var_map[kk, :, :, :]
            pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
            pred_edge_kk *= 255.0
            pred_edge_kk = pred_edge_kk.astype(np.uint8)
            name = '{:02d}_pred_3.png'.format(count)
            imageio.imwrite(self.image_save_path_3 + "/val_" + name, pred_edge_kk)

    def visualize_uncertainity(self, var_map, i):
        count = i
        for kk in range(var_map.shape[0]):
            pred_edge_kk = var_map[kk, :, :, :]
            pred_edge_kk = pred_edge_kk.detach().cpu().numpy().squeeze()
            pred_edge_kk *= 255.0
            pred_edge_kk = pred_edge_kk.astype(np.uint8)
            name = '{:02d}_pred.png'.format(count)
            imageio.imwrite(self.image_save_path_1 + "/uncertainity_" + name, pred_edge_kk)

    def evaluate_model_1(self, image_dir):

        target_list = np.array([])
        output_list = np.array([])
        output_pred_list = np.array([])
        test_dir = image_dir
        self.logger.info(test_dir)

        pred_files = glob.glob(test_dir + 'val_*_pred_1.png')
        gt_files = glob.glob(test_dir + 'val_*_gt.png')

        for file in pred_files:
            image = Image.open(file)
            output = np.asarray(image)
            output = output.flatten() / 255
            output_pred = (output > opt.threshold)
            output_list = np.concatenate((output_list, output), axis=None)
            output_pred_list = np.concatenate((output_pred_list, output_pred), axis=None)

        for file in gt_files:
            image = Image.open(file)
            target = np.asarray(image)
            target = target.flatten() / 255
            target = (target > opt.threshold)
            target_list = np.concatenate((target_list, target), axis=None)

        intersection = np.sum(output_pred_list * target_list)
        DSC = (2. * intersection) / (np.sum(output_pred_list) + np.sum(target_list))
        
        MAE = mean_absolute_error(target_list, output_pred_list)
        
        self.logger.info("Model 1 DSC : {:.4f}".format(DSC))
        self.logger.info("Model 1 MAE : {:.4f}".format(MAE))
        
        return DSC, MAE
    
    def evaluate_model_2(self, image_dir):

        target_list = np.array([])
        output_list = np.array([])
        output_pred_list = np.array([])
        test_dir = image_dir
        self.logger.info(test_dir)

        pred_files = glob.glob(test_dir + 'val_*_pred_2.png')
        gt_files = glob.glob(test_dir + 'val_*_gt.png')

        for file in pred_files:
            image = Image.open(file)
            output = np.asarray(image)
            output = output.flatten() / 255
            output_pred = (output > opt.threshold)
            output_list = np.concatenate((output_list, output), axis=None)
            output_pred_list = np.concatenate((output_pred_list, output_pred), axis=None)

        for file in gt_files:
            image = Image.open(file)
            target = np.asarray(image)
            target = target.flatten() / 255
            target = (target > opt.threshold)
            target_list = np.concatenate((target_list, target), axis=None)

        # Compute DSC and MAE
        intersection = np.sum(output_pred_list * target_list)
        DSC = (2. * intersection) / (np.sum(output_pred_list) + np.sum(target_list))
        MAE = mean_absolute_error(target_list, output_pred_list)

        self.logger.info("Model 2 DSC : {:.4f}".format(DSC))
        self.logger.info("Model 2 MAE : {:.4f}".format(MAE))

        return DSC, MAE

    def evaluate_model_3(self, image_dir):

        target_list = np.array([])
        output_list = np.array([])
        output_pred_list = np.array([])
        test_dir = image_dir
        self.logger.info(test_dir)

        pred_files = glob.glob(test_dir + 'val_*_pred_3.png')
        gt_files = glob.glob(test_dir + 'val_*_gt.png')

        for file in pred_files:
            image = Image.open(file)
            output = np.asarray(image)
            output = output.flatten() / 255
            output_pred = (output > opt.threshold)
            output_list = np.concatenate((output_list, output), axis=None)
            output_pred_list = np.concatenate((output_pred_list, output_pred), axis=None)

        for file in gt_files:
            image = Image.open(file)
            target = np.asarray(image)
            target = target.flatten() / 255
            target = (target > opt.threshold)
            target_list = np.concatenate((target_list, target), axis=None)

        # Compute DSC and MAE
        intersection = np.sum(output_pred_list * target_list)
        DSC = (2. * intersection) / (np.sum(output_pred_list) + np.sum(target_list))
        MAE = mean_absolute_error(target_list, output_pred_list)

        self.logger.info("Model 3 DSC : {:.4f}".format(DSC))
        self.logger.info("Model 3 MAE : {:.4f}".format(MAE))

        return DSC, MAE  
    

    def run(self):
        # Load models
        self.model_1.load_state_dict(torch.load(self.model_1_load_path))
        self.model_1.cuda()

        self.model_2.load_state_dict(torch.load(self.model_2_load_path))
        self.model_2.cuda()
        
        self.model_3.load_state_dict(torch.load(self.model_3_load_path))
        self.model_3.cuda()
        
        image_root = './data/'+ opt.dataset +'/train/image/'
        gt_root = './data/'+ opt.dataset +'/train/mask/'
        val_img_root = './data/'+ opt.dataset +'/test/image/'
        val_gt_root = './data/'+ opt.dataset +'/test/mask/'

        _, _, val_loader = image_loader(image_root, gt_root, val_img_root, val_gt_root, opt.batchsize, opt.trainsize)

        for i, pack in enumerate(val_loader, start=1):
            with torch.no_grad():
                images, gts = pack
                images = Variable(images).cuda()
                gts = Variable(gts).cuda()

                feat_map_1, _ = self.model_1(images)
                prediction1 = torch.sigmoid(feat_map_1)

                feat_map_2, _ = self.model_2(images)
                prediction2 = torch.sigmoid(feat_map_2)

                feat_map_3, _ = self.model_3(images)
                prediction3 = torch.sigmoid(feat_map_3)

            self.visualize_val_input(images, i)
            self.visualize_gt(gts, i)
            self.visualize_prediction1(prediction1, i)
            self.visualize_prediction2(prediction2, i)
            self.visualize_prediction3(prediction3, i)

        # Evaluate each model
        DSC_1, MAE_1 = self.evaluate_model_1('logs/kvasir/test/saved_images_1/')
        DSC_2, MAE_2 = self.evaluate_model_2('logs/kvasir/test/saved_images_2/')
        DSC_3, MAE_3 = self.evaluate_model_3('logs/kvasir/test/saved_images_3/')

        # Calculate averages
        avg_DSC = (DSC_1 + DSC_2 + DSC_3) / 3
        avg_MAE = (MAE_1 + MAE_2 + MAE_3) / 3

        # Save final averages clearly in the logs/kvasir/test/outputs folder
        self.logger.info("Average DSC (3 models): {:.4f}".format(avg_DSC))
        self.logger.info("Average MAE (3 models): {:.4f}".format(avg_MAE))

if __name__ == '__main__':
    Test_network = Test()
    Test_network.run()
