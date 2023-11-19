# from eval_combined import main

from timm.loss import LabelSmoothingCrossEntropy
from utils import *
from models import *
from torch.utils.tensorboard import SummaryWriter
from skimage import transform


import pathlib as pl
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import torchvision.transforms as transforms
import timm
import torch
import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os
import os.path as op
import cv2
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
import logging
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
import torchvision.transforms as transforms
import timm
import torch
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
from timm.optim.optim_factory import create_optimizer
from types import SimpleNamespace
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
import numpy as np
from timm.models.vision_transformer import VisionTransformer
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch.distributed as dist
import matplotlib.pyplot as plt

def get_args_parser():
    parser = argparse.ArgumentParser('EG-ViT', add_help=False)

    parser.add_argument('--model_version', type=int, default=1)
    parser.add_argument('--res_layer', type=int, default=11)

    parser.add_argument('--forward_with', default="gaze")  # grad   gaze
    parser.add_argument('--mask_G_or_S', default="S")  # G :use mask_filter  S: no use filter
    parser.add_argument('--cross_loss_para', default='sum')  # mean   sum
    parser.add_argument('--gaze_input_norm', default=True, help='weather Norm gaze hm')  # False   True
    parser.add_argument('--mask_mse_loss_weight', type=float, default=100.0)  # 1e-6   1e-2  1.0
    parser.add_argument('--gaze_mask_num', type=int, default=64)
    parser.add_argument('--random_use_mask', type=float, default=0.5)  # 0.3  0.5  0.7
    parser.add_argument('--lr', default=1e-2, type=float)
    parser.add_argument('--weight_decay', default=1e-5, type=float)

    parser.add_argument('--opt', default='sgd', type=str)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--warm_up', default=1, type=int)
    parser.add_argument('--img_size', default=224, type=int)
    parser.add_argument('--patch_size', default=16, type=int)

    # model setting
    parser.add_argument('--backbone_name', default='vit_small_patch16_224', type=str)
    parser.add_argument('--pre_trained', default=True)

    # dataset parameters
    parser.add_argument('--num_classes', default=20, type=int)
    # parser.add_argument('--data_name', default='SIIM_v3_org', type=str)
    parser.add_argument('--data_name', default='SIIM_v3_gaze', type=str)
    parser.add_argument('--train_data', default="data/train.csv")
    parser.add_argument('--test_data', default='data/test.csv')
    parser.add_argument('--output_dir', default='./save/')

    parser.add_argument('--model_save_interval', default=10, type=int)
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    # parser.add_argument('--exp_documentation', default=exp_documentation)

    parser.add_argument('--mask_sort_choice', default=0.3, type=float)
    parser.add_argument('--test_resize_flatten', default=True, type=bool) # wether resize (14, 14) and flatten to (196)

    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    return parser

class MyDataset_INbreast_Gaze_Mask49_Train_v3(Dataset):
    def __init__(self, args):
        self.data_list = open(args.train_data, 'r').readlines()
        self.data_size = len(self.data_list)
        self.gaze_input_norm = args.gaze_input_norm
        self.gaze_mask_num = args.gaze_mask_num
        self.to_tensor = ToTensorV2()
        self.patch_num = args.img_size // args.patch_size
        self.transform1 = A.Compose([
            A.GaussNoise(p=0.2),
            A.Resize(height=224, width=224, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            # A.ElasticTransform(p=0.2),
            A.Flip(p=0.2),
            # A.ElasticTransform(p=0.2),
            # A.Lambda(p=0.2),
            # A.MaskDropout(p=0.2),
            # A.OpticalDistortion(p=0.2),
            # A.CLAHE(p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.3, 0.3, 0.3), p=1.0),
           
	       
            # ToTensorV2(),
        ],is_check_shapes=False)
        ### 初始化summaryWrite
        


    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):

        data_index = idx % self.data_size
        item = self.data_list[data_index]

        img_path  = item.splitlines()[0]
        label = item.splitlines()[0].split('\\')[-2]
        label = int(name_dict[label])
        # print(img_path)
        #roi_mask ,pred_path = main(pl.Path(img_path))
        # roi_mask = np.ones((224,224))
        img_path_new = os.path.join('savefig',img_path.split('\\')[-1].split('.')[0]+'_smap.png')
        pred_img = cv2.imread(img_path_new)
        pred_img[pred_img<40] = 0
        pred_img[pred_img>40] = 1
        """get bk"""
        src_img = cv2.imread(img_path)
        src_img = cv2.resize(src_img,(640,480))
        src_img =  cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        
        src_img = src_img*pred_img
        # roi_mask = np.array(roi_mask.squeeze().cpu())*1000

        # img1 shape (x,y)

        # img_path, label = item.rstrip().split(',')
        # label = int(label)

        # img_name = img_path.split('/')[-1]
        # org_img_mid = img_name.split('r')[0][:-1]
        # roi_info = img_name.split('r')[1].split('.')[0]
        # tmp = roi_info.split('_')

        # if label == 1 or label == 2:  # r0_1053_1024_2077_736_230_827_343_c0_e0_b0_1_1
        #     npy_name = '{}.npy'.format(org_img_mid)
        #     roi_x, roi_y, roi_x1, roi_y1 = int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3])
        #     box_x, box_y, box_x1, box_y1 = int(tmp[4]), int(tmp[5]), int(tmp[6]), int(tmp[7])
        # elif label == 0:  # r31_2030_1055_3054_c0_e0_0_0
        #     npy_name = '{}.npy'.format(org_img_mid)
        #     roi_x, roi_y, roi_x1, roi_y1 = int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3])
        # gaze_mask_path = r'Gaze_heatmap_path_of_INbreast_dataset/{}'.format(npy_name)

        """get bk"""
        # src_img = cv2.imread(img_path)
        # src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        # bk_mask = np.load(gaze_mask_path)
        # roi_mask = bk_mask[roi_y:roi_y1, roi_x:roi_x1]  # .T
        # roi_mask2 = roi_mask
        ## 设定阈值判定
        # lowbound = np.mean(roi_mask[roi_mask > 0])
        # roi_mask[roi_mask < lowbound] = 0
        
        # plt.subplot(1,2,1)
        # plt.title('src_image')
        # plt.imshow(transform.resize(src_img,(224,224)))
        # plt.subplot(1,2,2)
        # plt.imshow(transform.resize(roi_mask,(224,224)),cmap='gray')
        # plt.title('roi_mask')
        # plt.show()
        
        # bk_mask = roi_mask
        # lowbound = np.mean(bk_mask[bk_mask > 0])
        # bk_mask[bk_mask < lowbound] = 0
        roi_mask = np.ones((224,224))
        norm_img = np.zeros(roi_mask.shape)
        roi_mask1 = cv2.normalize(roi_mask, norm_img, 0, 255, cv2.NORM_MINMAX)
        heat_img = np.asarray(roi_mask1, dtype=np.uint8)
       
        """after transform"""
        transformed = self.transform1(image=src_img, mask=heat_img)
        img_after = transformed['image']  # (224, 224)
        heat_img_after = transformed['mask']  # (224, 224)

        heat_img_196 = cv2.resize(heat_img_after, (self.patch_num, self.patch_num))  # (14, 14)
        ### 这里缩放成一个一个16*16 的小的patch

        tensor_trans = self.to_tensor(image=img_after, mask=heat_img_196)
        img_after = tensor_trans['image']
        heat_img_196 = tensor_trans['mask']
        
        # plt.imshow(heat_img_196)
        # plt.title('heat_img_16*16')
        # plt.show()

        sample = torch.tensor(img_after)
        heat_img_196 = torch.tensor(heat_img_196).flatten(0)
        # flatten的函数作用就是展平开来
        #print('heat_img_196 flatten shape',heat_img_196.shape)
        if self.gaze_input_norm:
            heat_img_196 = (heat_img_196 - heat_img_196.min()) / (heat_img_196.max() - heat_img_196.min() if heat_img_196.max() - heat_img_196.min() != 0 else 1.0)
        # label = torch.tensor(label)
        # print(heat_img_196)
        """generate mask"""
        mask_indexes = torch.argsort(heat_img_196, descending=True)[:self.gaze_mask_num]
        new_zeros = torch.zeros((self.patch_num*self.patch_num), dtype=torch.uint8)
        new_gaze_mask = new_zeros == 1
        new_gaze_mask[mask_indexes] = True
        if new_gaze_mask.sum() != self.gaze_mask_num:
            print('new_gaze_mask.sum() != args.gaze_mask_num')
            exit(0)
        return sample, label, torch.tensor(idx), new_gaze_mask, heat_img_196
        # return format data label path gaze_mask_true_false_196 heat_img_16*16

class MyDataset_SIIM_Gaze_Mask49_Test_v3(Dataset):
    def __init__(self, args):
        self.data_list = open(args.test_data, 'r').readlines()
        self.data_size = len(self.data_list)
        self.gaze_input_norm = True
        self.test_resize_flatten = args.test_resize_flatten
        self.to_tensor = ToTensorV2()
        self.patch_num = args.img_size // args.patch_size
        self.transform1 = A.Compose([
            A.Resize(height=224, width=224, p=1.0),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.3, 0.3, 0.3), p=1.0),
        ],is_check_shapes=False)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):

        data_index = idx % self.data_size
        item = self.data_list[data_index]

        img_path  = item.splitlines()[0]
        label = item.splitlines()[0].split('\\')[-2]
        label = int(name_dict[label])

        img_path_new = os.path.join('savefig',img_path.split('\\')[-1].split('.')[0]+'_smap.png')
        pred_img = cv2.imread(img_path_new)
        plt.imshow(pred_img)
        plt.show()
        pred_img[pred_img<40] = 0
        pred_img[pred_img>40] = 1
        """get bk"""
        # src_img = cv2.imread(img_path)
        # src_img = cv2.resize(src_img,(640,480))

        # roi_mask ,path = main(pl.Path(img_path))
        roi_mask = np.ones((224,224))
        # print(path)
        """get bk"""
        # roi_mask = np.array(roi_mask.squeeze().cpu())
        src_img = cv2.imread(img_path)
        src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        src_img = cv2.resize(src_img,(640,480))
        src_img = src_img*pred_img

        norm_img = np.zeros(roi_mask.shape)
        roi_mask1 = cv2.normalize(roi_mask, norm_img, 0, 255, cv2.NORM_MINMAX)
        heat_img = np.asarray(roi_mask1, dtype=np.uint8)
        """get bk"""
        # src_img = cv2.imread(img_path)
        # src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        # roi_mask = np.load(gaze_mask_path)

        # norm_img = np.zeros(roi_mask.shape)
        # cv2.normalize(roi_mask, norm_img, 0, 255, cv2.NORM_MINMAX)
        # heat_img = np.asarray(norm_img, dtype=np.uint8)

        """after transform"""
        transformed = self.transform1(image=src_img, mask=heat_img)
        img_after = transformed['image']  # (224, 224)
        heat_img_after = transformed['mask']  # (224, 224)
        if self.test_resize_flatten:
            heat_img_after = cv2.resize(heat_img_after, (self.patch_num, self.patch_num))  # (14, 14)

        tensor_trans = self.to_tensor(image=img_after, mask=heat_img_after)
        img_after = tensor_trans['image']
        heat_img_196 = tensor_trans['mask']
        if self.test_resize_flatten:
            heat_img_196 = heat_img_196.flatten(0)
        if self.gaze_input_norm:
            heat_img_196 = (heat_img_196 - heat_img_196.min()) / (heat_img_196.max() - heat_img_196.min() if heat_img_196.max() - heat_img_196.min() != 0 else 1.0)
        # plt.imshow(heat_img_196)
        label = torch.tensor(label)
        return img_after, label, torch.tensor(idx), heat_img_196

# parser = argparse.ArgumentParser('EG_ViT', parents=[get_args_parser()])

# args = parser.parse_args(args=[])

def train_one_epoch_with_gaze_mask_random_OUT(dataloader,model,device,criterion,optimizer, lr_scheduler, args, max_norm=-1,write=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    criterion.train()
    criterion=criterion.to(device)
    model=model.to(device)
    # model.backbone.patch_embed.register_backward_hook(backward_hook)
    epoch_accuracy,epoch_loss=0,0
    start = time.time()
    for idx, (data, label, imgpath, gaze_mask, gaze_hm) in enumerate(dataloader):
        # grad_block=list()

        if idx % 10 == 0:
            print(idx, time.time()-start)

        label=torch.tensor(label).to(device)
        data=data.to(device)
        mask = gaze_mask.to(device)

        if random.random() > args.random_use_mask:
            out = model(data)
        else:
            out = model(data)
            # out = model.forward_with_mask(data, mask)
        loss = criterion(out, label)
        model.zero_grad()
        loss.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        acc = (out.argmax(dim=1) == label).float().mean().item()
        epoch_accuracy += acc / len(dataloader)
        epoch_loss += (loss.item()) / len(dataloader)

        del mask
        # del inputs
    return epoch_loss, epoch_accuracy


# dataset = MyDataset_INbreast_Gaze_Mask49_Train_v3(args)
# dataloader = DataLoader(dataset=dataset,batch_size=8)
# next(iter(dataloader))

def main_SIIM(args):

    write = SummaryWriter('runs/train')

    os.makedirs(args.output_dir, exist_ok=True)
    # define logging
    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler("{}/log.txt".format(args.output_dir), mode='a', encoding='UTF-8'),
            logging.StreamHandler()
        ]
    )

    logging.info("##################\tconfig\t##################")
    for opt_key in vars(args).keys():
        logging.info("{}: {}".format(opt_key, vars(args)[opt_key]))

    """model"""
    if args.model_version == 1:
        logging.info("define model v1")
        mymodel = build_model_v1(args.res_layer, num_class=args.num_classes)  # add to the last layer, can change
    pretrain_weight = timm.create_model(args.backbone_name, num_classes=args.num_classes,pretrained=True).state_dict()
    mymodel.load_state_dict(pretrain_weight, strict=False)

    optimizer = create_optimizer(args, mymodel)
    scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, lr_min=1e-7, k_decay=0.1, warmup_t=args.warm_up)
    loss_fn = LabelSmoothingCrossEntropy(0.1)

    setup_seed(args.seed)

    logging.info("define dataset")
    # training_dataset, testing_dataset = datasets.build_dataset(args.data_name, args)
    training_dataset = MyDataset_INbreast_Gaze_Mask49_Train_v3(args=args)
    training_dataloader = DataLoader(training_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    testing_dataset = MyDataset_SIIM_Gaze_Mask49_Test_v3(args=args)
    testing_dataloader = DataLoader(testing_dataset, args.batch_size,shuffle=False,num_workers=args.num_workers)

    accelerator = Accelerator()
    device = accelerator.device
    mymodel.to(device)
    my_model, my_optimizer, my_training_dataloader = accelerator.prepare(mymodel, optimizer, training_dataloader)
    my_testing_dataloader = accelerator.prepare(testing_dataloader)

    """train"""
    logging.info("##################\tstart training\t##################")
    best_acc, best_epoch, best_auc, best_f1 = 0, 0, 0, 0
    epoch_loss, mask_loss, cont_loss, epoch_acc = None, None, None, None
    # test_data_len_size = 0
    start_time = time.time()

    """orgViT  without mask"""
    # train_epoch = train_one_epoch_without_mask

    """gaze mask"""
    train_epoch = train_one_epoch_with_gaze_mask_random_OUT


    test_epoch = evaluate_without_mask_2class

    logging.info("train use {}".format(train_epoch))
    logging.info("test use {}".format(test_epoch))
    for epoch in range(args.epochs):
        logging.info('Epoch {}, lr={}'.format(epoch, scheduler.optimizer.param_groups[0]['lr']))

        """Train"""
        epoch_loss, epoch_acc = train_epoch(my_training_dataloader, my_model, device, loss_fn, optimizer, scheduler, args,write=write)

        """test"""
        novel_test_acc, auc, f1, data_len_size = test_epoch(my_testing_dataloader, my_model, device)

        if epoch_loss != None and mask_loss == None and cont_loss == None and epoch_acc != None:
            logging.info('Loss:%.4f   TrainAcc:%.4f   TestAcc:%.4f' % (epoch_loss, epoch_acc, novel_test_acc))
        elif epoch_loss != None and mask_loss != None and cont_loss == None and epoch_acc != None:
            logging.info('Loss:%.4f    MaskLoss:%.4f  TrainAcc:%.4f   TestAcc:%.4f'
                % (epoch_loss, mask_loss, epoch_acc, novel_test_acc))
        elif epoch_loss != None and mask_loss == None and cont_loss != None and epoch_acc != None:
            logging.info('Loss:%.4f    cont_loss:%.4f  TrainAcc:%.4f   TestAcc:%.4f'
                 % (epoch_loss, cont_loss, epoch_acc, novel_test_acc))
        elif epoch_loss != None and mask_loss != None and cont_loss != None and epoch_acc != None:
            logging.info('Loss:%.4f    MaskLoss:%.4f   CosLoss:% .4f   TrainAcc:%.4f   TestAcc:%.4f'
                         % (epoch_loss, mask_loss, cont_loss, epoch_acc, novel_test_acc))

        logging.info("data len={}, ACC={}, AUC={}, F1={}".format(data_len_size, novel_test_acc, auc, f1))

        scheduler.step(epoch)

        if novel_test_acc > best_acc:
            best_checkpoint_path = op.join(args.output_dir, 'deit_small_best.pth')
            save_on_master(
                {'model': my_model.state_dict(),
                 'args': args,
                 'optimizer': optimizer.state_dict(),
                 'lr_scheduler': scheduler.state_dict(),
                 'epoch': epoch}, best_checkpoint_path)
            logging.info('### save best ### {}, {}'.format(novel_test_acc, best_checkpoint_path))
            best_acc = round(novel_test_acc, 6)
            best_epoch = epoch
            best_auc = round(auc, 6)
            best_f1 = round(f1, 6)

    logging.info('\n###\nBest at epoch {}, acc={}, auc={}, f1={}'.format(best_epoch, best_acc, best_auc, best_f1))
    logging.info("test data len={}, Time Cost {}".format(data_len_size, str(datetime.timedelta(seconds=int(time.time()-start_time)))))
    logging.info('save at {}'.format(best_checkpoint_path))

def evaluate_without_mask_2class(dataloader,model,device):
    model.eval()
    with torch.no_grad():
        one_hot_num = 0
        data_len_size = 0
        confusion_label, confusion_pred = [], []
        output_list = []
        softmax = nn.Softmax()
        for data, label, imgpath, gaze_mask in dataloader:
            label=torch.tensor([item for item in label]).to(device) #shape 16
            data = data.to(device)
            B=data.shape[0]
            output = model(data) # shape (16,20)

            acc = (output.argmax(dim=1) == label).float().sum().item()
            # print('acc',acc)
            one_hot_num += acc  # / len(dataloader)
            data_len_size += B

            output_list.append(output)
            # print('output_list',output_list)
            confusion_pred += list(output.argmax(dim=1).squeeze().cpu().numpy())
            # print('confusion_pred',confusion_pred)

            confusion_label += list(label.cpu().numpy())

    ACC = one_hot_num / data_len_size
    # new_var = print(confusion_label)
    # new_var
    out_cat = softmax(torch.cat(output_list, dim=0)).cpu().numpy()# shape 400 20
    # print(out_cat)
    # auc = roc_auc_score(confusion_label, out_cat.max(axis=1))
    # auc = roc_auc_score(confusion_label, out_cat[:, 1],multi_class='ovr')
    # print('auc',auc)
    auc = 0.5
    f1 = f1_score(confusion_label, confusion_pred, average='weighted')
    # print('f1',f1)
    return ACC, auc, f1, data_len_size

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    # 代码
    # torch.backends.cudnn.enabled = False
    parser = get_args_parser()
    args = parser.parse_args(args=[])
    main_SIIM(args=args)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # accelerator = Accelerator()
    # mymodel = build_model_v1(args.res_layer, num_class=args.num_classes)  # add to the last layer, can change
    
    # optimizer = create_optimizer(args, mymodel)
    
    # training_dataset = MyDataset_INbreast_Gaze_Mask49_Train_v3(args=args)
    # training_dataloader = DataLoader(training_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)

    # my_model, my_optimizer, my_training_dataloader = accelerator.prepare(mymodel, optimizer, training_dataloader)

    # pretrain_weight = timm.create_model(args.backbone_name, num_classes=args.num_classes).state_dict()
    # mymodel.load_state_dict(pretrain_weight, strict=False)
    # # train_epoch = train_one_epoch_with_gaze_mask_random_OUT
    # test_epoch = evaluate_without_mask_2class
    
    # testing_dataset = MyDataset_SIIM_Gaze_Mask49_Test_v3(args=args)
    # testing_dataloader = DataLoader(testing_dataset, args.batch_size, num_workers=args.num_workers,shuffle=True)

    
   

    # my_testing_dataloader = accelerator.prepare(testing_dataloader)

    # novel_test_acc, auc, f1, data_len_size = test_epoch(my_testing_dataloader, my_model, device)
