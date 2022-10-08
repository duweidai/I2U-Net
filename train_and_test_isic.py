#!/usr/bin/python3
# these code is for ISIC 2018: Skin Lesion Analysis Towards Melanoma Segmentation
# -*- coding: utf-8 -*-
# @Author  : Duwei Dai
import os
import torch
import math
import visdom
import torch.utils.data as Data
import argparse
import numpy as np
import sys
from tqdm import tqdm
import random
from thop import profile
from ptflops import get_model_complexity_info


from distutils.version import LooseVersion
from Datasets.ISIC2018 import ISIC2018_dataset
from Datasets.PH2 import ph2_dataset
from utils.transform import ISIC2018_transform, ISIC2018_transform_320, ISIC2018_transform_newdata
# ISIC2018_transform_newdata: not resize, only apply randomflip_rotate

from Models.I2U_Net import I2U_Net_L, I2U_Net_M, I2U_Net_S

from utils.dice_loss import get_soft_label, val_dice_isic
from utils.dice_loss import Intersection_over_Union_isic
from utils.dice_loss_github import SoftDiceLoss_git, CrossentropyND

from utils.evaluation import AverageMeter
from utils.binary import assd, dc, jc, precision, sensitivity, specificity, F1, ACC
from torch.optim import lr_scheduler

from time import *
from PIL import Image


Test_Model = {
                "I2U_Net_L": I2U_Net_L,
                "I2U_Net_M": I2U_Net_M,
                "I2U_Net_S": I2U_Net_S,           
             }
             
             
Test_Dataset = {'ISIC2018': ISIC2018_dataset}

Test_Transform = {'A': ISIC2018_transform, 'B':ISIC2018_transform_320, "C":ISIC2018_transform_newdata}


def structure_loss(out_f, target, num_classes=2):
    loss = []
    soft_dice_loss2 = SoftDiceLoss_git(batch_dice=False, dc_log=False)
    CE_loss_F = CrossentropyND()
    
    for i in range(len(out_f)):
        out_c = out_f[i]
        target_soft_a = get_soft_label(target, num_classes) 
        target_soft = target_soft_a.permute(0, 3, 1, 2)
        dice_loss_f = soft_dice_loss2(out_c, target_soft)    
        ce_loss_f = CE_loss_F(out_c, target)
        loss_f = dice_loss_f + ce_loss_f
        loss.append(loss_f)

    return sum(loss)


def one_loss(out_c, target, num_classes=2):
    soft_dice_loss2 = SoftDiceLoss_git(batch_dice=False, dc_log=False)
    CE_loss_F = CrossentropyND()
    
    target_soft_a = get_soft_label(target, num_classes) 
    target_soft = target_soft_a.permute(0, 3, 1, 2)
    dice_loss_f = soft_dice_loss2(out_c, target_soft)    
    ce_loss_f = CE_loss_F(out_c, target)
    loss_f = dice_loss_f + ce_loss_f

    return loss_f


class Logger(object):
    def __init__(self,logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass 
        
        
def train(train_loader, model, scheduler, optimizer, args, epoch):
    losses = AverageMeter()
    # current_loss_f = "CE_softdice"       # softdice or CE_softdice
    
    model.train()
    for step, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        image = x.float().cuda()                                   
        target = y.float().cuda()                                  

        out_f = model(image)
        """
        path_1: lesion          foreground
        path_2: background      background
        """
        # ---- loss function ----
        if isinstance(out_f, list) or isinstance(out_f, tuple):
            loss = structure_loss(out_f, target)
        else:
            loss = one_loss(out_f, target)
        
        losses.update(loss.data, image.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % (math.ceil(float(len(train_loader.dataset))/args.batch_size)) == 0:
                   print('current lr: {} | Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {losses.avg:.6f}'.format(
                   optimizer.state_dict()['param_groups'][0]['lr'],
                   epoch, step * len(image), len(train_loader.dataset),
                   100. * step / len(train_loader), losses=losses))
                
    print('The average loss:{losses.avg:.4f}'.format(losses=losses))
    return losses.avg


def valid_isic(valid_loader, model, optimizer, args, epoch, best_score, val_acc_log):
    isic_Jaccard = []
    isic_dc = []

    model.eval()
    for step, (t, k) in tqdm(enumerate(valid_loader), total=len(valid_loader), mininterval = 0.001):
        image = t.float().cuda()
        target = k.float().cuda()

        out_f = model(image)                                             # model output
        if isinstance(out_f, list) or isinstance(out_f, tuple):
            output = out_f[-1]
        else:
            output = out_f

        output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
        output_dis_test = output_dis.permute(0, 2, 3, 1).float()
        target_test = target.permute(0, 2, 3, 1).float()
        isic_b_Jaccard = jc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        isic_b_dc = dc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        isic_Jaccard.append(isic_b_Jaccard)
        isic_dc.append(isic_b_dc)

    isic_Jaccard_mean = np.average(isic_Jaccard)
    isic_dc_mean = np.average(isic_dc)
    net_score = isic_Jaccard_mean + isic_dc_mean
        
    print('The ISIC Dice score: {dice: .4f}; '
          'The ISIC JC score: {jc: .4f}'.format(
           dice=isic_dc_mean, jc=isic_Jaccard_mean))
           
    with open(val_acc_log, 'a') as vlog_file:
        line = "{} | {dice: .4f} | {jc: .4f}".format(epoch, dice=isic_dc_mean, jc=isic_Jaccard_mean)
        vlog_file.write(line+'\n')

    if net_score > max(best_score):
        best_score.append(net_score)
        print(best_score)
        modelname = args.ckpt + '/' + 'best_score' + '_' + args.data + '_checkpoint.pth.tar'
        print('the best model will be saved at {}'.format(modelname))
        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
        torch.save(state, modelname)

    return isic_Jaccard_mean, isic_dc_mean, net_score


def test_isic(test_loader, model, args, test_acc_log, date_type, save_img=True):

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    isic_dice = []
    isic_iou = []
   # isic_assd = []
    isic_acc = []
    isic_sensitive = []
    isic_specificy = []
    isic_precision = []
    isic_f1_score = []
    isic_Jaccard_M = []
    isic_Jaccard_N = []
    isic_Jaccard = []
    isic_dc = []
    infer_time = []
    
    print("******************************************************************** {} || start **********************************".format(date_type)+"\n")
    
    modelname = args.ckpt + '/' + 'best_score' + '_' + args.data + '_checkpoint.pth.tar'
    img_saved_dir_root = os.path.join(args.ckpt, "segmentation_result")
    if os.path.isfile(modelname):
        print("=> Loading checkpoint '{}'".format(modelname))
        checkpoint = torch.load(modelname)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> Loaded saved the best model at (epoch {})".format(checkpoint['epoch']))
    else:
        print("=> No checkpoint found at '{}'".format(modelname))

    model.eval()
    for step, (name, img, lab) in tqdm(enumerate(test_loader), total=len(test_loader)):
        image = img.float().cuda()
        target = lab.float().cuda() # [batch, 1, 224, 320]
        
        begin_time = time()
        out_f = model(image)
        
        if isinstance(out_f, list) or isinstance(out_f, tuple):
            output = out_f[-1]
        else:
            output = out_f
       
        end_time = time()
        pred_time = end_time - begin_time
        infer_time.append(pred_time)
        
        output_dis = torch.max(output, 1)[1].unsqueeze(dim=1)
        
        """
        save segmentation result
        """
        if save_img:
            if date_type=="ISIC2018" and args.val_folder == "folder1":
                npy_path = os.path.join(args.root_path, 'image', name[0])
                img = np.load(npy_path)
                im = Image.fromarray(np.uint8(img))
                im_path = name[0].split(".")[0] + "_img" + ".png"
                img_saved_dir = os.path.join(img_saved_dir_root, name[0].split(".")[0])
                if not os.path.isdir(img_saved_dir):
                    os.makedirs(img_saved_dir)
                im.save(os.path.join(img_saved_dir, im_path))
                
                target_np = target.squeeze().cpu().numpy()
                label = Image.fromarray(np.uint8(target_np*255))
                label_path = name[0].split(".")[0] + "_label" + ".png"
                label.save(os.path.join(img_saved_dir, label_path))
                
                seg_np = output_dis.squeeze().cpu().numpy()
                seg = Image.fromarray(np.uint8(seg_np*255))
                seg_path = name[0].split(".")[0] + "_seg" + ".png"
                seg.save(os.path.join(img_saved_dir, seg_path))
            else:
                pass        
        else:
            pass
        
        output_dis_test = output_dis.permute(0, 2, 3, 1).float()
        target_test = target.permute(0, 2, 3, 1).float()
        output_soft = get_soft_label(output_dis, 2) 
        target_soft = get_soft_label(target, 2) 

        label_arr = np.squeeze(target_soft.cpu().numpy()).astype(np.uint8)
        output_arr = np.squeeze(output_soft.cpu().byte().numpy()).astype(np.uint8)

        isic_b_dice = val_dice_isic(output_soft, target_soft, 2)                                         # the dice
        isic_b_iou = Intersection_over_Union_isic(output_dis_test, target_test, 1)                       # the iou
        # isic_b_asd = assd(output_arr[:, :, 1], label_arr[:, :, 1])                                     # the assd
        isic_b_acc = ACC(output_dis_test.cpu().numpy(), target_test.cpu().numpy())                       # the accuracy
        isic_b_sensitive = sensitivity(output_dis_test.cpu().numpy(), target_test.cpu().numpy())         # the sensitivity
        isic_b_specificy = specificity(output_dis_test.cpu().numpy(), target_test.cpu().numpy())         # the specificity
        isic_b_precision = precision(output_dis_test.cpu().numpy(), target_test.cpu().numpy())           # the precision
        isic_b_f1_score = F1(output_dis_test.cpu().numpy(), target_test.cpu().numpy())                   # the F1
        isic_b_Jaccard_m = jc(output_arr[:, :, 1], label_arr[:, :, 1])                                   # the Jaccard melanoma
        isic_b_Jaccard_n = jc(output_arr[:, :, 0], label_arr[:, :, 0])                                   # the Jaccard no-melanoma
        isic_b_Jaccard = jc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        isic_b_dc = dc(output_dis_test.cpu().numpy(), target_test.cpu().numpy())
        
        dice_np = isic_b_dice.data.cpu().numpy()
        iou_np = isic_b_iou.data.cpu().numpy()
       
        isic_dice.append(dice_np)
        isic_iou.append(iou_np)
       # isic_assd.append(isic_b_asd)
        isic_acc.append(isic_b_acc)
        isic_sensitive.append(isic_b_sensitive)
        isic_specificy.append(isic_b_specificy)
        isic_precision.append(isic_b_precision)
        isic_f1_score.append(isic_b_f1_score)
        isic_Jaccard_M.append(isic_b_Jaccard_m)
        isic_Jaccard_N.append(isic_b_Jaccard_n)
        isic_Jaccard.append(isic_b_Jaccard)
        isic_dc.append(isic_b_dc)
        
        if date_type== "ISIC2018":
            with open(test_acc_log, 'a') as tlog_file:
                line = "{} | {dice: .4f} | {jc: .4f}".format(name[0], dice=isic_b_dc, jc=isic_b_Jaccard)
                tlog_file.write(line+'\n')
        elif date_type== "PH2":
            pass
        else:
            print("can not supports dataset: {}", date_type)

    all_time = np.sum(infer_time)
    isic_dice_mean = np.average(isic_dice)
    isic_dice_std = np.std(isic_dice)

    isic_iou_mean = np.average(isic_iou)
    isic_iou_std = np.std(isic_iou)

   # isic_assd_mean = np.average(isic_assd)
   # isic_assd_std = np.std(isic_assd)
      
    isic_acc_mean = np.average(isic_acc)
    isic_acc_std = np.std(isic_acc)
    
    isic_sensitive_mean = np.average(isic_sensitive)
    isic_sensitive_std = np.std(isic_sensitive)
    
    isic_specificy_mean = np.average(isic_specificy)
    isic_specificy_std = np.std(isic_specificy)
    
    isic_precision_mean = np.average(isic_precision)
    isic_precision_std = np.std(isic_precision)
    
    isic_f1_score_mean = np.average(isic_f1_score)
    iisic_f1_score_std = np.std(isic_f1_score)
    
    isic_Jaccard_M_mean = np.average(isic_Jaccard_M)
    isic_Jaccard_M_std = np.std(isic_Jaccard_M)
    
    isic_Jaccard_N_mean = np.average(isic_Jaccard_N)
    isic_Jaccard_N_std = np.std(isic_Jaccard_N)
    
    isic_Jaccard_mean = np.average(isic_Jaccard)
    isic_Jaccard_std = np.std(isic_Jaccard)
    
    isic_dc_mean = np.average(isic_dc)
    isic_dc_std = np.std(isic_dc)

    print('The mean dice: {isic_dice_mean: .4f}; The dice std: {isic_dice_std: .4f}'.format(
           isic_dice_mean=isic_dice_mean, isic_dice_std=isic_dice_std))
    print('The mean IoU: {isic_iou_mean: .4f}; The IoU std: {isic_iou_std: .4f}'.format(
           isic_iou_mean=isic_iou_mean, isic_iou_std=isic_iou_std))
    print('The mean ACC: {isic_acc_mean: .4f}; The ACC std: {isic_acc_std: .4f}'.format(
           isic_acc_mean=isic_acc_mean, isic_acc_std=isic_acc_std))
    print('The mean sensitive: {isic_sensitive_mean: .4f}; The sensitive std: {isic_sensitive_std: .4f}'.format(
           isic_sensitive_mean=isic_sensitive_mean, isic_sensitive_std=isic_sensitive_std)) 
    print('The mean specificy: {isic_specificy_mean: .4f}; The specificy std: {isic_specificy_std: .4f}'.format(
           isic_specificy_mean=isic_specificy_mean, isic_specificy_std=isic_specificy_std))
    print('The mean precision: {isic_precision_mean: .4f}; The precision std: {isic_precision_std: .4f}'.format(
           isic_precision_mean=isic_precision_mean, isic_precision_std=isic_precision_std))
    print('The mean f1_score: {isic_f1_score_mean: .4f}; The f1_score std: {iisic_f1_score_std: .4f}'.format(
           isic_f1_score_mean=isic_f1_score_mean, iisic_f1_score_std=iisic_f1_score_std))
    print('The mean Jaccard_M: {isic_Jaccard_M_mean: .4f}; The Jaccard_M std: {isic_Jaccard_M_std: .4f}'.format(
           isic_Jaccard_M_mean=isic_Jaccard_M_mean, isic_Jaccard_M_std=isic_Jaccard_M_std))
    print('The mean Jaccard_N: {isic_Jaccard_N_mean: .4f}; The Jaccard_N std: {isic_Jaccard_N_std: .4f}'.format(
           isic_Jaccard_N_mean=isic_Jaccard_N_mean, isic_Jaccard_N_std=isic_Jaccard_N_std))
    print('The mean Jaccard: {isic_Jaccard_mean: .4f}; The Jaccard std: {isic_Jaccard_std: .4f}'.format(
           isic_Jaccard_mean=isic_Jaccard_mean, isic_Jaccard_std=isic_Jaccard_std))
    print('The mean dc: {isic_dc_mean: .4f}; The dc std: {isic_dc_std: .4f}'.format(
           isic_dc_mean=isic_dc_mean, isic_dc_std=isic_dc_std))
    print('The inference time: {time: .4f}'.format(time=all_time))

    print("******************************************************************** {} || end **********************************".format(date_type)+"\n")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic =True
  
    
def main(args, val_acc_log, test_acc_log):
    best_score = [0]
    start_epoch = args.start_epoch
    print('loading the {0},{1},{2} dataset ...'.format('train', 'test', 'test'))
    trainset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='train', 
                                       with_name=False, transform=Test_Transform[args.transform])
    validset = Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='test',
                                       with_name=False, transform=Test_Transform[args.transform])
    testset =  Test_Dataset[args.data](dataset_folder=args.root_path, folder=args.val_folder, train_type='test',
                                       with_name=True, transform=Test_Transform[args.transform])

    trainloader = Data.DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=6)
    validloader = Data.DataLoader(dataset=validset, batch_size=1, shuffle=False, pin_memory=True, num_workers=6)
    testloader = Data.DataLoader(dataset=testset, batch_size=1, shuffle=False, pin_memory=True, num_workers=6)
    
    # ph2_dir = "/home/gpu2/10t_disk/ddw/TMI_0712/Datasets/ISIC_2018/PH2_npy_224_320"
    # ph2_list = "/home/gpu2/10t_disk/ddw/TMI_0720_code/ISIC_main/PH2_npy_224_320/ph2_list.txt"
    
    ph2_dir = args.root_ph2
    ph2_list = args.root_ph2_list
    
    ph2_data = ph2_dataset(dataset_folder=ph2_dir, image_list=ph2_list, train_type='test', transform=Test_Transform[args.transform])
    ph2_img = Data.DataLoader(dataset=ph2_data, batch_size=1, shuffle=False, pin_memory=True, num_workers=6)
    
    print('Loading is done\n')

    args.num_input
    args.num_classes
    args.out_size
    print("args.out_size: ", args.out_size)
    print("args.h_init_type is: ", args.h_init_type)

    model = Test_Model[args.id](classes=2, channels=3)
    model = model.cuda()

    print("---------------------------------------------------------------------------------------------------------------------")
    print("Network Architecture of Model {}:".format(args.id))
    num_para = 0
    for name, param in model.named_parameters():
        num_mul = 1
        for x in param.size():
            num_mul *= x
        num_para += num_mul
    print(model)
    print("---------------------------------------------------------------------------------------------------------------------")
    
    input = torch.randn(1, 3, args.out_size[0], args.out_size[1]) # batch_size = 1
    flops, params = profile(model, inputs =(input.cuda(), ))
    print("---------------------------------------------------------------------------------------------------------------------")
    print("\n")
    print("profile test result: ")
    print("Flops: {} G".format(flops/1e9))
    print("params: {} M".format(params/1e6))

    flops, params = get_model_complexity_info(model, (3, args.out_size[0], args.out_size[1]), as_strings=True, print_per_layer_stat=False) # default batch_size =1
    print("\n")
    print("ptflops test result: ")
    print("Flops: {}".format(flops))
    print("Params: "+params)
    
    print("\n")
    print("Number of trainable parameters {0} in Model {1}".format(num_para, args.id))
    
    print("args.h_init_type is: ", args.h_init_type)
    print("---------------------------------------------------------------------------------------------------------------------")

    # Define optimizers and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_rate, weight_decay=args.weight_decay)    
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min = 0.00001)    # lr_3

    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> Loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['opt_dict'])
            print("=> Loaded checkpoint (epoch {})".format(checkpoint['epoch']))
        else:
            print("=> No checkpoint found at '{}'".format(args.resume))

    print("Start training ...")
    for epoch in range(start_epoch+1, args.epochs + 1):
        scheduler.step()
        train_avg_loss = train(trainloader, model, scheduler, optimizer, args, epoch)
        isic_Jaccard_mean, isic_dc_mean, net_score = valid_isic(validloader, model, optimizer, args, epoch, best_score, val_acc_log)
        if epoch > args.particular_epoch:
            if epoch % args.save_epochs_steps == 0:
                filename = args.ckpt + '/' + str(epoch) + '_' + args.data + '_checkpoint.pth.tar'
                print('the model will be saved at {}'.format(filename))
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}
                torch.save(state, filename)

    print('Training Done! Start testing')
    
    test_isic(testloader, model, args, test_acc_log, "ISIC2018", save_img=args.save_img) 
    test_isic(ph2_img, model, args, test_acc_log, "PH2", save_img=False)

    print('Testing Done!')
    
    
if __name__ == '__main__':

    """
    This project supports the following models:
    compared methods:
        Methods                                                       Params    Flops                              
        I2U_Net_L                                                     29.65     9.36                               
        I2U_Net_M                                                     27.49     8.26 
        I2U_Net_S                                                     7.03      2.74           
    """
    
    # setup_seed(200)
    os.environ['CUDA_VISIBLE_DEVICES']= '0'                                                 # gpu-id
    
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), 'PyTorch>=0.4.0 is required'
    parser = argparse.ArgumentParser(description='Comprehensive attention network for biomedical Dataset')
    
    parser.add_argument('--id', default="I2U_Net_S",
                        help='I2U_Net')                                                   # Select a loaded model name

    # Path related arguments
    parser.add_argument('--root_path', default='/home/gpu2/10t_disk/ddw/TMI_0712/Datasets/ISIC_2018/ISIC2018_npy_all_224_224',
                        help='root directory of training data')                                      # storage path of ISIC2018 dataset 
    parser.add_argument('--ckpt', default='./saved_models_0920/',
                        help='folder to output checkpoints')                                # The folder in which the trained model is saved
    parser.add_argument('--transform', default='C', type=str,
                        help='which ISIC2018_transform to choose') 
    parser.add_argument('--h_init_type', default='m_0', type=str)                   #  m_0--> zero; m_1--> torch.randn(); m_2--> image_resize;  m_3--> m_1+m_2  
    
    parser.add_argument('--data', default='ISIC2018', help='choose the dataset')            
    parser.add_argument('--out_size', default=(224, 224), help='the output image size')
    parser.add_argument('--val_folder', default='folder3', type=str,
                        help='folder1、folder2、folder3、folder4、folder5')                 # five-fold cross-validation
                        
    parser.add_argument('--seed', type=int, default=1234, help='random seed')               # default 1234
    parser.add_argument('--save_img', type=str, default=False, help='whether save segmentation result')
    
    parser.add_argument('--root_ph2', default='/home/gpu2/10t_disk/ddw/TMI_0720_code/ISIC_main/PH2_npy_224_320',
                        help='root directory of external validation')                                      # storage path of ph2 data
    parser.add_argument('--root_ph2_list', default='/home/gpu2/10t_disk/ddw/TMI_0720_code/ISIC_main/PH2_npy_224_320/ph2_list.txt',
                        help='image list of external validation')                                          # storage path of ph2 samples list   
    
    # optimization related arguments
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--batch_size', type=int, default=24, metavar='N',              
                        help='input batch size for training (default: 12)')                 # batch_size
    parser.add_argument('--lr_rate', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 0.001)')                              # default=1e-4                            
    parser.add_argument('--num_classes', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--num_input', default=3, type=int,
                        help='number of input image for each patient')
    parser.add_argument('--weight_decay', default=1e-8, type=float, help='weights regularizer')
    parser.add_argument('--particular_epoch', default=30, type=int,
                        help='after this number, we will save models more frequently')
    parser.add_argument('--save_epochs_steps', default=400, type=int,
                        help='frequency to save models after a particular number of epochs')
    parser.add_argument('--resume', default='',
                        help='the checkpoint that resumes from')

    args = parser.parse_args()
    
    args.ckpt = os.path.join(args.ckpt, args.data, args.val_folder, args.id)    
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)
    logfile = os.path.join(args.ckpt,'{}_{}.txt'.format(args.val_folder, args.id))          # path of the training log
    sys.stdout = Logger(logfile)  
    
    val_acc_log = os.path.join(args.ckpt,'val_acc_{}_{}.txt'.format(args.val_folder, args.id))   
    test_acc_log = os.path.join(args.ckpt,'test_acc_{}_{}.txt'.format(args.val_folder, args.id))    
    
    print('Models are saved at %s' % (args.ckpt))
    print("Input arguments:")
    for key, value in vars(args).items():
        print("{:16} {}".format(key, value))

    if args.start_epoch > 1:
        args.resume = args.ckpt + '/' + str(args.start_epoch) + '_' + args.data + '_checkpoint.pth.tar'

    main(args, val_acc_log, test_acc_log)
