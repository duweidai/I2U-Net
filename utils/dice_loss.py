import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, prediction, soft_ground_truth, num_class=3, weight_map=None, eps=1e-8):
        dice_loss_ave, dice_score_lesion = soft_dice_loss(prediction, soft_ground_truth, num_class, weight_map)
        return dice_loss_ave, dice_score_lesion


def get_soft_label(input_tensor, num_class):
    """
        convert a label tensor to soft label
        input_tensor: tensor with shape [N, C, H, W]
        output_tensor: shape [N, H, W, num_class]
    """
    tensor_list = []
    input_tensor = input_tensor.permute(0, 2, 3, 1)
    for i in range(num_class):
        temp_prob = torch.eq(input_tensor, i * torch.ones_like(input_tensor))
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=-1)
    output_tensor = output_tensor.float()
    return output_tensor


def soft_dice_loss(prediction, soft_ground_truth, num_class, weight_map=None):
    predict = prediction.permute(0, 2, 3, 1)
    pred = predict.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    n_voxels = ground.size(0)
    if weight_map is not None:
        weight_map = weight_map.view(-1)
        weight_map_nclass = weight_map.repeat(num_class).view_as(pred)
        ref_vol = torch.sum(weight_map_nclass * ground, 0)
        intersect = torch.sum(weight_map_nclass * ground * pred, 0)
        seg_vol = torch.sum(weight_map_nclass * pred, 0)
    else:
        ref_vol = torch.sum(ground, 0)
        intersect = torch.sum(ground * pred, 0)
        seg_vol = torch.sum(pred, 0)
    dice_score = (2.0 * intersect + 1e-5) / (ref_vol + seg_vol + 1.0 + 1e-5)
    # dice_loss = 1.0 - torch.mean(dice_score)
    # return dice_loss
    dice_loss = -torch.log(dice_score)
    dice_loss_ave = torch.mean(dice_loss)
    dice_score_lesion = dice_loss[1]
    return dice_loss_ave, dice_score_lesion

def IOU_loss(prediction, soft_ground_truth, num_class):
    predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    iou_score = intersect / (ref_vol + seg_vol - intersect + 1.0)
    iou_loss = torch.mean(-torch.log(iou_score))

    return iou_loss

def jc_loss(prediction, soft_ground_truth, num_class):
    predict = prediction.permute(0, 2, 3, 1)
    pred = predict[:,:,:,1].contiguous().view(-1, num_class)
   # pred = prediction[:,:,:,1].view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth[:,:,:,1].view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    iou_score = intersect / (ref_vol + seg_vol - intersect + 1.0)
    #jc = 10*(1-iou_score)
    jc = 20*torch.mean(-torch.log(iou_score))

    return jc


def val_dice_fetus(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)
    dice_mean_score = torch.mean(dice_score)
    placenta_dice = dice_score[1]
    brain_dice = dice_score[2]

    return placenta_dice, brain_dice


def Intersection_over_Union_fetus(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    iou_score = intersect / (ref_vol + seg_vol - intersect + 1.0)
    dice_mean_score = torch.mean(iou_score)
    placenta_iou = iou_score[1]
    brain_iou = iou_score[2]

    return placenta_iou, brain_iou


def val_dice_isic(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
   # pred = prediction.contiguous().view(-1, num_class)
    pred = prediction.view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
   # dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)
    dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1e-6)
    dice_mean_score = torch.mean(dice_score)

    return dice_mean_score
    

def val_dice_isic_v1(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
   # pred = prediction.contiguous().view(-1, num_class)
    pred = prediction.view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
   # dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)
    smooth = 0.001
    dice_score = 2.0 * (intersect+smooth) / (ref_vol + seg_vol + smooth)
    dice_mean_score = torch.mean(dice_score)

    return dice_mean_score
    


def val_dice_isic_raw0(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
   # dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)
    dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1e-6)
    dice_mean_score = torch.mean(dice_score)

    return dice_mean_score


def Intersection_over_Union_isic(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    # iou_score = intersect / (ref_vol + seg_vol - intersect + 1.0)
    iou_score = intersect / (ref_vol + seg_vol - intersect + 1e-6)
    iou_mean_score = torch.mean(iou_score)

    return iou_mean_score

def Intersection_over_Union_isic_v1(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    smooth = 0.001
    iou_score = (intersect+smooth )/ (ref_vol + seg_vol - intersect + smooth)
    iou_mean_score = torch.mean(iou_score)

    return iou_mean_score
