import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from models.origin_unet import UNet1
import medicaltorch.losses as mt_losses
import medicaltorch.metrics as mt_metrics
from torch.nn import functional as F
from . import origin_unet
import numpy as np
import surface_distance


###############################################################################
# Helper Functions
###############################################################################

def calculate_dice(pred, label):
    n_class = 5
    # pred = pred.squeeze()
    # label = label.squeeze()
    dice = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(5):
        tpred = np.array(pred == i, dtype=np.uint8)
        # print('avg',np.array(pred==i,dtype=np.uint8).sum())
        tlabel = np.array(label == i, dtype=np.uint8)
        dice[i] = mt_metrics.dice_score(tpred, tlabel)
    return dice

def calculate_precision(pred, label, entropy, threshold):
    n_class = 5
    # pred = pred.squeeze()
    # label = label.squeeze()
    score = np.array([0.0,0.0,0.0,0.0])
    score = 0.0
    print('unique:',np.unique(pred))
    # print('unique:',np.unique(pred==1))
    # print('unique:', np.unique(pred == 2))
    # print('unique:', np.unique(pred == 3))
    # print('unique:', np.unique(pred == 4))
    thresh = np.array(entropy <= threshold, dtype=np.uint8)
    tps = 0.0
    fore = np.array(pred>0, dtype=np.uint8)
    for i in range(5):
        tpred = np.array(pred == i, dtype=np.uint8)

        tpred = tpred*thresh
        tlabel = np.array(label == i, dtype=np.uint8) * thresh
        # print('avg p:',i, np.array(pred == i, dtype=np.uint8).sum())
        # print('avg l:', i, np.array(label == i, dtype=np.uint8).sum())
        # print('avg e:', i, np.array(entropy>=threshold, dtype=np.uint8).sum())
        tp = (tpred * tlabel).sum()
        tps += tp
        # # gt = tlabel.sum()
        # prediction = tpred.sum()
        # score[i] = 100.0*tp/(prediction)
    score = 100.0*tps/(thresh).sum()

    return score

def calculate_avg_distance(pred, label):
    n_class = 5
    pred = pred.squeeze()
    label = label.squeeze()
    distance = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    for i in range(5):
        tpred = np.array(pred == i)
        tlabel = np.array(label == i)
        results = surface_distance.compute_surface_distances(tlabel, tpred,(1,1,1))
        distance_result = surface_distance.compute_average_surface_distance(results)
        distance[i] = (distance_result[1]+distance_result[0])/2.0
    return distance


def calculate_dice_binary(pred, label):
    # n_class = 5
    pred = pred.squeeze()
    label = label.squeeze()
    dice = 0.0
    # for i in range(5):
    tpred = np.array(pred == 1, dtype=np.uint8)
    tlabel = np.array(label == 1, dtype=np.uint8)
    dice = mt_metrics.dice_score(tpred, tlabel)
    return dice


def calculate_precision_binary(pred, label, entropy,threshold):
    # n_class = 5
    pred = pred.squeeze()
    label = label.squeeze()
    score = 0.0
    # for i in range(5):
    tps = 0.0
    thresh = np.array(entropy<=threshold, dtype=np.uint8)
    tpred = np.array(pred == 1, dtype=np.uint8) * thresh
    # print(np.array(entropy>=threshold, dtype=np.uint8).sum())
    tlabel = np.array(label == 1, dtype=np.uint8) * thresh
    tp = (tpred*tlabel).sum()
    tps += tp

    tpred = np.array(pred == 0, dtype=np.uint8)
    # print(np.array(entropy>=threshold, dtype=np.uint8).sum())
    tlabel = np.array(label == 0, dtype=np.uint8) * thresh
    tp = (tpred * tlabel).sum()
    tps += tp
    score = 100.0 * tps/thresh.sum()
    return score


def calculate_distance_binary(pred, label):
    # n_class = 5
    pred = pred.squeeze()
    label = label.squeeze()
    distance = 0.0
    # for i in range(5):
    tpred = np.array(pred == 1)
    tlabel = np.array(label == 1)
    results = surface_distance.compute_surface_distances(tlabel, tpred,(1,1,1))
    distance_result = surface_distance.compute_average_surface_distance(results)
    distance = (distance_result[1]+distance_result[0])/2.0
    return distance

def encode_segmap(mask, filename):
    valid_classes = [0, 61, 126, 150, 246]
    train_classes = [0, 1, 2, 3, 4]
    class_map = dict(zip(train_classes, valid_classes))
    for validc in train_classes:
        mask[mask == validc] = class_map[validc]
    return mask


def threshold_predictions(predictions, thr=0.9):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer



def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


def define_UNet(n_channels, n_classes, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = origin_unet.unet_ori_new(n_channels, n_classes)
    return init_net(net, init_type, init_gain, gpu_ids)


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':  # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(
                *real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp  # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None



    ##############################################################################


# Classes
##############################################################################

def multiDiceLossL(input, label, gpu_id):
    # C = 5
    # diceLoss = 0.
    B,C,H,W = input.shape
    label = label.long()
    diceLoss = 0.
    if C > 1:
        target = np.zeros((C, label.shape[0], label.shape[1], label.shape[2]))  # 5,4,256,256
        train_gt = np.array(label.cpu())

        for i in range(C):
            target[i] = np.array((train_gt == i), dtype=np.uint8)
        target = target.transpose(1, 0, 2, 3)
        target = torch.FloatTensor(target).cuda('cuda:{}'.format(gpu_id[0]))
    if C > 1:
        for i in range(C-1):
            diceLoss += (1 + mt_losses.dice_loss(input[:, i, ::].contiguous(), target[:, i, ::].contiguous()))
        return diceLoss/(C-1)
    else:
        diceLoss += (1 + mt_losses.dice_loss(input[:, 0, ::].contiguous(), label[:, 0, ::].contiguous()))
        return diceLoss / C




def multiDiceLossConWeight(input, input1, label, gpu_id):
    # C = 5
    # diceLoss = 0.
    eps = 0.0001
    eps0 = 1e-9
    B, C, H, W = input.shape
    label = label.long()
    diceLoss = 0.
    if C>1:
        target = np.zeros((C, label.shape[0], label.shape[1], label.shape[2]))  # 5,4,256,256
        train_gt = np.array(label.cpu())
        enweight = torch.sigmoid(torch.sum(torch.log(input1+eps0) * (input1+eps0), dim=1))

        for i in range(C):
            target[i] = np.array((train_gt == i), dtype=np.uint8)
        target = target.transpose(1, 0, 2, 3)
        target = torch.FloatTensor(target).cuda('cuda:{}'.format(gpu_id[0]))
    if C > 1:
        for i in range(C):
            intersection = (enweight * input[:, i, ::].contiguous() * target[:, i, ::].contiguous()).sum()
            union = (enweight * input[:, i, ::].contiguous()).sum() + (enweight * target[:, i, ::].contiguous()).sum()

            dice = (2.0 * intersection + eps) / (union + eps)

            diceLoss += (1 - dice)
    else:
        intersection = (enweight * input[:, 0, ::].contiguous() * target[:, 0, ::].contiguous()).sum()
        union = (enweight * input[:, 0, ::].contiguous()).sum() + (enweight * target[:, 0, ::].contiguous()).sum()

        dice = (2.0 * intersection + eps) / (union + eps)

        diceLoss += (1 - dice)


    return diceLoss / C


def multiDiceLossCon(input, label, gpu_id):
    # C = 5
    # diceLoss = 0.
    B, C, H, W = input.shape
    label = label.long()
    diceLoss = 0.
    if C > 1:
        target = np.zeros((C, label.shape[0], label.shape[1], label.shape[2]))  # 5,4,256,256
        train_gt = np.array(label.cpu())

        for i in range(C):
            target[i] = np.array((train_gt == i), dtype=np.uint8)
        target = target.transpose(1, 0, 2, 3)
        target = torch.FloatTensor(target).cuda('cuda:{}'.format(gpu_id[0]))
    if C > 1:
        for i in range(C):
            diceLoss += (1 + mt_losses.dice_loss(input[:, i, ::].contiguous(), target[:, i, ::].contiguous()))
    else:
        diceLoss += (1 + mt_losses.dice_loss(input[:, 0, ::].contiguous(), label[:, 0, ::].contiguous()))

    return diceLoss / C


def multiDiceLossC(input, label):
    C = 5
    diceLoss = 0.
    target = np.zeros((C, label.shape[0], label.shape[1], label.shape[2]))  # 5,4,256,256
    train_gt = np.array(label.cpu())

    for i in range(C):
        target[i] = np.array((train_gt == i), dtype=np.int32)
    target = target.transpose(1, 0, 2, 3)
    target = torch.FloatTensor(target).cuda()

    if input.shape == target.shape:
        for i in range(1, C):
            diceLoss += 1 + mt_losses.dice_loss(input[:, i, ::].contiguous(), target[:, i, ::].contiguous())
    else:
        print("input.shape != target.shape")
    return diceLoss / (C - 1)


def multiCELossL(input, label, gpu_id):
    B, C, H, W = input.shape
    # label = label.long()

    if C > 1:
        label = label.long()
        loss = F.cross_entropy(
            input, label, weight=None, size_average=True)
    else:
        loss = F.binary_cross_entropy(input, label)
    return loss


class CrossEntropy2d(nn.Module):
    def __init__(self, size_average=True, temperature=None, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.temperature = temperature
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(
            0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(
            1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(
            2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if self.temperature:
            target = target / self.temperature
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(
            n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(
            predict, target, weight=weight, size_average=self.size_average)
        return loss


def loss_calc(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    label = label.long().cuda()
    criterion = CrossEntropy2d().cuda()
    return criterion(pred, label)




