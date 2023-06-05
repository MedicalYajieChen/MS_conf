import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
# from . import networks
from . import networks_seg as networks
from torch.nn import functional as F
import math
import numpy as np
import os


def get_current_consistency_weight(weight, epoch, num_epoch):
    """Consistency ramp-up from https://arxiv.org/abs/1610.02242"""
    return weight * math.exp(-5.0*(1.0-1.0*epoch/num_epoch))


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def threshold_predictions(predictions, thr=0.95):
    thresholded_preds = predictions[:]
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds

def genrate_pseudo(predictions):
    # thresholded_preds = predictions[:]
    # low_values_indices = thresholded_preds < thr
    # thresholded_preds[low_values_indices] = 0
    # low_values_indices = thresholded_preds >= thr
    # thresholded_preds[low_values_indices] = 1
    # print(predictions.shape[1])
    if predictions.shape[1] > 1:
        return torch.softmax(predictions, 1).max(1)[1]
    else:
        return (torch.sigmoid(predictions)>0.5).long()


class UNetCpsModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        # parser.set_defaults()  # default CycleGAN did not use dropout
        parser.set_defaults(no_dropout=True,norm='instance', netG='resnet_6blocks',netD='n_layers',n_layers_D=2,no_lsgan=False)
        if is_train:# 10 times
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            # parser.add_argument('--lambda_identity', type=float, default=1, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

        return parser

    def get_current_consistency_weight(self, epoch, num_epoch):
        """Consistency ramp-up from https://arxiv.org/abs/1610.02242"""
        self.consistencyweight = 1.0* math.exp(-5.0 * (1.0 - 1.0 * epoch / num_epoch))

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['U_cross','U_dice','U_cps']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # visual_names_A = ['real_A', 'fake_B', 'rec_A']
        # visual_names_B = ['real_B', 'fake_A', 'rec_B']
        # self.visual_names = ['target', 'target_label', 'pred_target_val','target_entropy_u1','target_entropy_u2']
        self.visual_names = ['target', 'target_label']

        self.opt = opt
        self.best = 0.0

        if self.isTrain:
            self.model_names = ['U1', 'U2']
        else:  # during test time, only load Gs
            self.model_names = ['U2']
        self.gpu_ids = opt.gpu_ids
        self.consistencyweight = 1.0
        if 'brats' in self.opt.dataset_mode:
            self.netU1 = networks.define_UNet(3, 1, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)
            self.netU2 = networks.define_UNet(3, 1, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)
        else:
            self.netU1 = networks.define_UNet(3,5,init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)
            self.netU2 = networks.define_UNet(3, 5, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)
        self.optimizer_U = torch.optim.Adam(itertools.chain(self.netU1.parameters(), self.netU2.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

        # self.optimizers.append(self.optimizer_U)



    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        # AtoB = self.opt.direction == 'AtoB'
        # self.real_A = input['A' if AtoB else 'B'].to(self.device)
        # self.real_B = input['B' if AtoB else 'A'].to(self.device)
        # self.source_label = input['A_label' if AtoB else 'B_label'].to(self.device)
        # self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.source = input['source'].to(self.device)
        self.source_label = input['source_label'].to(self.device)
        self.target = input['target'].to(self.device)
        self.target_label = input['target_label'].to(self.device)
        # self.seg_path = input['seg_pa]

    def set_input_val(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        self.target = input['target'].to(self.device)
        self.target_label = input['target_label'].to(self.device)
        self.seg_path = input['seg_path']

    def train(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.netU1.train()
        self.netU2.train()

        self.targets = []
        self.targets_label = []
        self.dices = []

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        self.pred_source1 = self.netU1(self.source)
        self.pred_source2 = self.netU2(self.source)
        self.pred_target1 = self.netU1(self.target)
        self.pred_target2 = self.netU2(self.target)


    def validate(self):

        self.pred_target2 = self.netU2(self.target)
        if 'brats' in self.opt.dataset_mode:
            self.targets.append((torch.sigmoid(self.pred_target2)[:,0,:,:]>0.5).cpu().numpy().astype(np.uint8))
        else:
            self.targets.append((torch.softmax(self.pred_target2, 1).max(1)[1]).cpu().numpy())
        self.targets_label.append(self.target_label.cpu().numpy())


    def eval(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.netU1.eval()
        self.netU2.eval()

        self.targets = []
        self.targets_label = []
        self.dices = []


    def evaluation(self,epoch):
        self.targets = np.concatenate(self.targets, axis=0)
        self.targets_label = np.concatenate(self.targets_label, axis=0)
        if 'abdo' in self.opt.dataset_mode:
            i = 0
            for slice in self.opt.case_slice:
                target_pred = self.targets[i :i+slice, :, :]
                target_label = self.targets_label[i :i+slice, :, :]
                self.dices.append(networks.calculate_dice(target_pred, target_label))
        else:
            for i in range(int(len(self.targets)/self.opt.num_slice)):
                target_pred = self.targets[i*self.opt.num_slice:(i+1)*self.opt.num_slice,:,:]
                target_label = self.targets_label[i * self.opt.num_slice:(i + 1) * self.opt.num_slice, :, :]
                if 'brats' in self.opt.dataset_mode:
                    self.dices.append(networks.calculate_dice_binary(target_pred, target_label))
                else:
                    self.dices.append(networks.calculate_dice(target_pred, target_label))
        self.dices = np.array(self.dices)
        w_str = ''
        val = 0.0
        if 'unalignwhs' in self.opt.dataset_mode:
            w_str = 'target epoch {}:MYO:{:.3f},LAC:{:.3f},LVC:{:.3f},AA:{:.3f},Average:{:.3f}\n'.format(epoch, self.dices[:,1].mean(),\
                        self.dices[:,2].mean(),self.dices[:,3].mean(),self.dices[:,4].mean(),\
                                self.dices[:,1:].mean())
            val = self.dices[:,1:].mean()
        elif 'unalignabdo' in self.opt.dataset_mode:
            w_str = 'target epoch {}:Liver:{:.3f},LKid:{:.3f},RKid:{:.3f},Spleen:{:.3f},Average:{:.3f}\n'.format(epoch, self.dices[:,1].mean(),\
                        self.dices[:,2].mean(),self.dices[:,3].mean(),self.dices[:,4].mean(),\
                                self.dices[:,1:].mean())
            val = self.dices[:, 1:].mean()
        else:
            w_str = 'target epoch {}:Tumer:{:.3f},Average:{:.3f}\n'.format(epoch, self.dices[:].mean(),\
                        self.dices[:].mean())
            val = self.dices[:].mean()
        with open(os.path.join(self.opt.log_dir, self.opt.name+'.txt'),'a+') as f:
            f.write(w_str)
        print(w_str)
        is_best = False
        if val > self.best:
            self.best = val
            is_best = True
        return is_best


    def backward_U(self):
        if 'brats' in self.opt.dataset_mode:
            self.loss_U_cross = networks.multiCELossL(torch.sigmoid(self.pred_source1), self.source_label, self.gpu_ids)
            self.loss_U_dice = networks.multiDiceLossL(torch.sigmoid(self.pred_source1), self.source_label, self.gpu_ids)

            self.loss_U_cross += networks.multiCELossL(torch.sigmoid(self.pred_source2), self.source_label, self.gpu_ids)
            self.loss_U_dice += networks.multiDiceLossL(torch.sigmoid(self.pred_source2), self.source_label,
                                                        self.gpu_ids)

        else:
            self.loss_U_cross = networks.multiCELossL(self.pred_source1, self.source_label, self.gpu_ids)
            self.loss_U_dice = networks.multiDiceLossL(F.softmax(self.pred_source1, 1), self.source_label, self.gpu_ids)

            self.loss_U_cross += networks.multiCELossL(self.pred_source2, self.source_label, self.gpu_ids)
            self.loss_U_dice += networks.multiDiceLossL(F.softmax(self.pred_source2, 1), self.source_label, self.gpu_ids)


        if self.opt.no_adaptation == 0:
            self.pred_target1_ema = genrate_pseudo(self.pred_target1)
            self.pred_target2_ema = genrate_pseudo(self.pred_target2)
            # self.pred_target2_ema = self.pred_target1 * self.pred_target2_ema
            if 'brats' in self.opt.dataset_mode:
                self.loss_U_cps = self.consistencyweight * networks.multiDiceLossCon(
                    torch.sigmoid(self.pred_target1), self.pred_target2_ema, self.gpu_ids) + \
                                  self.consistencyweight * networks.multiDiceLossCon(
                    torch.sigmoid(self.pred_target2), self.pred_target1_ema,
                    self.gpu_ids)
            else:
                self.loss_U_cps = self.consistencyweight * networks.multiDiceLossCon(torch.softmax(self.pred_target1, 1), self.pred_target2_ema, self.gpu_ids)+ \
                                  self.consistencyweight * networks.multiDiceLossCon(torch.softmax(self.pred_target2, 1), self.pred_target1_ema,
                                                                             self.gpu_ids)
        else:
            self.loss_U_cps = 0.0

        # combined loss and calculate gradients
        self.loss_U = self.loss_U_cross + self.loss_U_dice + self.loss_U_cps
        self.loss_U.backward()


    def backward_U_weight(self):
        if 'brats' in self.opt.dataset_mode:
            self.loss_U_cross = networks.multiCELossL(torch.sigmoid(self.pred_source1), self.source_label, self.gpu_ids)
            self.loss_U_dice = networks.multiDiceLossL(torch.sigmoid(self.pred_source1), self.source_label, self.gpu_ids)

            self.loss_U_cross += networks.multiCELossL(torch.sigmoid(self.pred_source2), self.source_label, self.gpu_ids)
            self.loss_U_dice += networks.multiDiceLossL(torch.sigmoid(self.pred_source2), self.source_label,
                                                        self.gpu_ids)

        else:
            self.loss_U_cross = networks.multiCELossL(self.pred_source1, self.source_label, self.gpu_ids)
            self.loss_U_dice = networks.multiDiceLossL(F.softmax(self.pred_source1, 1), self.source_label, self.gpu_ids)

            self.loss_U_cross += networks.multiCELossL(self.pred_source2, self.source_label, self.gpu_ids)
            self.loss_U_dice += networks.multiDiceLossL(F.softmax(self.pred_source2, 1), self.source_label, self.gpu_ids)

        if self.opt.no_adaptation == 0:
            self.pred_target1_ema = genrate_pseudo(self.pred_target1)
            self.pred_target2_ema = genrate_pseudo(self.pred_target2)
            # self.pred_target2_ema = self.pred_target1 * self.pred_target2_ema
            if 'brats' in self.opt.dataset_mode:
                self.loss_U_cps = self.consistencyweight * networks.multiDiceLossConWeight(
                    torch.sigmoid(self.pred_target1), torch.sigmoid(self.pred_target2), self.pred_target2_ema, self.gpu_ids) + \
                                  self.consistencyweight * networks.multiDiceLossConWeight(
                    torch.sigmoid(self.pred_target2), torch.sigmoid(self.pred_target1), self.pred_target1_ema,
                    self.gpu_ids)
            else:
                self.loss_U_cps = self.consistencyweight * networks.multiDiceLossConWeight(torch.softmax(self.pred_target1, 1),torch.softmax(self.pred_target2, 1), self.pred_target2_ema, self.gpu_ids)+ \
                                  self.consistencyweight * networks.multiDiceLossConWeight(torch.softmax(self.pred_target2, 1),torch.softmax(self.pred_target1, 1), self.pred_target1_ema,
                                                                             self.gpu_ids)

        else:
            self.loss_U_cps = 0.0

        # combined loss and calculate gradients
        self.loss_U = self.loss_U_cross + self.loss_U_dice + self.loss_U_cps
        self.loss_U.backward()


    def optimize_parameters(self,epoch,epoch_subsample):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netU1], True)
        self.set_requires_grad([self.netU2], True)
        self.optimizer_U.zero_grad()

        # consistency_weight = self.get_current_consistency_weight(1.0, epoch, self.opt.num_epoch)

        if epoch <= epoch_subsample:
            self.backward_U()
            torch.nn.utils.clip_grad_norm_(parameters=itertools.chain(self.netU1.parameters(), \
                                              self.netU2.parameters()), max_norm=10, norm_type=2)
            self.optimizer_U.step()
        else:
            self.backward_U_weight()
            torch.nn.utils.clip_grad_norm_(parameters=itertools.chain(self.netU1.parameters(), \
                                                self.netU2.parameters()), max_norm=10,
                                           norm_type=2)
            self.optimizer_U.step()


        # for param_group in self.optimizer_U.param_groups:
        #     param_group['lr'] = self.opt.lr
