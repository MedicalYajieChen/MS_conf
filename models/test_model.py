from .base_model import BaseModel
from . import networks_seg as networks
import torch
import numpy as np
import torch.nn.functional as F
import os
import math
import time


class TestModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        parser.set_defaults(no_dropout=True,norm='instance', netG='resnet_9blocks',no_lsgan=True)
        assert not is_train, 'TestModel cannot be used during training time'
        # parser.set_defaults(dataset_mode='single')
        # parser.add_argument('--model_suffix', type=str, default='_A', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        assert(not opt.isTrain)
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.visual_names = ['target_seg3' , 'target_label','target_entropy_u2', 'target_entropy_u1',\
                             'target_entropy_seg3','target']
        # self.visual_names = ['target', 'target_label']

        self.opt = opt

        self.model_names = ['U1', 'U2']

        self.gpu_ids = opt.gpu_ids
        if 'brats' in self.opt.dataset_mode:
            self.netU1 = networks.define_UNet(3, 1, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)
            self.netU2 = networks.define_UNet(3, 1, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)
        else:
            self.netU1 = networks.define_UNet(3,5,init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)
            self.netU2 = networks.define_UNet(3, 5, init_gain=opt.init_gain, gpu_ids=opt.gpu_ids)



    def set_input(self, input):
        self.target = input['target'].to(self.device)
        self.target_label = input['target_label'].to(self.device)
        self.seg_path = input['seg_path']
        self.image_paths = input['target_path']

    def eval(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.netU1.eval()
        self.netU2.eval()

        self.targets_u1 = []
        self.targets_label = []
        self.targets_u2 = []
        self.targets_avg = []
        self.targets_seg3 = []
        self.targets_entropy_avg = []
        self.targets_entropy_u2 = []
        self.dices_u1 = []
        self.dices_u2 = []
        self.dices_seg3 = []
        self.dices_avg = []
        self.threshs = []
        self.precisions = []

        self.distances_u1 = []
        self.distances_u2 = []
        self.distances_seg3 = []
        self.distances_avg = []
        self.time_avg =[]
        self.time_cgf = []

    def test(self):
        self.pred_target1 = self.netU1(self.target)
        self.pred_target2 = self.netU2(self.target)
        if 'brats' in self.opt.dataset_mode:
            self.target_u1 = (torch.sigmoid(self.pred_target1)[:, 0, :, :] > 0.5).cpu().numpy()
            self.target_u2 = (torch.sigmoid(self.pred_target2)[:, 0, :, :] > 0.5).cpu().numpy()
            self.target_p1 = torch.sigmoid(self.pred_target1)
            self.target_p2 = torch.sigmoid(self.pred_target2)
            start = time.time()
            self.target_avg = (
            ((self.target_p1+self.target_p2) / 2.0))
            inf_time = time.time()-start
            self.time_avg.append(inf_time)
            # print('avg:{}'.format(self.target_avg.max()))
            self.target_entropy_u1_raw = (-torch.log(self.target_p1)*self.target_p1)
            self.target_entropy_u2_raw = (-torch.log(self.target_p2) * self.target_p2)
            self.target_entropy_u1 = 2.0*torch.sigmoid(-self.target_entropy_u1_raw).cpu().numpy()[:,0,:,:]
            self.target_entropy_u2 = 2.0*torch.sigmoid(-self.target_entropy_u2_raw).cpu().numpy()[:,0,:,:]
            self.target_entropy_avg = (torch.log(self.target_avg)*self.target_avg)
            # self.target_entropy_avg = torch.abs(self.target_avg - (self.target_avg>=0.5).long())
            self.target_entropy_avg = 10.0*(2.0*torch.sigmoid(2.0*self.target_entropy_avg)-0.9).cpu().numpy()[:,0,:,:]
            self.target_avg = self.target_avg[:, 0, :, :].cpu().numpy()
            self.target_avg = np.array(self.target_avg>=0.5, dtype=np.uint8)
        else:
            self.target_u1 = (torch.softmax(self.pred_target1, 1).max(1)[1]).cpu().numpy()
            self.target_u2 = (torch.softmax(self.pred_target2, 1).max(1)[1]).cpu().numpy()
            self.target_p1 = torch.softmax(self.pred_target1, 1)
            self.target_p2 = torch.softmax(self.pred_target2, 1)

            # print(self.target_avg.shape)
            start = time.time()
            self.target_entropy_u1_raw = torch.sum(-torch.log(self.target_p1) * self.target_p1, dim=1)
            self.target_entropy_u2_raw = torch.sum(-torch.log(self.target_p2) * self.target_p2, dim=1)
            target_seg3 = self.target_p1.max(1)[1]*((self.target_entropy_u1_raw < self.target_entropy_u2_raw)).long()+ \
                          self.target_p2.max(1)[1]*((self.target_entropy_u2_raw <= self.target_entropy_u1_raw)).long()
            inf_time = time.time() - start
            self.time_cgf.append(inf_time)

            start = time.time()
            self.target_avg = (
                ((self.target_p1 + self.target_p2) / 2.0))
            target_avg = self.target_avg.max(1)[1]
            inf_time = time.time() - start
            self.time_avg.append(inf_time)
            self.target_entropy_u1 = (2.0*torch.sigmoid(-self.target_entropy_u1_raw)).cpu().numpy()
            self.target_entropy_u2 = (2.0*torch.sigmoid(-self.target_entropy_u2_raw)).cpu().numpy()
            # self.target_entropy_u2 = (self.target_p2.max(1)[0]-0.05).cpu().numpy()
            self.target_entropy_avg = torch.sum(torch.log(self.target_avg) * self.target_avg, dim=1)
            # self.target_entropy_avg = 50.0*(2.0*torch.sigmoid(5.0*self.target_entropy_avg)-0.98).cpu().numpy()
            # self.target_avg = np.array((self.target_avg.max(1)[1]).cpu().numpy(), dtype=np.uint8)
            self.target_avg = self.target_avg.max(1)[1]
            self.target_avg = self.target_avg.cpu().numpy()
            # print('max:',self.target_avg.max())
        self.target_seg3 = (self.target_entropy_u1<self.target_entropy_u2).astype(np.uint8)*self.target_u1+ \
                           (self.target_entropy_u2 <= self.target_entropy_u1).astype(np.uint8) * self.target_u2
        self.target_entropy_seg3 = ((self.target_entropy_u1_raw<self.target_entropy_u2_raw).cpu().numpy()).astype(np.uint8)*self.target_entropy_u1+ \
                                   ((self.target_entropy_u2_raw <= self.target_entropy_u1_raw).cpu().numpy()).astype(np.uint8) * self.target_entropy_u2
        self.target_entropy_seg3 = 2.0*(1/(1.0+np.exp(self.target_entropy_seg3)))
        self.targets_u1.append(self.target_u1)
        self.targets_u2.append(self.target_u2)
        self.targets_label.append(self.target_label.cpu().numpy())
        self.targets_seg3.append(self.target_seg3)
        self.targets_avg.append(self.target_avg)
        self.target = self.target.cpu().numpy()
        self.target_label = self.target_label.cpu().numpy()
        self.targets_entropy_avg.append(self.target_entropy_avg)
        self.targets_entropy_u2.append(self.target_entropy_u2)

    def forward(self):
        return


    def test_evaluation(self):
        self.targets_u1 = np.concatenate(self.targets_u1, axis=0)
        self.targets_u2 = np.concatenate(self.targets_u2, axis=0)
        self.targets_seg3 = np.concatenate(self.targets_seg3, axis=0)
        self.targets_avg = np.concatenate(self.targets_avg, axis=0)
        self.targets_label = np.concatenate(self.targets_label, axis=0)
        if 'abdo' in self.opt.dataset_mode:
            i = 0
            for slice in self.opt.case_slice:
                target_pred_u1 = self.targets_u1[i :i+slice, :, :]
                target_pred_u2 = self.targets_u2[i:i + slice, :, :]
                target_pred_seg3 = self.targets_seg3[i:i + slice, :, :]
                target_pred_avg = self.targets_avg[i:i + slice, :, :]
                target_label = self.targets_label[i :i+slice, :, :]
                self.dices_u1.append(networks.calculate_dice(target_pred_u1, target_label))
                self.dices_u2.append(networks.calculate_dice(target_pred_u2, target_label))
                self.dices_seg3.append(networks.calculate_dice(target_pred_seg3, target_label))
                self.dices_avg.append(networks.calculate_dice(target_pred_avg, target_label))

                self.distances_u1.append(networks.calculate_avg_distance(target_pred_u1, target_label))
                self.distances_u2.append(networks.calculate_avg_distance(target_pred_u2, target_label))
                self.distances_seg3.append(networks.calculate_avg_distance(target_pred_seg3, target_label))
                self.distances_avg.append(networks.calculate_avg_distance(target_pred_avg, target_label))
                i+= slice
        else:
            for i in range(int(len(self.targets_u1)/self.opt.num_slice)):
                target_pred_u1 = self.targets_u1[i*self.opt.num_slice:(i+1)*self.opt.num_slice,:,:]
                target_pred_u2 = self.targets_u2[i * self.opt.num_slice:(i + 1) * self.opt.num_slice, :, :]
                target_pred_seg3 = self.targets_seg3[i * self.opt.num_slice:(i + 1) * self.opt.num_slice, :, :]
                target_pred_avg = self.targets_avg[i * self.opt.num_slice:(i + 1) * self.opt.num_slice, :, :]
                target_label = self.targets_label[i * self.opt.num_slice:(i + 1) * self.opt.num_slice, :, :]
                if 'brats' in self.opt.dataset_mode:
                    self.dices_u1.append(networks.calculate_dice_binary(target_pred_u1, target_label))
                    self.dices_u2.append(networks.calculate_dice_binary(target_pred_u2, target_label))
                    self.dices_seg3.append(networks.calculate_dice_binary(target_pred_seg3, target_label))
                    self.dices_avg.append(networks.calculate_dice_binary(target_pred_avg, target_label))

                    self.distances_u1.append(networks.calculate_distance_binary(target_pred_u1, target_label))
                    self.distances_u2.append(networks.calculate_distance_binary(target_pred_u2, target_label))
                    self.distances_seg3.append(networks.calculate_distance_binary(target_pred_seg3, target_label))
                    self.distances_avg.append(networks.calculate_distance_binary(target_pred_avg, target_label))
                else:
                    self.dices_u1.append(networks.calculate_dice(target_pred_u1, target_label))
                    self.dices_u2.append(networks.calculate_dice(target_pred_u2, target_label))
                    self.dices_seg3.append(networks.calculate_dice(target_pred_seg3, target_label))
                    self.dices_avg.append(networks.calculate_dice(target_pred_avg, target_label))

                    self.distances_u1.append(networks.calculate_avg_distance(target_pred_u1, target_label))
                    self.distances_u2.append(networks.calculate_avg_distance(target_pred_u2, target_label))
                    self.distances_seg3.append(networks.calculate_avg_distance(target_pred_seg3, target_label))
                    self.distances_avg.append(networks.calculate_avg_distance(target_pred_avg, target_label))
        self.dices_u1 = np.array(self.dices_u1)
        self.dices_u2 = np.array(self.dices_u2)
        self.dices_seg3 = np.array(self.dices_seg3)
        self.dices_avg = np.array(self.dices_avg)
        self.distances_u1 = np.array(self.distances_u1)
        self.distances_u2 = np.array(self.distances_u2)
        self.distances_seg3 = np.array(self.distances_seg3)
        self.distances_avg = np.array(self.distances_avg)
        w_str1 = ''
        w_str2 = ''
        w_str3 = ''
        w_str4 = ''

        a_str1 = ''
        a_str2 = ''
        a_str3 = ''
        a_str4 = ''
        if 'unalignwhs' in self.opt.dataset_mode:
            w_str1 = 'target U1 :MYO:{:.3f},LAC:{:.3f},LVC:{:.3f},AA:{:.3f},Average:{:.3f}\n'.format(self.dices_u1[:,1].mean(),\
                        self.dices_u1[:,2].mean(),self.dices_u1[:,3].mean(),self.dices_u1[:,4].mean(),\
                                self.dices_u1[:,1:].mean())
            w_str2 = 'target U2 :MYO:{:.3f},LAC:{:.3f},LVC:{:.3f},AA:{:.3f},Average:{:.3f}\n'.format(
                    self.dices_u2[:,1].mean(), self.dices_u2[:,2].mean(),self.dices_u2[:,3].mean(),\
                self.dices_u2[:,4].mean(), self.dices_u2[:,1:].mean())
            w_str3 = 'target Seg3 :MYO:{:.3f},LAC:{:.3f},LVC:{:.3f},AA:{:.3f},Average:{:.3f}\n'.format(
                        self.dices_seg3[:, 1].mean(), self.dices_seg3[:, 2].mean(),self.dices_seg3[:, 3].mean(),\
                                self.dices_seg3[:, 4].mean(), self.dices_seg3[:,1:].mean())
            w_str4 = 'target Avg :MYO:{:.3f},LAC:{:.3f},LVC:{:.3f},AA:{:.3f},Average:{:.3f}\n'.format(
                self.dices_avg[:, 1].mean(), self.dices_avg[:, 2].mean(), self.dices_avg[:, 3].mean(), \
                self.dices_avg[:, 4].mean(), self.dices_avg[:, 1:].mean())

            a_str1 = 'distance U1 :MYO:{:.3f},LAC:{:.3f},LVC:{:.3f},AA:{:.3f},Average:{:.3f}\n'.format(
                self.distances_u1[:, 1].mean(), \
                self.distances_u1[:, 2].mean(), self.distances_u1[:, 3].mean(), self.distances_u1[:, 4].mean(), \
                self.distances_u1[:, 1:].mean())
            a_str2 = 'distance U2 :MYO:{:.3f},LAC:{:.3f},LVC:{:.3f},AA:{:.3f},Average:{:.3f}\n'.format(
                self.distances_u2[:, 1].mean(), self.distances_u2[:, 2].mean(), self.distances_u2[:, 3].mean(), \
                self.distances_u2[:, 4].mean(), self.distances_u2[:, 1:].mean())
            a_str3 = 'distance Seg3 :MYO:{:.3f},LAC:{:.3f},LVC:{:.3f},AA:{:.3f},Average:{:.3f}\n'.format(
                self.distances_seg3[:, 1].mean(), self.distances_seg3[:, 2].mean(), self.distances_seg3[:, 3].mean(), \
                self.distances_seg3[:, 4].mean(), self.distances_seg3[:, 1:].mean())
            a_str4 = 'distance Avg :MYO:{:.3f},LAC:{:.3f},LVC:{:.3f},AA:{:.3f},Average:{:.3f}\n'.format(
                self.distances_avg[:, 1].mean(), self.distances_avg[:, 2].mean(), self.distances_avg[:, 3].mean(), \
                self.distances_avg[:, 4].mean(), self.distances_avg[:, 1:].mean())
        elif 'unalignabdo' in self.opt.dataset_mode:
            w_str1 = 'target U1:Liver:{:.3f},LKid:{:.3f},RKid:{:.3f},Spleen:{:.3f},Average:{:.3f}\n'.format(self.dices_u1[:,1].mean(),\
                        self.dices_u1[:,2].mean(),self.dices_u1[:,3].mean(),self.dices_u1[:,4].mean(),\
                                self.dices_u1[:,1:].mean())
            w_str2 = 'target U2:Liver:{:.3f},LKid:{:.3f},RKid:{:.3f},Spleen:{:.3f},Average:{:.3f}\n'.format(
                        self.dices_u2[:,1].mean(), self.dices_u2[:,2].mean(), self.dices_u2[:,3].mean(),self.dices_u2[:,4].mean(), \
                            self.dices_u2[:,1:].mean())
            w_str3 = 'target Seg3:Liver:{:.3f},LKid:{:.3f},RKid:{:.3f},Spleen:{:.3f},Average:{:.3f}\n'.format(
                        self.dices_seg3[:,1].mean(),self.dices_seg3[:,2].mean(),self.dices_seg3[:,3].mean(),self.dices_seg3[:,4].mean(), \
                            self.dices_seg3[:,1:].mean())
            w_str4 = 'target Avg:Liver:{:.3f},LKid:{:.3f},RKid:{:.3f},Spleen:{:.3f},Average:{:.3f}\n'.format(
                self.dices_avg[:, 1].mean(), self.dices_avg[:, 2].mean(), self.dices_avg[:, 3].mean(),
                self.dices_avg[:, 4].mean(), \
                self.dices_avg[:, 1:].mean())

            a_str1 = 'distance U1:Liver:{:.3f},LKid:{:.3f},RKid:{:.3f},Spleen:{:.3f},Average:{:.3f}\n'.format(
                self.distances_u1[:, 1].mean(), \
                self.distances_u1[:, 2].mean(), self.distances_u1[:, 3].mean(), self.distances_u1[:, 4].mean(), \
                self.distances_u1[:, 1:].mean())
            a_str2 = 'distance U2:Liver:{:.3f},LKid:{:.3f},RKid:{:.3f},Spleen:{:.3f},Average:{:.3f}\n'.format(
                self.distances_u2[:, 1].mean(), self.distances_u2[:, 2].mean(), self.distances_u2[:, 3].mean(),
                self.distances_u2[:, 4].mean(), \
                self.distances_u2[:, 1:].mean())
            a_str3 = 'distance Seg3:Liver:{:.3f},LKid:{:.3f},RKid:{:.3f},Spleen:{:.3f},Average:{:.3f}\n'.format(
                self.distances_seg3[:, 1].mean(), self.distances_seg3[:, 2].mean(), self.distances_seg3[:, 3].mean(),
                self.distances_seg3[:, 4].mean(), \
                self.distances_seg3[:, 1:].mean())
            a_str4 = 'distance Avg:Liver:{:.3f},LKid:{:.3f},RKid:{:.3f},Spleen:{:.3f},Average:{:.3f}\n'.format(
                self.distances_avg[:, 1].mean(), self.distances_avg[:, 2].mean(), self.distances_avg[:, 3].mean(),
                self.distances_avg[:, 4].mean(), \
                self.distances_avg[:, 1:].mean())
        else:
            w_str1 = 'target U1:Tumer:{:.3f},Average:{:.3f}\n'.format(self.dices_u1[:].mean(),\
                        self.dices_u1[:].mean())
            w_str2 = 'target U2:Tumer:{:.3f},Average:{:.3f}\n'.format( self.dices_u2[:].mean(), \
                                                                            self.dices_u2[:].mean())
            w_str3 = 'target Seg3:Tumer:{:.3f},Average:{:.3f}\n'.format(self.dices_seg3[:].mean(), \
                                                                            self.dices_seg3[:].mean())
            w_str4 = 'target Avg:Tumer:{:.3f},Average:{:.3f}\n'.format(self.dices_avg[:].mean(), \
                                                                    self.dices_avg[:].mean())

            a_str1 = 'distance U1:Tumer:{:.3f},Average:{:.3f}\n'.format(self.distances_u1[:].mean(), \
                                                                      self.distances_u1[:].mean())
            a_str2 = 'distance U2:Tumer:{:.3f},Average:{:.3f}\n'.format(self.distances_u2[:].mean(), \
                                                                      self.distances_u2[:].mean())
            a_str3 = 'distance Seg3:Tumer:{:.3f},Average:{:.3f}\n'.format(self.distances_seg3[:].mean(), \
                                                                        self.distances_seg3[:].mean())
            a_str4 = 'distance Avg:Tumer:{:.3f},Average:{:.3f}\n'.format(self.distances_avg[:].mean(), \
                                                                       self.distances_avg[:].mean())

        with open(os.path.join(self.opt.log_dir, self.opt.name+'.txt'),'a+') as f:
            f.write(w_str1)
            f.write(w_str2)
            f.write(w_str3)
            f.write(w_str4)
            f.write(a_str1)
            f.write(a_str2)
            f.write(a_str3)
            f.write(a_str4)
        print(w_str1,w_str2,w_str3, w_str4,a_str1,a_str2,a_str3, a_str4)

        return

    def precision(self):
        # self.targets_avg = np.concatenate(self.targets_avg, axis=0)
        # self.targets_label = np.concatenate(self.targets_label, axis=0)
        # self.targets_entropy_avg = np.concatenate(self.targets_entropy_avg, axis=0)
        # for thresh in sorted(self.opt.thresholds):
        #     self.dices_avg = []
        #     self.threshs.append(thresh)
        #     if 'abdo' in self.opt.dataset_mode:
        #         i = 0
        #         for slice in self.opt.case_slice:
        #             target_pred_avg = self.targets_avg[i:i + slice, :, :]
        #             target_label = self.targets_label[i :i+slice, :, :]
        #             target_entropy_avg = self.targets_entropy_avg[i:i + slice, :, :]
        #             i+= slice
        #             # print('max avg:{}'.format(target_pred_avg.shape))
        #
        #             self.dices_avg.append(networks.calculate_precision(target_pred_avg, target_label, target_entropy_avg,thresh))
        #     else:
        #         for i in range(int(len(self.targets_avg)/self.opt.num_slice)):
        #             target_pred_avg = self.targets_avg[i * self.opt.num_slice:(i + 1) * self.opt.num_slice, :, :]
        #             target_label = self.targets_label[i * self.opt.num_slice:(i + 1) * self.opt.num_slice, :, :]
        #             target_entropy_avg = self.targets_entropy_avg[i * self.opt.num_slice:(i + 1) * self.opt.num_slice, :, :]
        #             print('mean:{}'.format(target_entropy_avg.mean()))
        #             # print('max:{}'.format(target_entropy_avg.shape))
        #             # print('max avg:{}'.format(target_pred_avg.max()))
        #             if 'brats' in self.opt.dataset_mode:
        #                 self.dices_avg.append(networks.calculate_precision_binary(target_pred_avg, target_label, target_entropy_avg,thresh))
        #             else:
        #                 self.dices_avg.append(networks.calculate_precision(target_pred_avg, target_label, target_entropy_avg, thresh))
        #     self.dices_avg = np.array(self.dices_avg)
        time_avg = np.array(self.time_avg).mean()
        time_cgf = np.array(self.time_cgf).mean()
        w_str4 = ''

            # if 'unalignwhs' in self.opt.dataset_mode:
            #     self.precisions.append(np.nanmean(self.dices_avg[:, :]))
            #     w_str4 = 'target Avg thresh:{:.3f}: Average:{}\n'.format( thresh,
            #         np.nanmean(self.dices_avg[:, :]))
            # elif 'unalignabdo' in self.opt.dataset_mode:
            #     self.precisions.append(np.nanmean(self.dices_avg[:, :]))
            #     w_str4 = 'target Avg thresh:{:.3f}: Average:{}\n'.format( thresh,
            #         np.nanmean(self.dices_avg[:, :]))
            # else:
            # self.precisions.append(np.nanmean(self.dices_avg[:]))
            # w_str4 = 'target Avg thresh:{:.3f}: Average:{}, Avg Time:{}, CGF Time:{}\n'.format(thresh,
            #                 np.nanmean(self.dices_avg[:]), time_avg, time_cgf)

        w_str4 = 'target Avg : Avg Time:{}, CGF Time:{}\n'.format(\
                                                                                   time_avg, time_cgf)

        with open(os.path.join(self.opt.log_dir, self.opt.name+'_precision.txt'),'a+') as f:
            f.write(w_str4)
            # f.write('threshs:{}\n'.format(self.threshs))
            # f.write('precisions:{}\n'.format(self.precisions))
        print(w_str4)

        # with open(os.path.join(self.opt.log_dir, self.opt.name + '_precision.txt'), 'a+') as f:
        #     # f.write(w_str4)
        #     f.write('threshs:{}\n'.format(self.threshs))
        #     f.write('precisions:{}\n'.format(self.precisions))
        # print(w_str4)

        return

    def optimize_parameters(self):
        """No optimization for test model."""
        pass

    def get_image_paths(self):
        return self.image_paths

