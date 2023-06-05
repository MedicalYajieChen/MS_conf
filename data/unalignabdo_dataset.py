import os.path
from data.base_dataset import BaseDataset#, get_transform
from PIL import Image
import random
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
import albumentations as A


def get_transform():
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.pgm','.PGM', '.npy'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class UnalignAbdoDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.phase = opt.phase
        self.valid_classes = [0, 63, 127, 191, 255]
        self.train_classes = [0, 1, 2, 3, 4]

        self.class_map = dict(zip(self.valid_classes, self.train_classes))

        # self.source_dir = os.path.join(opt.dataroot, "t22ct_"+opt.data_input+"_seg_data_ep_new"+opt.select_epoch) \
        #     if opt.direction == "BtoA" else os.path.join(opt.dataroot, "ct2t2_"+opt.data_input+"_seg_data_ep_new"+opt.select_epoch)
        self.source_dir = os.path.join(opt.dataroot, "t22ct/t22ct_" + opt.data_input + \
                                       "_seg_data_ep_new"+opt.select_epoch) \
            if opt.direction == "BtoA" else os.path.join(opt.dataroot, "ct2t2/ct2t2_" + \
                                                         opt.data_input + "_seg_data_ep_new"+opt.select_epoch)
        print('source_dir', self.source_dir)
        self.target_dir = os.path.join(opt.dataroot, 'ct_'+opt.data_input+'_seg_data') \
            if opt.direction == "BtoA" else os.path.join(opt.dataroot, 't2_'+opt.data_input+'_seg_data')

        self.isTrain = opt.data_input == 'train'
        # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, "mr_" + opt.phase+"_npy")  # create a path '/path/to/data/trainB'
        # self.vis_path = 'abdo_vis/'+'mr2ct/'+opt.data_input+'_'+opt.name+'_'+opt.epoch if opt.direction == "BtoA"\
        #     else 'abdo_vis/'+'ct2mr/'+opt.data_input+'_'+opt.name+'_'+opt.epoch
        # if not os.path.exists(self.vis_path):
        #     os.mkdir(self.vis_path)

        self.trans = A.Compose([A.Affine(scale=(0.9, 1.1), \
                                         translate_percent=0.1, rotate=5, shear=5, \
                                         interpolation=cv2.INTER_LINEAR, \
                                         mask_interpolation=cv2.INTER_NEAREST)])


        self.source_paths = sorted(self.make_dataset(self.source_dir))   # load images from '/path/to/data/trainA'
        self.target_paths = sorted(self.make_dataset(self.target_dir))
        # self.B_paths = sorted(self.make_dataset(self.dir_B))
        # # load images from '/path/to/data/trainB'
        # self.A_size = len(self.A_paths)  # get the size of dataset A
        # self.B_size = len(self.B_paths)  # get the size of dataset B
        self.source_size = len(self.source_paths)
        self.target_size = len(self.target_paths)
        # print('size:',self.source_size)
        # btoA = self.opt.direction == 'BtoA'
        # input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        # output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        # # self.transform_A = get_transform(self.opt)
        # # self.transform_B = get_transform(self.opt)

    def __getitem__(self, index):
        source_path = self.source_paths[index % self.source_size]  # make sure index is within then range
        target_path = self.target_paths[index % self.target_size]

        # get label path in the source data
        src_lb_path = source_path.replace('img', 'label').replace('.png', '.pgm')

        # get label path in the target data
        tgt_lb_path = target_path.replace('img', 'label').replace('.png', '.pgm')
        # load image datas
        source = cv2.imread(source_path)
        # print('max:',source.max())
        target = cv2.imread(target_path)
        # print('max:',target.max())

        # load labels
        source_label = cv2.imread(src_lb_path, 0)
        target_label = cv2.imread(tgt_lb_path, 0)
        # print('source_label shape:',source_label.shape)


        # source_label = self.encode_segmap(np.array(source_label))

        # augment images
        source, source_label = self.my_transform(source, source_label)
        target, target_label = self.my_transform(target, target_label)
        # target_label = torch.from_numpy(target_label)
        # source_label = torch.from_numpy(source_label)

        return {'source':source, 'target':target, 'source_label':source_label, 'target_label':target_label,\
                'source_path':source_path, 'target_path':target_path, 'src_lb_path':src_lb_path}
        # return {'A': A, 'B': B,'label_A': A_label, 'label_B': B_label, 'A_paths': A_path, 'B_paths': B_path}


    def make_dataset(self, imgdir):
        imagenames = []
        # assert osp.isdir(imgdir), '%s is not a valid directory' % imgdir
        for root, _, fnames in sorted(os.walk(imgdir)):
            for fname in fnames:
                if "img" in fname and is_image_file(fname):
                    fnamepath = os.path.join(imgdir,fname)
                    imagenames.append(fnamepath)
        return imagenames


    def encode_segmap(self, mask):
        for validc in self.valid_classes:
            mask[mask==validc] = self.class_map[validc]
        return mask


    def my_transform(self, image, label):
        image = np.array(image)
        # image = (image - image.min())/(image.max()-image.min()+1e-7)


        # image = np.expand_dims(image, 0)
        # image = image.transpose((2, 0, 1))
        # data = self.trans(image=image, mask=label)
        #
        # image = data['image']
        # label = data['mask']
        image = image.transpose((2, 0, 1))
        image = image / 255.0
        # print(image.shape)
        image = image * 2.0 - 1
        # print('shape:',image.shape)
        # print('label shape:', label.shape)

        image = torch.FloatTensor(image)
        # label = torch.LongTensor(label)
        return image, label


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        if self.phase == 'train':
            return max(self.source_size, self.target_size)
        else:
            return self.target_size
