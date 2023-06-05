import os.path
from data.base_dataset import BaseDataset#, get_transform
from PIL import Image
import random
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
import random

def get_transform():
    transform_list = []
    transform_list += [transforms.ToTensor()]
    transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP','.pgm','.PGM'
]





def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


class UnalignBratsTestDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt

        self.target_dir = os.path.join(opt.dataroot, 't1_test_seg_data') if opt.target == "t1" \
            else os.path.join(opt.dataroot, "t1ce_test_seg_data") if opt.target == "t1ce" \
            else os.path.join(opt.dataroot, "flair_test_seg_data") if opt.target == "flair"\
            else os.path.join(opt.dataroot, "t2_test_seg_data")
        # self.target_dir = os.path.join(opt.dataroot, 'flair_test_seg_data')

        self.target_paths = sorted(self.make_dataset(self.target_dir))
        self.target_size = len(self.target_paths)
        print(self.target_size)
        self.vis_path = 'brats_vis/t22flair/' + opt.data_input + '_' + opt.name + '_' + opt.epoch



    def __getitem__(self, index):
        # source_path = self.source_paths[index % self.source_size]  # make sure index is within then range
        target_path = self.target_paths[index % self.target_size]

        tgt_lb_path = target_path.replace('_t1', '_label').replace('.png', '.pgm') if self.opt.target == "t1" \
            else target_path.replace('_t1ce', '_label').replace('.png', '.pgm') if self.opt.target == "t1ce" \
            else target_path.replace('_flair', '_label').replace('.png', '.pgm') if self.opt.target == "flair" \
            else target_path.replace('_t2', '_label').replace('.png', '.pgm')
        # tgt_lb_path = target_path.replace('_flair', '_label').replace('.png', '.pgm')
        target = cv2.imread(target_path)
        target_label = cv2.imread(tgt_lb_path, 0)
        target_label = np.expand_dims(target_label,axis=0)
        target = self.my_transform(target)

        return {'target': target, 'target_label': target_label, \
                'target_path': target_path,'seg_path':self.vis_path}


    def make_dataset(self, imgdir):
        imagenames = []
        # assert osp.isdir(imgdir), '%s is not a valid directory' % imgdir
        for root, _, fnames in sorted(os.walk(imgdir)):
            for fname in fnames:
                if "png" in fname and is_image_file(fname):
                    fnamepath = os.path.join(imgdir,fname)
                    imagenames.append(fnamepath)
        return imagenames

    def encode_segmap(self, mask):
        for validc in self.valid_classes:
            mask[mask==validc] = self.class_map[validc]
        return mask

    def my_transform(self,image):
        image = np.array(image)
        # image = (image - image.min())/(image.max()-image.min()+1e-7)
        image = image / 255.0
        # print(image.shape)
        image = image * 2 - 1
        # image = np.expand_dims(image, 0)
        image = image.transpose((2,0,1))
        image = torch.FloatTensor(image)
        return image


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.target_size
