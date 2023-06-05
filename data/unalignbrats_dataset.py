import os.path 
from data.base_dataset import BaseDataset#, get_transform
from PIL import Image
import random
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms

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


class UnalignBratsDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.phase = opt.phase

        self.source_dir = os.path.join(opt.dataroot, "t2_train_seg_data")
        self.target_dir = os.path.join(opt.dataroot, 't1_train_seg_data') if opt.target == "t1" \
            else os.path.join(opt.dataroot, "t1ce_train_seg_data") if opt.target == "t1ce" \
            else os.path.join(opt.dataroot, "flair_train_seg_data")

        self.isTrain = opt.data_input == 'train'

        self.source_paths = sorted(self.make_dataset(self.source_dir))   # load images from '/path/to/data/trainA'
        self.target_paths = sorted(self.make_dataset(self.target_dir))

        self.source_size = len(self.source_paths)
        self.target_size = len(self.target_paths)
        print(self.source_size)

    def __getitem__(self, index):
        source_path = self.source_paths[index % self.source_size]  # make sure index is within then range
        target_path = self.target_paths[index % self.target_size]

        # get label path in the source data
        src_lb_path = source_path.replace('_t2', '_label').replace('.png', '.pgm') \

        # get label path in the target data
        tgt_lb_path = target_path.replace('_t1', '_label').replace('.png', '.pgm') if self.opt.target == "t1" \
            else target_path.replace('_t1ce', '_label').replace('.png', '.pgm') if self.opt.target == "t1ce" \
            else target_path.replace('_flair', '_label').replace('.png', '.pgm')

        # load image datas
        source = cv2.imread(source_path)
        target = cv2.imread(target_path)

        # load labels
        source_label = cv2.imread(src_lb_path, 0)
        source_label = np.expand_dims(source_label, axis=0)
        target_label = cv2.imread(tgt_lb_path, 0)
        target_label = np.expand_dims(target_label, axis=0)

        # augment images
        source, source_label = self.my_transform(source, source_label)
        target, target_label = self.my_transform(target, target_label)

        return {'source':source, 'target':target, 'source_label':source_label, 'target_label':target_label,\
                'source_path':source_path, 'target_path':target_path, 'src_lb_path':src_lb_path}
        # return {'A': A, 'B': B,'label_A': A_label, 'label_B': B_label, 'A_paths': A_path, 'B_paths': B_path}


    def make_dataset(self, imgdir):
        imagenames = []
        # assert osp.isdir(imgdir), '%s is not a valid directory' % imgdir
        for root, _, fnames in sorted(os.walk(imgdir)):
            for fname in fnames:
                if "png" in fname and is_image_file(fname):
                    fnamepath = os.path.join(imgdir,fname)
                    imagenames.append(fnamepath)
        return imagenames


    def my_transform(self,image, label):
        image = np.array(image)
        # image = (image - image.min())/(image.max()-image.min()+1e-7)
        image = image / 255.0
        # print(image.shape)
        image = image * 2 - 1
        # image = np.expand_dims(image, 0)
        image = image.transpose((2, 0, 1))
        image = torch.FloatTensor(image)
        label = torch.FloatTensor(label)
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
