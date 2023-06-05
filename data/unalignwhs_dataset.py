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


class UnalignWHSDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.phase = opt.phase
        self.valid_classes = [0, 63, 127, 191, 255]
        self.train_classes = [0, 1, 2, 3, 4]

        self.class_map = dict(zip(self.valid_classes, self.train_classes))
        # if opt.no_adaptation == 0:
        self.source_dir = os.path.join(opt.dataroot, "mr2ct/mr2ct_"+opt.data_input+"_seg_data_ep_new"+opt.select_epoch) \
            if opt.direction == "BtoA" else os.path.join(opt.dataroot, "ct2mr/ct2mr_"+opt.data_input+"_seg_data_ep_new"+opt.select_epoch)
        # else:
        #     self.source_dir = os.path.join(opt.dataroot, "mr_" + opt.data_input + "_seg_data") \
        #         if opt.direction == "BtoA" else os.path.join(opt.dataroot, "ct_" + opt.data_input + "_seg_data")

        self.target_dir = os.path.join(opt.dataroot, 'ct_'+opt.data_input+'_seg_data') \
            if opt.direction == "BtoA" else os.path.join(opt.dataroot, 'mr_'+opt.data_input+'_seg_data')

        self.isTrain = opt.data_input == 'train'
        self.vis_path = 'whs_vis/mr2ct/'+opt.data_input+'_'+opt.name+'_'+opt.epoch if opt.direction == "BtoA" \
            else 'whs_vis/ct2mr/'+opt.data_input+'_'+opt.name+'_'+opt.epoch
        if not os.path.exists(self.vis_path):
            os.mkdir(self.vis_path)


        self.source_paths = sorted(self.make_dataset(self.source_dir))   # load images from '/path/to/data/trainA'
        self.target_paths = sorted(self.make_dataset(self.target_dir))

        self.source_size = len(self.source_paths)
        self.target_size = len(self.target_paths)
        print('source size:{}'.format(self.source_size))


    def __getitem__(self, index):
        source_path = self.source_paths[index % self.source_size]  # make sure index is within then range
        target_path = self.target_paths[index % self.target_size]

        # get label path in the source data
        # src_lb_path = source_path.replace('image', 'label').replace('.png', '.pgm')\
        #     if self.isTrain else \
        #     source_path.replace('image', 'label')
        src_lb_path = source_path.replace('image', 'label')
        # get label path in the target data
        tgt_lb_path = target_path.replace('image', 'label')

        # load image datas
        source = cv2.imread(source_path)
        target = cv2.imread(target_path)

        # load labels
        source_label = cv2.imread(src_lb_path, 0)
        target_label = cv2.imread(tgt_lb_path, 0)

        # target_label = 51.0 * target_label
        # cv2.imwrite(tgt_lb_path.replace('.png', '.jpg'), target_label)


        # source_label = self.encode_segmap(np.array(source_label))
        source = source.transpose((2, 0, 1))
        target = target.transpose((2, 0, 1))
        # cv2.imwrite(source_path.replace('.png', '.jpg'), source[0])
        # augment images
        source = self.my_transform(source)
        target = self.my_transform(target)


        return {'source':source, 'target':target, 'source_label':source_label, 'target_label':target_label,\
                'source_path':source_path, 'target_path':target_path, 'src_lb_path':src_lb_path, 'seg_path':self.vis_path}
        # return {'A': A, 'B': B,'label_A': A_label, 'label_B': B_label, 'A_paths': A_path, 'B_paths': B_path}


    def make_dataset(self, imgdir):
        imagenames = []
        # assert osp.isdir(imgdir), '%s is not a valid directory' % imgdir
        for root, _, fnames in sorted(os.walk(imgdir)):
            for fname in fnames:
                if "image" in fname and is_image_file(fname):
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
        # image = image.transpose((2, 0, 1))
        # cv2.imshow(image[0])
        image = torch.FloatTensor(image)
        return image


    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        if self.phase == 'train':
            return max(self.source_size, self.target_size)
            # return self.source_size
        else:
            return self.target_size
