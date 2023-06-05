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


class UnalignAbdoTestDataset(BaseDataset):
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.opt = opt
        self.valid_classes = [0, 61, 126, 150, 246]
        self.train_classes = [0, 1, 2, 3, 4]

        # self.class_map = dict(zip(self.valid_classes, self.train_classes))

        # self.source_dir = os.path.join(opt.dataroot, "mr_val_seg_data")
        self.target_dir = os.path.join(opt.dataroot, 'ct_'+opt.data_input+'_seg_data') if opt.direction == 'BtoA'\
            else os.path.join(opt.dataroot, 't2_'+opt.data_input+'_seg_data')
        # self.target_dir = os.path.join(opt.dataroot, 'ct_val_seg_data')
        self.isTrain = opt.data_input == 'train'
        # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, "mr_" + opt.phase+"_npy")  # create a path '/path/to/data/trainB'


        # self.source_paths = sorted(self.make_dataset(self.source_dir))   # load images from '/path/to/data/trainA'
        self.target_paths = sorted(self.make_dataset(self.target_dir), \
                                   key = lambda i:int(i.split('/')[-1].split('_')[0].split('img')[1]))
        # random.shuffle(self.target_paths)
        # self.B_paths = sorted(self.make_dataset(self.dir_B))
        # # load images from '/path/to/data/trainB'
        # self.A_size = len(self.A_paths)  # get the size of dataset A
        # self.B_size = len(self.B_paths)  # get the size of dataset B
        # self.source_size = len(self.source_paths)
        self.target_size = len(self.target_paths)
        print(self.target_size)
        self.vis_path = 'abdo_vis/t22ct/' + opt.data_input + '_' + opt.name + '_' + opt.epoch \
            if opt.direction == "BtoA" else 'abdo_vis/ct2t2/' + opt.data_input + '_' + opt.name + '_' + opt.epoch
        # self.vis_path = 'whs_vis/'+'mr2ct/'+opt.data_input+'_'+opt.name+'_' + opt.epoch if opt.direction == 'BtoA'\
        #     else 'whs_vis/'+'ct2mr/'+opt.data_input+'_'+opt.name+'_' + opt.epoch
        # if not os.path.exists(self.vis_path):
        #     os.mkdir(self.vis_path)
        # btoA = self.opt.direction == 'BtoA'
        # input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        # output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        # # self.transform_A = get_transform(self.opt)
        # # self.transform_B = get_transform(self.opt)

    def __getitem__(self, index):
        # source_path = self.source_paths[index % self.source_size]  # make sure index is within then range
        target_path = self.target_paths[index % self.target_size]
        # if self.opt.serial_batches:   # make sure index is within then range
        #     index_B = index % self.B_size
        # else:   # randomize the index for domain B to avoid fixed pairs.
        #     index_B = random.randint(0, self.B_size - 1)
        # B_path = self.B_paths[index_B]
        # src_lb_path = source_path.replace('image','label')
        tgt_lb_path = target_path.replace('img', 'label').replace('.png', '.pgm')
        # lblB_path = B_path.replace("image","label")
        # source = Image.open(source_path).convert('RGB')
        # source = cv2.imread(source_path)
        # source = Image.open(source_path).convert('RGB')
        # target = Image.open(target_path).convert('RGB')
        target = cv2.imread(target_path)
        # B_img = Image.open(B_path).convert('RGB')

        # A_img = Image.open(A_path).convert('L')
        # B_img = Image.open(B_path).convert('L')
        # A_img = np.array(np.load(A_path))[:,:,1]
        # B_img = np.array(np.load(B_path))[:,:,1]
        # B_img_copy = B_img.copy()
        # B_img_copy = 255.0*(B_img_copy-B_img_copy.min())/(B_img_copy.max()-B_img_copy.min())
        # cv2.imwrite('B.png',B_img_copy)

        # source_label = np.array(cv2.imread(src_lb_path, 0))
        # source_label = source_label.transpose((2, 0, 1))

        # target_label = np.array(np.load(tgt_lb_path))
        target_label = cv2.imread(tgt_lb_path, 0)
        # print('max:',target_label.max())
        # target_label = target_label.transpose((2, 0, 1))
        # source_label = source_label.astype(np.uint64)
        # A_label = np.array(np.load(lblA_path))[:, :, 0]
        # B_label_copy = B_label.copy()
        # B_label_copy = 255.0*(B_label_copy-B_label_copy.min())/(B_label_copy.max()-B_label_copy.min())
        # cv2.imwrite('B_label.png',B_label_copy)
        # A_label = Image.open(lblA_path).convert('L')
        # B_label = Image.open(lblB_path).convert('L')
        # source_label = self.encode_segmap(np.array(source_label))
        # B_label = self.encode_segmap(np.array(B_label))
        # apply image transformation
        # source = self.my_transform(source)
        # img_transform = get_transform()
        target = self.my_transform(target)
        target_label = torch.from_numpy(target_label)

        # B = self.my_transform(B_img)
        # if opt.phase == 'test':
        #     seg_path = os.path.join(opt.checkpoints_dir, 'seg_data')
        #     os.mkdir(seg_path)
        #     return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'seg_path':seg_path}
        # else:
        #     return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}
        # return {'source':source, 'target':target, 'source_label':source_label, 'target_label':target_label,\
        #         'source_path':source_path, 'target_path':target_path, 'src_lb_path':src_lb_path}

        return {'target': target, 'target_label': target_label, \
                'target_path': target_path,'seg_path':self.vis_path}
        # return {'source': source, 'source_label': source_label, \
        #         'source_path': source_path, 'seg_path': 'New_result'}
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
