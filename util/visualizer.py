import numpy as np
import os
import sys
import ntpath
import time
from . import util, html
from subprocess import Popen, PIPE
from PIL import Image
import matplotlib.pyplot as plt
# from scipy.misc import imresize

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(visuals, seg_path, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    # image_dir = webpage.get_image_dir()
    # print('path:',image_path)
    short_path = ntpath.basename(image_path[0])
    # print('st',short_path)
    name = os.path.splitext(short_path)[0]
    if not os.path.exists(seg_path[0]):
        os.mkdir(seg_path[0])
    # print('image:',image_path[0])

    # webpage.add_header(name)
    # ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im_data = im_data.squeeze(0)


        # print(label)
        # print('shape:', im_data.shape)

        if label == 'target_p':
            # print(im_data.shape)
            pred_name = '%s_%s.npy' % (name,label)
            np.save(os.path.join(seg_path[0], pred_name), im_data)
        else:
            im = util.tensor2im(im_data, label)
            image_name = '%s_%s.png'%(name, label)
            # print('im:',im.shape)
            save_path = os.path.join(seg_path[0], image_name)
            # if label == 'target_entropy' or label == 'target_kl':
            #     ar = np.array(im).flatten()
            #     plt.hist(ar, bins=256, density=1, stacked=True, facecolor='r', edgecolor='r')
            #     plt.savefig(save_path)
            #     plt.close()
            #     return
            h, w = im.shape[0], im.shape[1]
            if aspect_ratio > 1.0:
                im = np.array(Image.fromarray(im).resize(h, int(w * aspect_ratio)))
            if aspect_ratio < 1.0:
                im = np.array(Image.fromarray(im).resize(int(h / aspect_ratio), w))
            util.save_image(im, save_path)


def save_image_data(visual, seg_path, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]
    # print(seg_path)
    # with open('log.txt', 'a+') as f:
    #     f.write(seg_path)
    if not os.path.exists(seg_path):
        os.mkdir(seg_path)

    im_data = visual.squeeze(0)
    im_data = im_data.transpose((1,2,0))
    # print(im_data.shape)
    im = util.tensor2im(im_data, '')
    image_name = '%s.png'%(name)

    save_path = os.path.join(seg_path, image_name)
    h, w = im.shape[0], im.shape[1]
    if aspect_ratio > 1.0:
        im = np.array(Image.fromarray(im).resize(h, int(w * aspect_ratio)))
    if aspect_ratio < 1.0:
        im = np.array(Image.fromarray(im).resize(int(h / aspect_ratio), w))
    util.save_image(im, save_path)


class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False

        self.img_dir = os.path.join(opt.checkpoints_dir, opt.name, 'images')
        print('create img directory %s...' % self.img_dir)
        util.mkdirs([self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """


        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items():
                image_numpy = util.tensor2im(image, label)
                print('image_numpy:',image_numpy.shape)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                util.save_image(image_numpy, img_path)


    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message
