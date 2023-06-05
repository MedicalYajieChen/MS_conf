
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from util.visualizer import save_images
from util import html
import random
import torch
import numpy as np

seed = 0

random.seed(seed)
np.random.seed(seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    # visualizer = Visualizer(opt)
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()
    with torch.no_grad():
        for i, data in enumerate(dataset):
            if i >= opt.num_test:  # only apply our model to opt.num_test images.
                break
            model.set_input(data)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()  # get image paths
            # print('img:',img_path)
            seg_path = model.get_seg_path()
            # print('seg_path:',seg_path)
            save_images(visuals, seg_path, img_path)
            if (i+1) % 100 == 0:
                print('iter %d'%(i+1))

        if opt.precision == 0:
            model.test_evaluation()
        else:
            model.precision()
    # webpage.save()  # save the HTML
