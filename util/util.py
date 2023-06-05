"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2


def tensor2im(input_image, label, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """

    # print('label:',label)
    # print('image_numpy:', input_image.shape)

    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_numpy = input_image.cpu().numpy()
        else:
            return input_image

    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
        if label == 'target_label' or label == 'target_u1' or label == 'target_u2' or label == 'target_seg3' or label == 'target_seg4':
            # image_numpy = 255.0 * image_numpy / 5.0
            canvas = np.zeros((image_numpy.shape[0], image_numpy.shape[1], 3))
            canvas[image_numpy == 1] = [255, 0, 0]
            canvas[image_numpy == 2] = [0, 255, 0]
            canvas[image_numpy == 3] = [0, 0, 255]
            canvas[image_numpy == 4] = [0, 255, 255]
            image_numpy = canvas

        elif label == 'target_cps' or label == 'target_js' or label == 'target_entropy_u1' or label == 'target_entropy_u2' \
                or label == 'target_entropy_seg3':
            # print('max:',image_numpy.max())
            # print('min:', image_numpy.min())
            # image_numpy = (image_numpy - image_numpy.min()) / \
            #               (image_numpy.max() - image_numpy.min()) * 255.0
            image_numpy = image_numpy * 255.0
            # image_numpy = cv2.applyColorMap(image_numpy, 2)
        elif label == 'target_kl':
            image_numpy = image_numpy * 12.0
        else:
            # image_numpy = (image_numpy + 1.0) * 127.5
            image_numpy = image_numpy.transpose((1,2,0))
            image_numpy = (image_numpy - image_numpy.min()) / \
                          (image_numpy.max() - image_numpy.min()) * 255.0

    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    # print('numpy shape:',image_numpy.shape)
    # image_pil = Image.fromarray(image_numpy)
    # image_pil.save(image_path)
    cv2.imwrite(image_path, image_numpy)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
