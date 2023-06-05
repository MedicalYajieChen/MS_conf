import nibabel as nib
import numpy as np
import os
import glob
import cv2

ctpath = '../../datasets/2017-MM-WHS/data/mr_test_data/'
spath = '../../datasets/2017-MM-WHS/data/mr_test_seg_data/'
if not os.path.exists(spath):
    os.mkdir(spath)

ctim_paths = glob.glob(ctpath+'image*.nii.gz')
print(ctim_paths)

for filename in ctim_paths:
    img = nib.load(filename)
    lbname = filename.replace('image','gth')
    label = nib.load(filename.replace('image','gth'))
    img_arr = np.array(img.get_fdata())
    label_arr = np.array(label.get_fdata())
    img_arr = img_arr.transpose((2, 1, 0))
    label_arr = label_arr.transpose((2, 1, 0))
    print(img_arr.shape)
    i = 0
    for im_slice, lb_slice in zip(img_arr, label_arr):
        im_slice = 255.0*(im_slice-im_slice.min())/(im_slice.max()-im_slice.min())
        print(im_slice.max())
        # im_slice = 2.0*(im_slice+2.8)/6.0-1.0
        # im_slice = 2.0*(im_slice+1.8)/6.2-1.0
        # im_slice = (im_slice+1.0)*127.5
        # im_slice = np.flip(im_slice, axis=0)
        # im_slice = np.flip(im_slice, axis=1)
        # # lb_slice = 255.0*(lb_slice)/5.0
        # lb_slice = np.flip(lb_slice, axis=0)
        # lb_slice = np.flip(lb_slice, axis=1)
        cv2.imwrite(os.path.join(spath, filename.split('/')[-1].replace('.nii.gz', '_%03d.png'%i)), im_slice)
        cv2.imwrite(os.path.join(spath, lbname.split('/')[-1].replace('.nii.gz', '_%03d.pgm'%i).replace('gth_', 'label_')), lb_slice)
        # lb_slice = 255.0*lb_slice/5.0
        # cv2.imwrite(lbname.replace('_data', '_seg_data').replace('.nii.gz', '_%03d.jpg' % i),
        #         lb_slice)

        i += 1

