import tensorflow as tf
import json
import glob
import nibabel as nib
import numpy as np
import argparse
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
# tf.enable_eager_execution()
# print(tf.eagerly())

# with open('./config_param.json') as config_file:
#     config = json.load(config_file)
#
# BATCH_SIZE = int(config['batch_size'])

@tf.function
def _decode_samples(serialized_example, shuffle=False):
    decomp_feature = {
        # image size, dimensions of 3 consecutive slices
        'dsize_dim0': tf.io.FixedLenFeature([], tf.int64), # 256
        'dsize_dim1': tf.io.FixedLenFeature([], tf.int64), # 256
        'dsize_dim2': tf.io.FixedLenFeature([], tf.int64), # 3
        # label size, dimension of the middle slice
        'lsize_dim0': tf.io.FixedLenFeature([], tf.int64), # 256
        'lsize_dim1': tf.io.FixedLenFeature([], tf.int64), # 256
        'lsize_dim2': tf.io.FixedLenFeature([], tf.int64), # 1
        # image slices of size [256, 256, 3]
        'data_vol': tf.io.FixedLenFeature([], tf.string),
        # label slice of size [256, 256, 3]
        'label_vol': tf.io.FixedLenFeature([], tf.string)}

    raw_size = [256, 256, 3]
    volume_size = [256, 256, 3]
    label_size = [256, 256, 1] # the label has size [256,256,3] in the preprocessed data, but only the middle slice is used

    # data_queue = tf.data.Dataset.from_tensor_slices(image_list)
    # reader = tf.TFRecordReader()
    # fid, serialized_example = reader.read(data_queue)
    parser = tf.io.parse_single_example(serialized_example, features=decomp_feature)

    data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
    data_vol = tf.reshape(data_vol, raw_size)
    data_vol = tf.slice(data_vol, [0, 0, 0], volume_size)
    #
    label_vol = tf.decode_raw(parser['label_vol'], tf.float32)
    label_vol = tf.reshape(label_vol, raw_size)
    label_vol = tf.slice(label_vol, [0, 0, 1], label_size)
    # sess = tf.Session()
    # sess.run(data_vol)

    # data_vol = data_vol.eval()
    # data_vol = data_vol.numpy()
    # data_vol = data_vol.numpy()
    # label_vol = label_vol.eval()
    # label_vol = label_vol.numpy()
    # print('shape:{}'.format(data_vol.shape))

    # batch_y = tf.one_hot(tf.cast(tf.squeeze(label_vol), tf.uint8), 5)
    # if 'mr' in filename:
    data_vol = tf.subtract(tf.multiply(tf.div(tf.subtract(data_vol, -1.8), tf.subtract(4.4, -1.8)), 2.0), 1)
        # image = 2.0 * (image + 1.8) / 6.2 - 1.0
    # elif 'ct' in filename:
        # image_i = tf.subtract(tf.multiply(tf.div(tf.subtract(image_i, -2.8), tf.subtract(3.2, -2.8)), 2.0), 1)
        # image = 2.0 * (image + 2.8) / 6.0 - 1.0

    return data_vol, label_vol


def _load_samples(source_pth, save_dir):

    # with open(source_pth, 'r') as fp:
    #     rows = fp.readlines()
    # imagea_list = [row[:-1] for row in rows]
    #
    # with open(target_pth, 'r') as fp:
    #     rows = fp.readlines()
    # imageb_list = [row[:-1] for row in rows]

    image_list = glob.glob(source_pth+'/*.tfrecords')
    image_list.sort()
    for img_name in image_list[1200:]:
        dataset = tf.data.TFRecordDataset([img_name,])
        dataset = dataset.map(_decode_samples).batch(1)
        iterator = dataset.make_one_shot_iterator()
        element = iterator.get_next()
        # data_vol, label_vol = _decode_samples(image_list, shuffle=True)
        # data_volb, label_volb = _decode_samples(imageb_list, shuffle=True)
        # for data in dataset:
        #     data_vol, label_vol = data
        with tf.Session() as sess:
            # while True:
            #     # data_vol, label_vol = sess.run(element)
            #     # print(image_vol.shape, image_vol.dtype, label_vol)  # (28, 28, 3) float64 1
            #     try:
            data_vol, label_vol = sess.run(element)
            # print(np.array(data_vol).shape, np.array(data_vol).dtype, np.array(label_vol).shape)  # (28, 28, 3) float64 1
                # except OutOfRangeError:
                #     print("数据读取完毕")
                #     break
        proc_data(np.array(data_vol[0,:,:,1]), np.array(label_vol[0,:,:,0]), save_dir, img_name)

    return data_vol, label_vol


def proc_data(image, gt, save_dir, filename):

    # For converting the value range to be [-1 1] using the equation 2*[(x-x_min)/(x_max-x_min)]-1.
    # The values {-1.8, 4.4, -2.8, 3.2} need to be changed according to the statistics of specific datasets
    # if 'mr' in filename:
    #     # image_i = tf.subtract(tf.multiply(tf.div(tf.subtract(image_i, -1.8), tf.subtract(4.4, -1.8)), 2.0), 1)
    #     image = 2.0 * (image + 1.8) / 6.2 - 1.0
    # elif 'ct' in filename:
    #     # image_i = tf.subtract(tf.multiply(tf.div(tf.subtract(image_i, -2.8), tf.subtract(3.2, -2.8)), 2.0), 1)
    #     image = 2.0 * (image + 2.8) / 6.0 - 1.0

    image = np.rot90(image, 2)
    # print('image:{}'.format(image.shape))
    # image = np.flip(image, axis=1)
    image = image.transpose(1,0)
    gt = np.rot90(gt, 2)
    # gt = np.flip(gt, axis=1)
    gt = gt.transpose(1, 0)

    image = (255.0*(image-image.min())/(image.max() - image.min()))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(save_dir, filename.split('/')[-1].replace('slice', 'image').replace('.tfrecords', '.png')),image)
    cv2.imwrite(os.path.join(save_dir, filename.split('/')[-1].replace('slice', 'label').replace('.tfrecords', '.png')),
                gt)
    # label_vis = (51.0 * gt)
    # cv2.imwrite(os.path.join(save_dir, filename.split('/')[-1].replace('slice', 'label').replace('.tfrecords', '.jpg')),
    #             label_vis)



    # if 'ct' in target_pth:
    #     image_j = tf.subtract(tf.multiply(tf.div(tf.subtract(image_j, -2.8), tf.subtract(3.2, -2.8)), 2.0), 1)
    # elif 'mr' in target_pth:
    #     image_j = tf.subtract(tf.multiply(tf.div(tf.subtract(image_j, -1.8), tf.subtract(4.4, -1.8)), 2.0), 1)


    # Batch
    # if do_shuffle is True:
    #     images_i, gt_i = tf.train.shuffle_batch([image_i, gt_i], BATCH_SIZE, 500, 100)
    # else:
    #     images_i, gt_i = tf.train.batch([image_i, gt_i], batch_size=1, num_threads=1, capacity=500)

    return image, gt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../../datasets/2017-MM-WHS/data/')
    parser.add_argument('--module', type=str, default='mr')
    parser.add_argument('--phase',  type=str, default='train')
    args = parser.parse_args()
    root = args.root
    module = args.module
    phase = args.phase

    data_path = os.path.join(root, module+'_'+phase+'_tfs')
    save_dir = os.path.join(root, module+'_'+phase+'_seg_data')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    _load_samples(data_path, save_dir)
