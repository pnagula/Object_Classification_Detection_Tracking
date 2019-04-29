#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 12:04:44 2017

@author: pivotalit
"""


from __future__ import print_function

import os
import numpy as np
import cv2
from skimage.io import imsave, imread
#from  PIL import Image
data_path = '/users/pivotalit/Downloads/'

image_rows = 512
image_cols = 512


def create_train_data():
    train_data_path = os.path.join(data_path, 'training_512’)
    images = os.listdir(train_data_path)

    total = len(images) / 2
    imgs = np.ndarray((image_rows, image_cols,3), dtype=np.uint8)
    imgs_mask = np.ndarray((image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating training images...')
    print('-'*30)
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.png'
        img = imread(os.path.join(train_data_path, image_name))
        img_mask = imread(os.path.join(train_data_path, image_mask_name))
    
        img = np.array([img])
	imgs[i]=img
        img_mask = np.array([img_mask])
	imgs_mask[i]=img_mask
#        img_mask=img_mask[...,np.newaxis]
        
#        path=data_path+'training_1024_npy/{0}.npy'.format(i)
#        np.save(path,img)
#        path=data_path+'training_1024_npy/{0}_mask.npy'.format(i)
#        np.save(path, img_mask)
    
        if i % 100 == 0:
           print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(data_path+'training_512/imgs_train_512_vhq.npy', imgs)
    np.save(data_path+'training_512/imgs_mask_train_512_vhq.npy', imgs_mask)
    print('Saving to .npy files done.')


def load_train_data():
    imgs_train = np.load(data_path+'training_512/imgs_train_512_vhq.npy')
    imgs_mask_train = np.load(data_path+'training_512/imgs_mask_train_512_vhq.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    #train_data_path = os.path.join(data_path, 'test')
    data_path = '/users/pivotalit/downloads/test_512/'
    train_data_path=data_path
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, image_rows, image_cols,3), dtype=np.uint8)
    iname=[]

    i = 1
    print('-'*30)
    print('Creating test images...')
    print('-'*30)
    for image_name in images:
        img = cv2.imread(os.path.join(train_data_path, image_name))
        img = np.array([img])

        imgs[i] = img
#        iname.append(image_name)
        if i % 100 == 0:
           print('Done: {0}/{1} images'.format(i, total))
        i += 1
           
    print('Loading done.')
    np.save(data_path+'imgs_test.npy', imgs)
#    with open("/users/pivotalit/downloads/test_512/image_names.txt", "w") as f:
#         for s in iname:
#             f.write(str(s) +"\n")
  #  np.save('imgs_id_test.npy', imgs_id)
    print('Saving to .npy files done.')


def load_test_data():
    imgs_test = np.load('imgs_test.npy')
    imgs_id = np.load('imgs_id_test.npy')
    return imgs_test, imgs_id

if __name__ == '__main__':
    create_train_data()
   # load_train_data()
   # create_test_data()
