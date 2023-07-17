"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
import os.path
from skimage import io as IO
import natsort
import random

def default_loader(path):
    return Image.open(path).convert('RGB')
    IO.imread()
    

def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data
from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    images = natsort.natsorted(images)
    return images

class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader, data_name=None, data_type=None):
        self.data_name = data_name
        self.root = root
        self.data_type = data_type #indoor or outdoor or realworld
        if self.data_type == 'indoor':
            if self.data_name == 'trainA':
                imgs = make_dataset(self.root)#[0:6990]
            # 测试-----------------------------------
                f = open("./trainA.txt", 'a')
                for i in range(len(imgs)):
                  f.write(imgs[i])  # 将字符串写入文件中
                  f.write("\n")  # 换行
                f.close()
            # -----------------------------------------
            elif self.data_name == 'trainB':
                imgs = make_dataset(self.root)#[7000:]
                # 测试-----------------------------------
                f = open("./trainB.txt", 'a')
                for i in range(len(imgs)):
                    f.write(imgs[i])  # 将字符串写入文件中
                    f.write("\n")  # 换行
                f.close()
            # -----------------------------------------
            elif self.data_name == 'testA':
                imgs = make_dataset(self.root)
            elif self.data_name == 'testB':
                imgs = make_dataset(self.root)
            elif self.data_name == 'valA':
                imgs = make_dataset(self.root)
            elif self.data_name == 'valB':
                imgs = make_dataset(self.root)
        elif self.data_type == 'outdoor':
            if self.data_name == 'trainA':
                imgs = make_dataset_trainA_mul_dir(self.root)#[0:9100]
                # 测试-----------------------------------
                f = open("./trainA_outdoor.txt", 'a')
                for i in range(len(imgs)):
                    f.write(imgs[i])  # 将字符串写入文件中
                    f.write("\n")  # 换行
                f.close()
            # -----------------------------------------
            elif self.data_name == 'trainB':
                imgs = make_dataset_trainB_mul_dir(self.root)#[9100:]
                # 测试-----------------------------------
                f = open("./trainB_outdoor.txt", 'a')
                for i in range(len(imgs)):
                    f.write(imgs[i])  # 将字符串写入文件中
                    f.write("\n")  # 换行
                f.close()
            # -----------------------------------------
            elif self.data_name == 'testA':
                imgs = make_dataset_testA_mul_dir(self.root)
            elif self.data_name == 'testB':
                imgs = make_dataset(self.root)
            elif self.data_name == 'valA':
                imgs = make_dataset(self.root)
            elif self.data_name == 'valB':
                imgs = make_dataset(self.root)
        elif self.data_type == 'realworld':
            if self.data_name == 'trainA':
                imgs = make_dataset_trainA_mul_dir(self.root)
            elif self.data_name == 'trainB':
                imgs = make_dataset_trainB_mul_dir(self.root)
            elif self.data_name == 'testA':
                imgs = make_dataset_testA_mul_dir(self.root)
            elif self.data_name == 'testB':
                imgs = make_dataset(self.root)
            elif self.data_name == 'valA':
                imgs = make_dataset(self.root)
            elif self.data_name == 'valB':
                imgs = make_dataset(self.root)

        num_img = len(imgs)
        if num_img == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))
        else:
            print('{}_{}_num:{}'.format(self.data_type, self.data_name, num_img))

        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader


    def __getitem__(self, index):
        if self.data_name == 'trainB':
            index = random.randint(0, len(self.imgs) - 1)
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)






#----------------------------
def make_dataset_trainA__single_dir(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    num = len(images)
    # images = natsort.natsorted(images)[0:6990]
    images = natsort.natsorted(images)[0:4700]
    print('trainA size:', len(images), '/', num)
    return images

def make_dataset_trainA(dir):
    total_images = []
    for i, dir in enumerate(dir):  # 字典是value
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        num = len(images)
        # images = natsort.natsorted(images)[0:6990]
        images = natsort.natsorted(images)
        print('trainA_dir_{} size: {}/{}'.format(i+1, len(images), num))
        total_images += images
    print('trainA_total size:{}'.format(len(total_images)))
    return total_images

def make_dataset_trainA_mul_dir(dir):
    total_images = []
    for i, dir in enumerate(dir):  # 字典是value
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        num = len(images)
        # images = natsort.natsorted(images)[0:6990]
        images = natsort.natsorted(images)
        print('trainA_dir_{} size: {}/{}'.format(i+1, len(images), num))
        total_images += images
    print('trainA_total size:{}'.format(len(total_images)))
    # print('......................'.format(len(total_images)))
    return total_images
def make_dataset_trainB_mul_dir(dir):
    total_images = []
    for i, dir in enumerate(dir):  # 字典是value
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        num = len(images)
        # images = natsort.natsorted(images)[0:6990]
        images = natsort.natsorted(images)
        print('trainB_dir_{} size: {}/{}'.format(i+1, len(images), num))
        total_images += images
    print('trainB_total size:{}'.format(len(total_images)))
    # print('......................'.format(len(total_images)))
    return total_images
def make_dataset_testA_mul_dir(dir):
    total_images = []
    for i, dir in enumerate(dir):  # 字典是value
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        num = len(images)
        images = natsort.natsorted(images)
        print('test_dir_{} size: {}/{}'.format(i+1, len(images), num))
        total_images += images
    print('test_total size:{}'.format(len(total_images)))
    return total_images


def make_dataset_trainB(dir):
    total_images = []
    for i, dir in enumerate(dir.values()):
        images = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        num = len(images)
        # images = natsort.natsorted(images)[0:6990]
        images = natsort.natsorted(images)
        print('trainB_dir_{} size: {}/{}'.format(i+1, len(images), num))
        total_images += images
    print('trainB_total size:{}'.format(len(total_images)))
    return images

def make_dataset_testA_single_dir(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    images = natsort.natsorted(images)
    print('testA size:', len(images))
    return images

def make_dataset_testB(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    images = natsort.natsorted(images)
    print('testB size:', len(images))
    return images

def make_dataset_valA(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    images = natsort.natsorted(images)
    print('valA size:', len(images))
    return images

def make_dataset_valB(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                # for _ in range(10):  #indoor
                for _ in range(1):    #outdoor
                    images.append(path)
    images = natsort.natsorted(images)
    print('valB size:', len(images))
    return images

class ImageFolder_1(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader, data_name=None):
        self.data_name = data_name
        if self.data_name == 'trainA':
            imgs = make_dataset_trainA(root)

        elif self.data_name == 'trainB':
            imgs = make_dataset_trainB(root)

        elif self.data_name == 'testA':
            imgs = make_dataset_testA(root)

        elif self.data_name == 'testB':
            imgs = make_dataset_testB(root)

        elif self.data_name == 'valA':
            imgs = make_dataset_valA(root)

        elif self.data_name == 'valB':
            imgs = make_dataset_valB(root)

        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader


    def __getitem__(self, index):
        if self.data_name == 'trainB':
            index = random.randint(0, len(self.imgs) - 1)
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)