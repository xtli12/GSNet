import os
import os.path
import glob
import numpy as np
import argparse
from torchvision import *
from PIL import Image
import cv2
import numpy as np
# import pylab as pl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.signal
from FWLBP import *
from skimage.feature import local_binary_pattern
import sys
import albumentations
from albumentations import (RandomGridShuffle,GaussianBlur,Blur,Flip,ShiftScaleRotate,GridDistortion,ElasticTransform,HorizontalFlip,CenterCrop,
                            HueSaturationValue,Transpose,RandomBrightnessContrast,CLAHE,RandomCrop,Cutout,CoarseDropout,
                            CoarseDropout,Normalize,ToFloat,OneOf,Compose,Resize,RandomRain,RandomFog,Lambda
                            ,ChannelDropout,ISONoise,VerticalFlip,RandomGamma,RandomRotate90)

# 默认参数与文件路径声明
parser = argparse.ArgumentParser(description="SpectralSR")
parser.add_argument("--data_path", type=str, default='./datasets/validdata_rm_dup_rmerror_ep/', help="data path")
parser.add_argument("--train_data_path1", type=str, default='./datasets/validdata_rm_dup_rmerror_ep/',help="preprocess_data_path")
parser.add_argument("--train_data_path2", type=str, default='./train_rm_dup_rmerror_newep24_addLBP/',help="preprocess_data_path")
opt = parser.parse_args()


# 加载待处理数据路径并调用预处理主程序
# def rgb2gray(rgb):
#     r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
#     gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
#     return gray
#
#
# def im2col(A, BLKSZ):
#     # Parameters
#     M, N = A.shape
#     col_extent = N - BLKSZ[1] + 1
#     row_extent = M - BLKSZ[0] + 1
#
#     # Get Starting block indices
#     start_idx = np.arange(BLKSZ[0])[:, None] * N + np.arange(BLKSZ[1])
#
#     # Get offsetted indices across the height and width of input array
#     offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
#
#     # Get all actual indices & index into input array for final output
#     return np.take(A, start_idx.ravel()[:, None] + offset_idx.ravel())
#
#
# def coltfilt(A, size):
#     original_shape = np.shape(A)
#     a, b = 0, 0
#     if (size % 2 == 0):
#         a, b = int(size / 2) - 1, int(size / 2)
#     else:
#         a, b = int(size / 2), int(size / 2)
#     A = np.lib.pad(A, (a, b), 'constant')
#     Acol = im2col(A, (size, size))
#     rc = np.floor((Acol.max(axis=0) - Acol.min(axis=0)) / float(size)) + 1
#     return np.reshape(rc, original_shape)
#
#
# def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
#     """
#     2D gaussian mask - should give the same result as MATLAB's
#     fspecial('gaussian',[shape],[sigma])
#     """
#     m, n = [(ss - 1.) / 2. for ss in shape]
#     y, x = np.ogrid[-m:m + 1, -n:n + 1]
#     h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
#     h[h < np.finfo(h.dtype).eps * h.max()] = 0
#     sumh = h.sum()
#     if sumh != 0:
#         h /= sumh
#     return h
#
#
# def mat2gray(mat):
#     maxI = np.max(mat)
#     minI = np.min(mat)
#     gray = (mat[:, :] - minI) / (maxI - minI)
#     return gray

def main():
    data = data_augment(opt.data_path, opt.train_data_path1)
    data.augment()
    # a = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0]  # 晶粒度等级
    # for k in range(14):
    #     '''遍历各个晶粒度等级文件夹内的图片，再进行排序，并且创建一个储存结果的文件夹'''
    #     a1 = str(a[k])  # 将等级数转化成字符串类型数据
    #     img_path = os.path.join('./train_rm_dup_rmerror_new4/', a1)  # 找到各个晶粒度等级的文件夹
    #     img_path_name = glob.glob(os.path.join(img_path, '*.jpg'))
    #     img_path_name.sort()  # glob.glob是一个遍历函数，把某个晶粒度等级的文件夹内部的文件遍历出来，再排序
    #     result_path = os.path.join(opt.train_data_path2, a1)  # 用于储存扩增后数据的文件夹
    #     if not os.path.exists(result_path):  # 对文件夹进行操作的时候需要先对文件夹是否存在进行判断
    #         os.makedirs(result_path)  # 若不存在则创建一个文件夹
    #     for i in range(len(img_path_name)):  # 对某个晶粒度级别文件夹内的图片进行遍历
    #         # data.patch(i,img_path_name,result_path)
    #         data.augment(i)
    #     print(a[k])


class data_augment():
    def __init__(self,data_path,output_path):
        self.data = data_path
        self.out = output_path
    # def patch(self,num,img_path_name,result_path):
    #     print(img_path_name[num])
    #     d = img_path_name[num].split('/')[-1][0:18]  # 照片的名字，取/后的字符串，取1到16位
    #     rgb = cv2.imread(img_path_name[num])  # 读取图片
    #     rgb0 = rgb[0:960, 0:960, :]  # 对图片进行切片
    #     rgb_name = d + '_' + '0'  # 对生成的图片进行命名：等级+第几张图片+1
    #     rgb_dir = os.path.join(result_path, rgb_name)  # 将图片的存在路径传给rgb_dir
    #     cv2.imwrite(rgb_dir + '.jpg', rgb0, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 保存这个图片
    #     rgb1 = rgb[0:960, 416:1376, :]
    #     rgb_name = d + '_' + '1'
    #     rgb_dir = os.path.join(result_path, rgb_name)
    #     cv2.imwrite(rgb_dir + '.jpg', rgb1, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #     rgb2 = rgb[144:1104, 416:1376, :]
    #     rgb_name = d + '_' + '2'
    #     rgb_dir = os.path.join(result_path, rgb_name)
    #     cv2.imwrite(rgb_dir + '.jpg', rgb2, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #     rgb3 = rgb[144:1104, 0:960, :]
    #     rgb_name = d + '_' + '3'
    #     rgb_dir = os.path.join(result_path, rgb_name)
    #     cv2.imwrite(rgb_dir + '.jpg', rgb3, [cv2.IMWRITE_PNG_COMPRESSION, 0])



    def augment(self):
        b = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0]  # 晶粒度等级
        # b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for k in range(14):
            '''遍历各个晶粒度等级文件夹内的图片，再进行排序，并且创建一个储存结果的文件夹'''
            a1 = str(b[k])  # 将等级数转化成字符串类型数据
            img_path = os.path.join(self.data, a1)  # 找到各个晶粒度等级的文件夹
            # img_path_name = glob.glob(os.path.join(img_path, '*.jpg'))
            img_path_name = glob.glob(os.path.join(img_path, '*.png'))
            img_path_name.sort()  # glob.glob是一个遍历函数，把某个晶粒度等级的文件夹内部的文件遍历出来，再排序
            result_path = os.path.join(self.out, a1)  # 用于储存扩增后数据的文件夹
            if not os.path.exists(result_path):  # 对文件夹进行操作的时候需要先对文件夹是否存在进行判断
                os.makedirs(result_path)  # 若不存在则创建一个文件夹
            for i in range(len(img_path_name)):
                # load rgb images
                print(img_path_name[i])
                d = img_path_name[i].split('/')[-1][0:18]  # 照片的名字，取/后的字符串，取1到16位
                im_gray = cv2.imread(img_path_name[i])  # 读取图片


                im_gray = LBP(im_gray)

                # # im_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
                # im_gray = im_gray[::,::,0:1:1]
                # im_gray = im_gray.squeeze(2)
                # r1 = 1
                # # Number of points to be considered as neighbourers
                # no1_points = 8 * r1
                # # Uniform LBP is used
                # lbp1 = local_binary_pattern(im_gray, no1_points, r1, method='uniform')
                # plt.imshow(lbp1)
                # plt.show()
                # r2 = 2
                # # Number of points to be considered as neighbourers
                # no2_points = 8 * r2
                # # Uniform LBP is used
                # lbp2 = local_binary_pattern(im_gray, no2_points, r2, method='uniform')
                # plt.imshow(lbp2)
                # plt.show()
                # r3 = 3
                # # Number of points to be considered as neighbourers
                # no3_points = 8 * r3
                # # Uniform LBP is used
                # lbp3 = local_binary_pattern(im_gray, no3_points, r3, method='uniform')
                # plt.imshow(lbp3)
                # plt.show()


                lbp1 = torch.tensor(im_gray)
                lbp1 =lbp1.unsqueeze(0)
                lbp2 = torch.tensor(im_gray)
                lbp2 =lbp2.unsqueeze(0)
                lbp3 = torch.tensor(im_gray)
                lbp3 =lbp3.unsqueeze(0)
                # e = a
                # c = a

                intercept_image = torch.cat([lbp1,lbp2,lbp3],0)
                intercept_image = intercept_image.transpose(2 ,0)
                intercept_image = intercept_image.transpose(1 ,0)
                intercept_image = intercept_image.data.numpy()
                # plt.imshow(intercept_image)
                # plt.show()
                intercept_image = intercept_image*255

                rgb_name = d  + '_9'  # 对生成的图片进行命名：等级+第几张图片+1
                rgb_dir = os.path.join(result_path, rgb_name)  # 将图片的存在路径传给rgb_dir
                rgb_dir =rgb_dir + '.jpg'
                cv2.imwrite(rgb_dir , intercept_image)  # 保存这个图片
                # image_black_white = intercept_image.convert('1')
                # plt.imshow(image_black_white)
                # plt.show()
                # image_black_white.save(rgb_dir + '.jpg')
                # rgb_name = d + '_' + '6'
                # rgb_dir = os.path.join(result_path, rgb_name)  # 将图片的存在路径传给rgb_dir
                # cv2.imwrite(rgb_dir + '.jpg', rgb, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 保存这个图片
                # 切片随机重排
                # a = np.array(rgb)
                # rgb0 = RandomGridShuffle(grid=(3, 3), always_apply=False, p=0.5)(image=a)['image']
                # rgb_name = d + '_' + '0'  # 对生成的图片进行命名：等级+第几张图片+1
                # rgb_dir = os.path.join(result_path, rgb_name)  # 将图片的存在路径传给rgb_dir
                # cv2.imwrite(rgb_dir + '.jpg', rgb0, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 保存这个图片
                # 高斯模糊
                # a = np.array(rgb)
                # rgb1 = GaussianBlur(blur_limit=7, always_apply=False, p=0.6)(image=a)['image']
                # rgb_name = d + '_' + '1'
                # rgb_dir = os.path.join(result_path, rgb_name)
                # cv2.imwrite(rgb_dir + '.jpg', rgb1, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                # 随机旋转5°
                # rgb2 = Image.fromarray(rgb)
                # h = transforms.RandomRotation(5, resample=False, expand=False, center=None)
                # rgb2 = h(rgb2)
                # rgb2 = np.array(rgb2)
                # rgb_name = d + '_' + '2'
                # rgb_dir = os.path.join(result_path, rgb_name)
                # cv2.imwrite(rgb_dir + '.jpg', rgb2, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                # 随机擦除
                # a = np.array(rgb)
                # rgb3 = CoarseDropout(max_holes=100, max_height=20, max_width=20, min_holes=None, min_height=None, min_width=None, fill_value=250, always_apply=True, p=0.5)(image=a)['image']
                # rgb_name = d + '_' + '3'
                # rgb_dir = os.path.join(result_path, rgb_name)
                # cv2.imwrite(rgb_dir + '.jpg', rgb3, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                # # 水平翻转
                # rgb4 = Image.fromarray(rgb)
                # H = transforms.RandomHorizontalFlip(p=1)
                # rgb4 = H(rgb4)
                # rgb4 = np.array(rgb4)
                # rgb_name = d + '_' + '4'  # 对生成的图片进行命名：等级+第几张图片+1
                # rgb_dir = os.path.join(result_path, rgb_name)  # 将图片的存在路径传给rgb_dir
                # cv2.imwrite(rgb_dir + '.jpg', rgb4, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 保存这个图片
                # # 垂直翻转
                # rgb5 = Image.fromarray(rgb)
                # V = transforms.RandomVerticalFlip(p=1)
                # rgb5 = V(rgb5)
                # rgb5 = np.array(rgb5)
                # rgb_name = d + '_' + '5'
                # rgb_dir = os.path.join(result_path, rgb_name)
                # cv2.imwrite(rgb_dir + '.jpg', rgb5, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            print(b[k])



if __name__ == '__main__':
    main()
    print('Get it!')
