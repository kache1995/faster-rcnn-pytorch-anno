
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from PIL import Image
import torch

from lib. model.utils.config import cfg
from lib.roi_data_layer.minibatch import get_minibatch, get_minibatch
from lib.model.rpn.bbox_transform import bbox_transform_inv, clip_boxes

import numpy as np
import random
import time
import pdb

class roibatchLoader(data.Dataset):#继承了dataset类
  def __init__(self, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
    '''

    :param roidb:
    :param ratio_list: [0.1,0.2,0.6,0.7,0.71,...]所有训练图片的长宽比（已经按照从小到大排序）
    :param ratio_index: [4,2,1,...] 代表roidb里面第4张图片长宽比最小，然后是第2张，....
    :param batch_size: b
    :param num_classes: 21,
    :param training: True
    :param normalize: None

      roidb：=[{},{},{},....{}]

      {'boxes': array([[ 47, 239, 194, 370],
     [  7,  11, 351, 497]], dtype=uint16), 'gt_classes': array([12, 15]), 'gt_ishard': array([0, 0]), 'gt_overlaps': <2x21 sparse matrix of type '<class 'numpy.float32'>'
  with 2 stored elements in Compressed Sparse Row format>, 'flipped': False, 'seg_areas': array([ 19536., 168015.], dtype=float32)}

  box.shape=(n,4) n代表这张图片上有几个目标
  gt_classes.shape=(n)   例如：array([12, 15]) 代表第一个目标是第12类（猫），第二个目标是第15类（人）
  gt_overlaps.   overlaps.shape=(n,21) 值为0或1  默认0， 如果[2,20]=1代表该张图片上第2个目标是第20类的分割
  flipped= False(前5011个为false ，后5011个为True)
  seg_areas.shape=(n) 例如：array([ 19536., 168015.] 代表第一个目标box的面积是19536， 第二个目标box的面积是168015
  max_classes=[12,15]  表示第一个目标是第12类，第二个目标是滴15类
  max_overlaps=[1,1,1,..] 如果这张图片有n个obj ，那么就有n个1
  need_crop = 0或1   1：代表图片长宽比太大或者台太小，需要裁剪  0：代表不需要





    '''
    self._roidb = roidb
    self._num_classes = num_classes#21
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT#600
    self.trim_width = cfg.TRAIN.TRIM_WIDTH#600
    self.max_num_box = cfg.MAX_NUM_GT_BOXES#20一张图片最多有20个gt
    self.training = training#True
    self.normalize = normalize#None
    self.ratio_list = ratio_list
    self.ratio_index = ratio_index
    self.batch_size = batch_size
    self.data_size = len(self.ratio_list)#10022 训练图片的数量

    # given the ratio_list, we want to make the ratio same for each batch.
    self.ratio_list_batch = torch.Tensor(self.data_size).zero_()#[0,0,0,0,....] 共10022个0
    num_batch = int(np.ceil(len(ratio_index) / batch_size)) #10022/batchsize 取整
    for i in range(num_batch):
        left_idx = i*batch_size
        right_idx = min((i+1)*batch_size-1, self.data_size-1)

        #选择这个batch里面长宽差距较大的图片的窗宽比作为这个batch的同意长宽比
        if ratio_list[right_idx] < 1:
            # for ratio < 1, we preserve the leftmost in each batch.
            target_ratio = ratio_list[left_idx]
        elif ratio_list[left_idx] > 1:
            # for ratio > 1, we preserve the rightmost in each batch.
            target_ratio = ratio_list[right_idx]
        else:
            # for ratio cross 1, we make it to be 1.
            target_ratio = 1

        self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio#设置长宽比（每个batch使用共同的长宽比）


  def __getitem__(self, index):
    if self.training:
        index_ratio = int(self.ratio_index[index])
        #照这样看来，这里并不直接选择索引index的图片，而是选择按长宽比排序的第index个图片（ratio_index里面保存的是按长宽比排序的图片的索引）
    else:
        index_ratio = index

    # get the anchor index for current sample index
    # here we set the anchor index to the last one
    # sample in this group
    minibatch_db = [self._roidb[index_ratio]]
    '''  
    minibatch_db=[{}] 里面就一个字典
    
    box.shape=(n,4) n代表这张图片上有几个目标
  gt_classes.shape=(n)   例如：array([12, 15]) 代表第一个目标是第12类（猫），第二个目标是第15类（人）
  gt_overlaps.   overlaps.shape=(n,21) 值为0或1  默认0， 如果[2,20]=1代表该张图片上第2个目标是第20类的分割
  flipped= False(前5011个为false ，后5011个为True)
  seg_areas.shape=(n) 例如：array([ 19536., 168015.] 代表第一个目标box的面积是19536， 第二个目标box的面积是168015
  max_classes=[12,15]  表示第一个目标是第12类，第二个目标是滴15类
  max_overlaps=[1,1,1,..] 如果这张图片有n个obj ，那么就有n个1
  need_crop = 0或1   1：代表图片长宽比太大或者台太小，需要裁剪  0：代表不需要

    
    '''




    blobs = get_minibatch(minibatch_db, self._num_classes)
    '''
    blobs:{}

    data : im_blob.shape=(1,W,H,3)是经过尺寸调整的图片
    gt_boxes : shape=（n, 5）  5:[x1,y1,x2,y2,kind]
    im_info  : shape=(1,3) 3:w,h,scale
    img_id   : 00026

    '''


    data = torch.from_numpy(blobs['data'])
    im_info = torch.from_numpy(blobs['im_info'])
    # we need to random shuffle the bounding box.
    data_height, data_width = data.size(1), data.size(2)
    if self.training:
        np.random.shuffle(blobs['gt_boxes'])
        gt_boxes = torch.from_numpy(blobs['gt_boxes'])

        ########################################################
        # padding the input image to fixed size for each group #
        ########################################################

        # NOTE1: need to cope with the case where a group cover both conditions. (done)
        # NOTE2: need to consider the situation for the tail samples. (no worry)
        # NOTE3: need to implement a parallel data loader. (no worry)
        # get the index range

        # if the image need to crop, crop to the target size.
        ratio = self.ratio_list_batch[index]

        if self._roidb[index_ratio]['need_crop']:
            if ratio < 1:
                # this means that data_width << data_height, we need to crop the
                # data_height
                min_y = int(torch.min(gt_boxes[:,1]))
                max_y = int(torch.max(gt_boxes[:,3]))
                trim_size = int(np.floor(data_width / ratio))
                if trim_size > data_height:
                    trim_size = data_height                
                box_region = max_y - min_y + 1
                if min_y == 0:
                    y_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        y_s_min = max(max_y-trim_size, 0)
                        y_s_max = min(min_y, data_height-trim_size)
                        if y_s_min == y_s_max:
                            y_s = y_s_min
                        else:
                            y_s = np.random.choice(range(y_s_min, y_s_max))
                    else:
                        y_s_add = int((box_region-trim_size)/2)
                        if y_s_add == 0:
                            y_s = min_y
                        else:
                            y_s = np.random.choice(range(min_y, min_y+y_s_add))
                # crop the image
                data = data[:, y_s:(y_s + trim_size), :, :]

                # shift y coordiante of gt_boxes
                gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

                # update gt bounding box according the trip
                gt_boxes[:, 1].clamp_(0, trim_size - 1)
                gt_boxes[:, 3].clamp_(0, trim_size - 1)

            else:
                # this means that data_width >> data_height, we need to crop the
                # data_width
                min_x = int(torch.min(gt_boxes[:,0]))
                max_x = int(torch.max(gt_boxes[:,2]))
                trim_size = int(np.ceil(data_height * ratio))
                if trim_size > data_width:
                    trim_size = data_width                
                box_region = max_x - min_x + 1
                if min_x == 0:
                    x_s = 0
                else:
                    if (box_region-trim_size) < 0:
                        x_s_min = max(max_x-trim_size, 0)
                        x_s_max = min(min_x, data_width-trim_size)
                        if x_s_min == x_s_max:
                            x_s = x_s_min
                        else:
                            x_s = np.random.choice(range(x_s_min, x_s_max))
                    else:
                        x_s_add = int((box_region-trim_size)/2)
                        if x_s_add == 0:
                            x_s = min_x
                        else:
                            x_s = np.random.choice(range(min_x, min_x+x_s_add))
                # crop the image
                data = data[:, :, x_s:(x_s + trim_size), :]

                # shift x coordiante of gt_boxes
                gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                # update gt bounding box according the trip
                gt_boxes[:, 0].clamp_(0, trim_size - 1)
                gt_boxes[:, 2].clamp_(0, trim_size - 1)

        # based on the ratio, padding the image.
        if ratio < 1:
            # this means that data_width < data_height
            trim_size = int(np.floor(data_width / ratio))

            padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)), \
                                             data_width, 3).zero_()

            padding_data[:data_height, :, :] = data[0]
            # update im_info
            im_info[0, 0] = padding_data.size(0)
            # print("height %d %d \n" %(index, anchor_idx))
        elif ratio > 1:
            # this means that data_width > data_height
            # if the image need to crop.
            padding_data = torch.FloatTensor(data_height, \
                                             int(np.ceil(data_height * ratio)), 3).zero_()
            padding_data[:, :data_width, :] = data[0]
            im_info[0, 1] = padding_data.size(1)
        else:
            trim_size = min(data_height, data_width)
            padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
            padding_data = data[0][:trim_size, :trim_size, :]
            # gt_boxes.clamp_(0, trim_size)
            gt_boxes[:, :4].clamp_(0, trim_size)
            im_info[0, 0] = trim_size
            im_info[0, 1] = trim_size


        # check the bounding box:
        not_keep = (gt_boxes[:,0] == gt_boxes[:,2]) | (gt_boxes[:,1] == gt_boxes[:,3])
        keep = torch.nonzero(not_keep == 0).view(-1)

        gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
        if keep.numel() != 0:
            gt_boxes = gt_boxes[keep]
            num_boxes = min(gt_boxes.size(0), self.max_num_box)
            gt_boxes_padding[:num_boxes,:] = gt_boxes[:num_boxes]
        else:
            num_boxes = 0

            # permute trim_data to adapt to downstream processing
        padding_data = padding_data.permute(2, 0, 1).contiguous()
        im_info = im_info.view(3)

        return padding_data, im_info, gt_boxes_padding, num_boxes


    # '''train
    #   im_data.shape=(b,3,512,512)
    #   im_info.sahpe=(b,3)
    #   gt_boxes.sahpe=(b,20,5)一应该是这张图上有20个gt，5分别为4个坐标加一个类别  前n个是真正的gt，后面20-n都是0
    #   num_boxes = (n)
    # '''

    else:
        data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
        im_info = im_info.view(3)

        gt_boxes = torch.FloatTensor([1,1,1,1,1])
        num_boxes = 0

        return data, im_info, gt_boxes, num_boxes

        # '''test
        #   im_data.shape=(b,3,512,512)
        #   im_info.sahpe=(b,3)
        #   gt_boxes.sahpe=(b,n,5)一应该是这张图上有n个gt，5分别为4个坐标加一个类别
        #   num_boxes = (b)
        # '''

  def __len__(self):
    return len(self._roidb)#10022
