from __future__ import absolute_import
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# --------------------------------------------------------
# Reorganized and modified by Jianwei Yang and Jiasen Lu
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr

from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import clip_boxes, bbox_overlaps_batch, bbox_transform_batch

import pdb

DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3


class _AnchorTargetLayer(nn.Module):
    """
        Assign anchors to ground-truth targets. Produces anchor classification
        labels and bounding-box regression targets.
    """
    def __init__(self, feat_stride, scales, ratios):
        '''

        :param feat_stride: 16
        :param scales: [8,16,32]
        :param ratios: [0.5,1,2]
        '''
        super(_AnchorTargetLayer, self).__init__()

        self._feat_stride = feat_stride#16
        self._scales = scales#[8,16,32]
        anchor_scales = scales
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(anchor_scales), ratios=np.array(ratios))).float()
        # anchors.shape=(9,4)应该是9中anchor的坐标和大小，这只是9个anchor的标准，还不是全图上面所有的anchor
        self._num_anchors = self._anchors.size(0)#9

        # allow boxes to sit over the edge by a small amount//允许盒子少量地坐在边缘上
        self._allowed_border = 0  # default is 0

    def forward(self, input):
        '''

            input[0]==#rpn_cls_score.shape=(b,2*9,w,h)也即是每个点9个anchor的前背景概率预测
            input[1]==#gt_boxes: shape=（b,20,5）应该不是每个图片都有20个目标，可能20取的是一个比较大的值，并不是20里面都有gt的数据
            input[2]==#im_info: shape=(b,3)  3=[W,H,2.2901]最后一个2.2901含义还不太清楚
            input[3]==#num_boxes: shape=(b) b=[k,j,...]应该是代表这个batch里面第一张图片上有k个gt，第二张图片上有j个gt
        '''


        # Algorithm:
        #
        # for each (H, W) location i
        #   generate 9 anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the 9 anchors
        # filter out-of-image anchors

        rpn_cls_score = input[0]#shape=(b,2*9,w,h)也即是每个点9个anchor的前背景概率预测
        gt_boxes = input[1]#shape=（b,20,5)
        im_info = input[2]#shape=(b,3)  3=[W,H,2.2901]最后一个2.2901含义还不太清楚
        num_boxes = input[3]#shape=(b) b=[k,j,...]应该是代表这个batch里面第一张图片上有k个gt，第二张图片上有j个gt

        # map of shape (..., H, W)
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)#w,h

        batch_size = gt_boxes.size(0)#b

        feat_height, feat_width = rpn_cls_score.size(2), rpn_cls_score.size(3)#w,h
        shift_x = np.arange(0, feat_width) * self._feat_stride
        shift_y = np.arange(0, feat_height) * self._feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        #shift_x.shape = shift_y.shape = (40,32)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())
        shifts = shifts.contiguous().type_as(rpn_cls_score).float()#shape=(w*h,4)

        A = self._num_anchors#9
        K = shifts.size(0)#w*h

        self._anchors = self._anchors.type_as(gt_boxes) # move to specific gpu.
        #self._anchors.shape=(9,4)
        all_anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        all_anchors = all_anchors.view(K * A, 4)#shape=(9*w*h,4)
        '''到此就得到了默认的anchor的坐标，应该是相对于原图的'''

        total_anchors = int(K * A)#=9*w*h

        #判断anchor有没超出图片边界
        keep = ((all_anchors[:, 0] >= -self._allowed_border) &
                (all_anchors[:, 1] >= -self._allowed_border) &
                (all_anchors[:, 2] < long(im_info[0][1]) + self._allowed_border) &
                (all_anchors[:, 3] < long(im_info[0][0]) + self._allowed_border))

        inds_inside = torch.nonzero(keep).view(-1)#inds_inside.shape=(n) 存的是需要保留的anchor的索引

        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]#shape=(n,4)保留了n个anchor//  n<=w*h*9
        '''
         对于38*50的特征图，产生17100个anchor,
         all_anchors.shape=(17100,4)
         keep.shape=(17100)
         inds_inside.shape=(5944)
         anchors.shape=(5944,4)
        '''

        # label: 1 is positive, 0 is negative, -1 is dont care
        labels = gt_boxes.new(batch_size, inds_inside.size(0)).fill_(-1)#labels.shape=(b,n) 值全为-1
        bbox_inside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()#shape=(b,n) 值全为0
        bbox_outside_weights = gt_boxes.new(batch_size, inds_inside.size(0)).zero_()#shape=(b,n) 值全为0

        overlaps = bbox_overlaps_batch(anchors, gt_boxes)
        # anchors.shape=(n,4)  gt_boxes.shape=(b,20,4)
        #overlaps.shape=(b, n, 20)

        max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
        #max_overlaps.shape=(b,n)里面存的是与这张图片上的gt的最大的iou
        #argmax_overlaps.shape=(b,n)里面存的是与其iou最大的gt的索引
        gt_max_overlaps, _ = torch.max(overlaps, 1)
        #gt_max_overlaps.shape=(b,20)里面存的是每一个gt（20个）与其iou最大的anchor的iou

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:#not False = True
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0 #将iou小于0.3的anchor设为负样本

        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)
        #keep.shape=(b,n)里面的值代表一个anchor匹配了几个gt

        if torch.sum(keep) > 0:
            labels[keep>0] = 1#keep>0 =shape(b,n)里面代表了哪些anchor匹配的gt数大于0（也就是存在不止一个的gt与其匹配， 里面的值全为0或者1），那么将其标记为正样本
            #todo:感觉这里没有太明白，不是应该与gt的iou大于阈值才判定其为正样本吗，怎么现在与gt沾了一点半就认为是正样本了

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1#将与gt iou大于0.7 的anchor标记为正样本

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:#False
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0
        '''到目前为止，所有的anchor（n个）都被标记为正负样本了（还有ignore的）'''

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)#RPN_FG_FRACTION=0.5（前景anchor数量最多比例）  RPN_BATCHSIZE=256(rpn网络总共训练的anchor数目)
        #num_fg = 128

        sum_fg = torch.sum((labels == 1).int(), 1)
        sum_bg = torch.sum((labels == 0).int(), 1)
        #sum_fg.shape=(b) 代表每一张图片上有多少正样本
        #sum_bg.shape=(b) 代表每一张图片上有多少负样本

        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:#如果第i张图片上正样本的数量大于128
                fg_inds = torch.nonzero(labels[i] == 1).view(-1)#fg_inds.shape=(n) [0,1,3,6...]代表第0、1、3、6...这些anchor是正样本
                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()#产生一个随机向量（也就是将0-（n-1）打乱顺序）
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]#取出rand_num前j个随机数，作为索引取出fg_inds中相应位置的正样本anchor的序号（在labels里的序号）
                labels[i][disable_inds] = -1#去掉一些正样本（将其标记为-1）

#           num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]  #256-

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:#如果第i张图片上背景anchor数量大于
                bg_inds = torch.nonzero(labels[i] == 0).view(-1)
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1#去掉一些负样本（将其标记为-1）
        '''
        到目前为止已经将n个anchor标记为正负样本（已经考虑过正负样本过多的问题）  1：正样本  0：负样本  -1：ignore
        '''
        offset = torch.arange(0, batch_size)*gt_boxes.size(1)#gt_boxes.shape=(b,20,5)
        #offset=[0,1*20,2*20,3*20,....b*20]

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        #offset.view.shape=(b,1)=[[0],[1*20],[2*20]...[b*20]]  argmax_overlaps.shape=(b,n)里面存的是与其iou最大的gt的索引
        #argmax_overlaps.shape=(b,n)
        #todo:这里为什么要加上offset,可能要在后面转化为二维的张量来进行运算，（因为后面将其维度简化为（b*n）并且gt=（b*20,5）要从gt里面去的每张图片上
        # 每个anchor所对应的gt，所以要第k图片上的n要加上20*k，才能从gt（b*20,5）里面取出对应的gt）


        bbox_targets = _compute_targets_batch(anchors, gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5))
        #anchors.shape=(n,4)保留了n个anchor//  n<=w*h*9
        #gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5)
        #gt_boxes.shape=（b,20,5)  gt_boxes.view(-1,5)=(b*20,5)   argmax_overlaps.view(-1)=（b*n） gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :]=(b*n,5)
        #gt_boxes.view(-1,5)[argmax_overlaps.view(-1), :].view(batch_size, -1, 5)=(b,n,5)#相当于将原来的gt.shape=(b,20,5)进行了扩展，使其维度与anchor.shape=(b,20,5)一致
        #以便将每一个anchor和其匹配的gt对应起来，方便计算相对回归值
        '''bbox_targets.shape=(b,n,4)到这里得到了每一个anchor的回归值'''

        # use a single value instead of 4 values for easy index.
        #bbox_inside_weights.shape=(b,n)全为0， labels.shape=(b,n)标记了anchor是否为正负样本或者ignore  cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]=1.0
        bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]#将positive的anchor的权重设置为1.0
        '''这里inside权重是论文中求回归loss公式中 求和符号里面的pi*，其实代表正样本算回归损失（inside权重为1），负样本不算回归损失（inside权重为0） 算不算loss有inside权重控制'''

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:#RPN_POSITIVE_WEIGHT=-1
            num_examples = torch.sum(labels[i] >= 0)#得到labels[i]里值大于等于0的个数，也就是第i张图片里的正负样本的总数，假设总数为k
            positive_weights = 1.0 / num_examples.item() #1/k
            negative_weights = 1.0 / num_examples.item() #1/k
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        ##bbox_outside_weights.shape=(b,n) 值全为0   #labels.shape=(b,n)值为1,0,-1
        bbox_outside_weights[labels == 1] = positive_weights #将正样本的权重设置为1/k  k是正负样本的总数
        bbox_outside_weights[labels == 0] = negative_weights #将负样本的权重设置为1/k  k是正负样本的总数
        '''这里outside权重是论文中求回归loss公式中，求和符号外面的Nreg， 代表将回归损失的和做一个归一化，也就是除以正负样本的总数'''

        labels = _unmap(labels, total_anchors, inds_inside, batch_size, fill=-1)
        #参数：labels.shape=(b,n)里面值为1，0，-1  total_anchors=9*w*h  #inds_inside.shape=(n)存的是需要保留的anchor的索引(没超过图象边界的anchor)  batchsize=b
        #labels.shape=(b,9*w*h)
        '''使用upmap的原因：  本来一张图上anchor的个数是9*w*h ，但是上面我们分配标签的时候只对其中n个anchor（n<9*w*h， 也就是没有超出图像边界的n个anchor）
            进行了判别正负样本，但是我们需要所有anchor的标签，所有用了upmap函数，将其与的9*w*h-n个anchor标记为-1（ignore），其他的n个anchor就用上面判断好的标签'''

        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, batch_size, fill=0)
        #参数：bbox_targets.shape=(b,n,4)到这里得到了每一个anchor的回归值 , total_anchors=9*w*h, inds_inside.shape=(n)存的是需要保留的anchor的索引(没超过图象边界的anchor) , batch_size=b
        #bbox_targets.shape=(b,9*w*h,4)  将剩余的anchor（9*w*h-n个）回归值设置为0
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, batch_size, fill=0)
        # bbox_inside_weights.shape=(b,9*w*h) 将除了被标记为正样本的其他anchor的权重设置为0
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, batch_size, fill=0)
        # bbox_outside_weights.shape=(b,9*w*h) 将除了已经标记好正负样本的（n个anchor里面的）其他anchor权重设置为0


        '''
        到这里我们为每张图上9*w*h个anchor都分配了标签（1：正 0：负  -1：ignore） 以及为其中的n个anchor（包含正、负、ignore）匹配的回归值
        '''



        outputs = []

        labels = labels.view(batch_size, height, width, A).permute(0,3,1,2).contiguous()#shape=(b,9,h,w)
        labels = labels.view(batch_size, 1, A * height, width)#shape=(b,1,9*h,w)
        outputs.append(labels)

        bbox_targets = bbox_targets.view(batch_size, height, width, A*4).permute(0,3,1,2).contiguous()#shape=(b,9*4,h,w)
        outputs.append(bbox_targets)

        anchors_count = bbox_inside_weights.size(1)#9*w*h
        bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
        #bbox_inside_weights.view(batch_size,anchors_count,1) .sahpe=(b,9*w*h,1)  --->(b,9*w*h,4) 4：都是同一个值

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        #shape=(b,h,w,4*9) -->(b,4*9, h,w)

        outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)#shape=(b, 9*w*h,4) 4:同一个值
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        #shape=(b,h,w,4*9)--->(b,4*9,h,w)
        outputs.append(bbox_outside_weights)

        return outputs

    '''
    outputs=[   labels.shape=(b,1,9*h,w),   所有anchor的标签  1：正样本  0：负样本  -1：ignore
                bbox_targets.shape=(b,9*4,h,w),  所有anchor的目标回归值
                bbox_inside_weights.shape=(b,4*9, h,w), 所有anchor的回归inside权重
                bbox_outside_weights.shape=(b,4*9,h,w)] 所有anchor的回归outside权重
                
    
    
    '''

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _unmap(data, count, inds, batch_size, fill=0):
    '''

    :param data: labels.shape=(b,n)里面值为1，0，-1
    :param count: total_anchors=9*w*h
    :param inds: #inds_inside.shape=(n)存的是需要保留的anchor的索引(没超过图象边界的anchor)
    :param batch_size: batchsize=b
    :param fill: -1
    :return:
    '''
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """


    # 参数：labels.shape=(b,n)里面值为1，0，-1  total_anchors=9*w*h  #inds_inside.shape=(n)存的是需要保留的anchor的索引(没超过图象边界的anchor)  batchsize=b

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)#shape=(b,9*w*h)  值全为-1
        ret[:, inds] = data#将没超过边界的anchor（保留的anchor）的标签标记为分配好的标签
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
