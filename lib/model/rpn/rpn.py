from __future__ import absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from lib.model.utils.config import cfg
from .proposal_layer import _ProposalLayer
from .anchor_target_layer import _AnchorTargetLayer
from lib.model.utils.net_utils import _smooth_l1_loss

import numpy as np
import math
import pdb
import time

class _RPN(nn.Module):
    """ region proposal network """
    def __init__(self, din):
        super(_RPN, self).__init__()
        
        self.din = din  # get depth of input feature map, e.g., 512
        self.anchor_scales = cfg.ANCHOR_SCALES#[8,16,32]
        self.anchor_ratios = cfg.ANCHOR_RATIOS#[0.5,1,2]
        self.feat_stride = cfg.FEAT_STRIDE[0]#16

        # define the convrelu layers processing input feature map
        self.RPN_Conv = nn.Conv2d(self.din, 512, 3, 1, 1, bias=True)

        # define bg/fg classifcation score layer
        self.nc_score_out = len(self.anchor_scales) * len(self.anchor_ratios) * 2 # 2(bg/fg) * 9 (anchors)
        self.RPN_cls_score = nn.Conv2d(512, self.nc_score_out, 1, 1, 0)

        # define anchor box offset prediction layer
        self.nc_bbox_out = len(self.anchor_scales) * len(self.anchor_ratios) * 4 # 4(coords) * 9 (anchors)
        self.RPN_bbox_pred = nn.Conv2d(512, self.nc_bbox_out, 1, 1, 0)

        # define proposal layer
        self.RPN_proposal = _ProposalLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.RPN_anchor_target = _AnchorTargetLayer(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    def forward(self, base_feat, im_info, gt_boxes, num_boxes):
        '''

        :param base_feat: shape=(b,1024,w,h)特征提取的输出
        :param im_info: shape=(b,3)  3=[W,H,2.2901]最后一个2.2901含义还不太清楚
        :param gt_boxes: shape=（b,20,5）应该不是每个图片都有20个目标，可能20取的是一个比较大的值，并不是20里面都有gt的数据
        :param num_boxes:shape=(b) b=[k,j,...]应该是代表这个batch里面第一张图片上有k个gt，第二张图片上有j个gt
        :return:
        '''
        #base_feat.shape=(b,1024,w,h)
        batch_size = base_feat.size(0)

        # return feature map after convrelu layer
        rpn_conv1 = F.relu(self.RPN_Conv(base_feat), inplace=True)#shape=(b,512,w,h)
        # get rpn classification score
        rpn_cls_score = self.RPN_cls_score(rpn_conv1)#shape=(b,2*9,w,h)也即是每个点9个anchor的前背景概率预测

        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)#shape=(b,2,9*w,h)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, 1)#shape=(b,2,9*w,h)
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out)#shape=(b,2*9,w,h)

        # get rpn offsets to the anchor boxes
        rpn_bbox_pred = self.RPN_bbox_pred(rpn_conv1)#shape=(b,4*9,w,h)

        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'

        '''
        得到vgg的特征图之后，分别用两个卷积预测每一个anchor的分数（前背景。shape=(b,2*9,w,h)） 以及每一个anchor的回归值（shape=(b,4*9,w,h)）
        '''


        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data,
                                 im_info, cfg_key))
        # rois=output.shape=(b,2000,5)  5:[当前box在那个图片上（0-batchsize）,x1,y1,x2,y2]
        '''
        这里的rois产生的过程：
        1：rpn预测每一个anchor的分数以及回归值
        2：self.RPN_proposal模块先根据预测的回归值调整初始的anchor
            然后取出调整之后的anchor 分数前12000个box，将这12000个box计算nms，，得到保留的box 的索引
            然后根据nms得到的索引取出保留的k个box的坐标以及他们的前景分数
            在按照分数从k个box（nms之后还保留的box）取出分数前2000个框的坐标
        3：最后得到rpn 网络proposal 的结果（2000个roi）
        
        '''

        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None

            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, im_info, num_boxes))
            #rpn_cls_score.shape=(b,2*9,w,h)也即是每个点9个anchor的前背景概率预测
            #gt_boxes: shape=（b,20,5）应该不是每个图片都有20个目标，可能20取的是一个比较大的值，并不是20里面都有gt的数据
            #im_info: shape=(b,3)  3=[W,H,2.2901]最后一个2.2901含义还不太清楚
            #num_boxes: shape=(b) b=[k,j,...]应该是代表这个batch里面第一张图片上有k个gt，第二张图片上有j个gt
            '''
            rpn_data = outputs=[   labels.shape=(b,1,9*h,w),   所有anchor的标签  1：正样本  0：负样本  -1：ignore
                                    bbox_targets.shape=(b,9*4,h,w),  所有anchor的目标回归值
                                    bbox_inside_weights.shape=(b,4*9, h,w), 所有anchor的回归inside权重
                                    bbox_outside_weights.shape=(b,4*9,h,w)所有anchor的回归outside权重
                                ]
                                
            self.RPN_anchor_target 函数为每一个anchor（9*w*h个anchor）(是为原始的anchor匹配的标签，而不是经过预测调整之后的achor)匹配到了标签，（1：正样本  0：负样本  -1：ignore），以及他们的回归target
                                同时已经处理了正负样本过多的问题（将一些标记为-1）
            
            '''



            # compute classification loss

            # rpn_cls_score_reshape.shape=(b,2,9*w,h)
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)#shape=(b,9*w*h,2)也即是每个点9个anchor的前背景概率预测
            rpn_label = rpn_data[0].view(batch_size, -1)#rpn_label.shape=(b,9*h*w)标记了每个anchor的标签

            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))#shape=
            '''
            rpn_label=[[ 1.,  1.,  0.,  1., -1.,  0., -1.,  1.,  0.],
                        [ 0.,  0.,  0., -1.,  1.,  1., -1.,  0.,  0.]]
            rpn_keep = [ 0,  1,  2,  3,  5,  7,  8,  9, 10, 11, 13, 14, 16, 17] 也就是将rpn_label先拉平，然后将里面非-1的值的索引取出，也就是将正、负样本的索引取出
            '''

            rpn_cls_score = torch.index_select(rpn_cls_score.view(-1,2), 0, rpn_keep)
            #rpn_cls_score.view(-1, 2).shape=(b*9*h*w,2)  假设rpn_keep.shape=(k)也就是正样本和负样本总数为k，里面存了正负样本的索引
            #rpn_cls_score.shape=(k) 将这个batch里面所有图片正负样本的预测分数找出来（正、负样本数量共k个）
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
            #rpn_label.shape=(k) 将这个batch里面所有的图片的正负样本的标签取出， k:[1,1,0,1,0,0...]
            rpn_label = Variable(rpn_label.long())
            self.rpn_loss_cls = F.cross_entropy(rpn_cls_score, rpn_label)
            '''算出了rpn 的分类loss'''
            fg_cnt = torch.sum(rpn_label.data.ne(0))#正样本的数量（统计rpn_label里面非0的元素的个数）

            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
            #rpn_bbox_targets = bbox_targets.shape = (b, 9 * 4, h, w), 所有anchor的目标回归值
            #rpn_bbox_inside_weights = bbox_inside_weights.shape = (b, 4 * 9, h, w), 所有anchor的回归inside权重  值为0或1，正样本为1，负样本为0，代表负样本不算回归损失
            #rpn_bbox_outside_weights = bbox_outside_weights.shape = (b, 4 * 9, h, w) 值为0或1/k  k是正、负样本的总数（一个batch里面的总数）
            # 所有anchor的回归outside权重


            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)

            ##rpn_bbox_pred.shape=(b,4*9,w,h)预测的anchor的回归值
            self.rpn_loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                            rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])
            #self.rpn_loss_box=(2.36)  是一个值，代表一个batch里面各个图片上的回归损失求的平均

        return rois, self.rpn_loss_cls, self.rpn_loss_box
    #rois=output.shape=(b,2000,5)  5:[当前box在那个图片上（0-batchsize）,x1,y1,x2,y2]
    #rpn_loss_cls：rpn网络的分类loss   只计算了正样本和负样本的分类loss（根据匹配的标签，先得到正样本和负样本的索引，然后从预测的分数中取出
    #        正样本和负样本的分类分数，然后在从匹配的标签中取出正样本和负样本的标签（1,0），然后两者之间算交叉熵loss）
    #rpn_loss_box=(2.36)  是一个值，代表一个batch里面各个图片上的回归损失求的平均
