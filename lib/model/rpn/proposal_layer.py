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
import math
import yaml
from model.utils.config import cfg
from .generate_anchors import generate_anchors
from .bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch
from lib.model.nms.nms_wrapper import nms

import pdb

DEBUG = False

class _ProposalLayer(nn.Module):
    """
    Outputs object detection proposals by applying estimated bounding-box
    transformations to a set of regular boxes (called "anchors").
    """

    def __init__(self, feat_stride, scales, ratios):
        '''

        :param feat_stride: 16
        :param scales: [8,16,32]
        :param ratios: [0.5,1,2]
        '''
        super(_ProposalLayer, self).__init__()

        self._feat_stride = feat_stride
        self._anchors = torch.from_numpy(generate_anchors(scales=np.array(scales), 
            ratios=np.array(ratios))).float()
        #anchors.shape=(9,4)应该是9中anchor的坐标和大小，这只是9个anchor的标准，还不是全图上面所有的anchor
        self._num_anchors = self._anchors.size(0)#9

        # rois blob: holds R regions of interest, each is a 5-tuple
        # (n, x1, y1, x2, y2) specifying an image batch index n and a
        # rectangle (x1, y1, x2, y2)
        # top[0].reshape(1, 5)
        #
        # # scores blob: holds scores for R regions of interest
        # if len(top) > 1:
        #     top[1].reshape(1, 1, 1, 1)

    def forward(self, input):


        #input=(rpn_cls_prob.data, rpn_bbox_pred.data,im_info, cfg_key)
        #rpn_cls_prob=(b,2*9,w,h)  rpn_bbox_pred=(b,4*9,w,h)  im_info=(b,3)=[[w,h,3],[..]](这里wh是原图的尺寸)   cfg_key=‘train’or‘test’

        # Algorithm:
        #
        # for each (H, W) location i
        #   generate A anchor boxes centered on cell i
        #   apply predicted bbox deltas at cell i to each of the A anchors
        # clip predicted boxes to image
        # remove predicted boxes with either height or width < threshold
        # sort all (proposal, score) pairs by score from highest to lowest
        # take top pre_nms_topN proposals before NMS
        # apply NMS with threshold 0.7 to remaining proposals
        # take after_nms_topN proposals after NMS
        # return the top proposals (-> RoIs top, scores top)


        # the first set of _num_anchors channels are bg probs
        # the second set are the fg probs
        '''

        :param input:
        :return:
        '''
        '''
        这里的rois(proposal)产生的过程：
        1：先生成9*w*h个anchor的坐标--->(b,w*h*9,4)
        2：根据预测的9*w*h个anchor的回归值对所有的anchor进行位置调整（超出边界的框进行修剪）
        3：然后针对batch中的每一张图片进行：
            1：按照前景分数取出9*w*h中前12000个框的分数以及他们的位置box（test：6000）
            2：对这12000（train）个box进行nms，取出nms之后剩下的框里面的前2000个box的位置和分数（按照分数取）
            3：将每张图片的2000个保留的box合并到一起（合到一个batch里面）
        4：返回该batch保留的box（b,2000,5）
        
        '''

        scores = input[0][:, self._num_anchors:, :, :]#shape=(b,9,w,h)取出预测的所有的anchor的前景概率
        bbox_deltas = input[1]#=(b,4*9,w,h)
        im_info = input[2]#=（b，3）
        cfg_key = input[3]#=train

        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N#train:12000   test:6000
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N#train:2000   test:300
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH#train:0.7        test:0.7
        min_size      = cfg[cfg_key].RPN_MIN_SIZE#train:8 rpn的最小尺寸 test:16

        batch_size = bbox_deltas.size(0)

        feat_height, feat_width = scores.size(2), scores.size(3)#h,w
        shift_x = np.arange(0, feat_width) * self._feat_stride#[0,16,16*2,16*3,...16*h]
        shift_y = np.arange(0, feat_height) * self._feat_stride#[0,16,16*2,16*3,...16*w]
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                  shift_x.ravel(), shift_y.ravel())).transpose())#shifts.shape=(w*h,4)坐标是相对于原图的  4:[x,y,x,y]
        shifts = shifts.contiguous().type_as(scores).float()#shifts.shape=(w*h,4)坐标是相对于原图的

        A = self._num_anchors#9
        K = shifts.size(0)#w*h

        self._anchors = self._anchors.type_as(scores)#shape=(9,4)每个位置9个anchor的尺寸
        # anchors = self._anchors.view(1, A, 4) + shifts.view(1, K, 4).permute(1, 0, 2).contiguous()
        anchors = self._anchors.view(1, A, 4) + shifts.view(K, 1, 4)
        anchors = anchors.view(1, K * A, 4).expand(batch_size, K * A, 4)
        #anchors.shape=(b,w*h*9,4)
        #到这里就产生了默认的anchor

        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:

        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()#shape=(b,w,h,4*9)
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)#shape=(b,w*h*9,4)

        # Same story for the scores:
        scores = scores.permute(0, 2, 3, 1).contiguous()#(b,w,h,9)
        scores = scores.view(batch_size, -1)#shape=(b,w*h*9)

        # Convert anchors into proposals via bbox transformations
        proposals = bbox_transform_inv(anchors, bbox_deltas, batch_size)
        #proposal.shape=(b,w*h*9,4) 根据预测出来的偏移调整acnhor的位置， 4：[x1,y1,x2,y2]两个角点的坐标

        # 2. clip predicted boxes to image
        proposals = clip_boxes(proposals, im_info, batch_size)
        # proposals = clip_boxes_batch(proposals, im_info, batch_size)

        # assign the score to 0 if it's non keep.
        # keep = self._filter_boxes(proposals, min_size * im_info[:, 2])

        # trim keep index to make it euqal over batch
        # keep_idx = torch.cat(tuple(keep_idx), 0)

        # scores_keep = scores.view(-1)[keep_idx].view(batch_size, trim_size)
        # proposals_keep = proposals.view(-1, 4)[keep_idx, :].contiguous().view(batch_size, trim_size, 4)
        
        # _, order = torch.sort(scores_keep, 1, True)
        
        scores_keep = scores#shape=(b,w*h*9)
        proposals_keep = proposals#shape=(b,w*h*9,4) 根据预测出来的偏移调整的acnhor的位置， 4：[x1,y1,x2,y2]两个角点的坐标
        _, order = torch.sort(scores_keep, 1, True)
        #order.shape=[b,w*h*9] w*h*9个数[2,1,0,3,,,]表示分数从高到低排序，各个框的idx（分数第2>第1》第0个》第3个框。。。。）

        output = scores.new(batch_size, post_nms_topN, 5).zero_()#shape=(b,2000,5) 全零
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i]#shape=(w*h*9,4) 根据预测出来的偏移调整acnhor的位置， 4：[x1,y1,x2,y2]两个角点的坐标（取出每张图片上的回归之后的anchor）
            scores_single = scores_keep[i]#shape=(w*h*9)每张图片上预测的anchor的前景概率

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]#order_single.shape=（w*h*9） w*h*9个数[2,1,0,3,,,]表示分数从高到低排序，各个框的idx（分数第2>第1》第0个》第3个框。。。。）
            #到此位置取出了一张图片上的所有的调整之后的anchor，以及按分数排序的索引

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]#order_single.shape=（12000）只取出高分前12000个框的索引

            proposals_single = proposals_single[order_single, :]##shape=(12000,4)根据索引取出高分的12000个框的坐标
            scores_single = scores_single[order_single].view(-1,1)##shape=(12000,1)根据索引取出高分的12000个框的分数

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)

            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)#shape=(k,1),每个值代表要保留的bbox的索引
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]#只取nms之后保留的 前2000个  shape=(2000) ,里面的值代表要保留的box的索引
            proposals_single = proposals_single[keep_idx_i, :]#shape=（2000,4）
            scores_single = scores_single[keep_idx_i, :]#shape=（2000）

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)#=2000，得到的proposal的个数
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single
            #output.shape=(b,2000,5)  5:[当前box在那个图片上（0-batchsize）， x1,y1,x2,y2]

        return output

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _filter_boxes(self, boxes, min_size):
        """Remove all boxes with any side smaller than min_size."""
        ws = boxes[:, :, 2] - boxes[:, :, 0] + 1
        hs = boxes[:, :, 3] - boxes[:, :, 1] + 1
        keep = ((ws >= min_size.view(-1,1).expand_as(ws)) & (hs >= min_size.view(-1,1).expand_as(hs)))
        return keep
