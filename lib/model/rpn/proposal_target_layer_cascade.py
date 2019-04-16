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
from ..utils.config import cfg
from .bbox_transform import bbox_overlaps_batch, bbox_transform_batch
import pdb

class _ProposalTargetLayer(nn.Module):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    将rpn输出的rois=(b,2000,5)2000个roi分配gt ，产生分类和回归的标签
    """

    def __init__(self, nclasses):
        super(_ProposalTargetLayer, self).__init__()
        self._num_classes = nclasses
        self.BBOX_NORMALIZE_MEANS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)#=[0.0, 0.0, 0.0, 0.0]
        self.BBOX_NORMALIZE_STDS = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS)#=[0.1, 0.1, 0.2, 0.2]
        self.BBOX_INSIDE_WEIGHTS = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS)#=[1.0, 1.0, 1.0, 1.0]

    def forward(self, all_rois, gt_boxes, num_boxes):
        # all_rois=rois.shape=(b,2000,5) 5:[当前box在那个图片上（0-batchsize）,x1,y1,x2,y2]
        # gt_boxes: shape =（b, 20, 5）应该不是每个图片都有20个目标，可能20取的是一个比较大的值，并不是20里面都有gt的数据（固定的（b, 20, 5）维度）
        # num_boxes:shape=(b) b=[k,j,...]应该是代表这个batch里面第一张图片上有k个gt，第二张图片上有j个gt

        '''

        :param all_rois:
        :param gt_boxes:
        :param num_boxes:
        :return:
        '''
        '''
        函数功能：从rpn网络提供的（b,2000,5）个proposal里面选出128个作为roi用来给后面的rcnn网络，同时计算出这128个roi对应的回归和分类target
        
        
        '''

        self.BBOX_NORMALIZE_MEANS = self.BBOX_NORMALIZE_MEANS.type_as(gt_boxes)
        self.BBOX_NORMALIZE_STDS = self.BBOX_NORMALIZE_STDS.type_as(gt_boxes)
        self.BBOX_INSIDE_WEIGHTS = self.BBOX_INSIDE_WEIGHTS.type_as(gt_boxes)

        gt_boxes_append = gt_boxes.new(gt_boxes.size()).zero_()#shape=(b,20,5)  全0
        gt_boxes_append[:,:,1:5] = gt_boxes[:,:,:4]  #shape=(b,20,5)  5:[0,x1,y1,x2,y2]

        # Include ground-truth boxes in the set of candidate rois
        all_rois = torch.cat([all_rois, gt_boxes_append], 1)#shape=(b,2020,5) 将gt加入到候选rois里面
        '''
        解释一下为什么要将20个gt加到2000个proposal里面：
        
        刚开始训练的时候，如果仅仅使用RPN生成的2000个proposal，这些proposal的质量肯定会很差，这样的话后面rcnn部分就会得不到训练或者训练很差（因为rpn
        提供的propodal质量很差），所以刚开始训练的时候会把gt也加入到rpn的proposal里面（把gt当成proposal），这样刚开始训练的时候最起码有一点质量好的proposal（其实是gt）
        ，这样就使得后面rcnn得到好的训练。
        你与可能会这样想，把gt加入到proposal里面，后面选择128*0.25=32个正样本的时候肯定会把这些gt选进去，其实不是，后面选的规则是：
        先找到2020个proposal里面的这个样本（iou大于阈值），然后从这些正样本中随机选的32个，所以并不一定会把20个gt都选进去。
        
        '''

        num_images = 1
        rois_per_image = int(cfg.TRAIN.BATCH_SIZE / num_images)#cfg.TRAIN.BATCH_SIZE=128  # Minibatch size (number of regions of interest [ROIs])
        fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))# __C.TRAIN.FG_FRACTION = 0.25正样本的比例   Fraction of minibatch that is labeled foreground (i.e. class > 0)
        #fg_rois_per_image=128*0.25=32 每张图上应该取多少前景rois
        fg_rois_per_image = 1 if fg_rois_per_image == 0 else fg_rois_per_image

        labels, rois, bbox_targets, bbox_inside_weights = self._sample_rois_pytorch(
            all_rois, gt_boxes, fg_rois_per_image,
            rois_per_image, self._num_classes)
        '''
        labels=labels_batch.shape=(b, 128) 记录了每张图片上128个样本（正、负）的类别target（就是与其iou最大的gt的类别，精确到哪一类，而不是前背景类）
        rois = rois_batch.shape=(b, 128, 5) 记录了这128个roi的box（是从rpn预测出来的2020个proposal里面选出的128个box） 5:第一个数i代表该图片是该batch中的第i张图片
        bbox_targets.shape=(b,128,4) 存了正样本和负样本的回归target（负样本的回归目标在 self._get_bbox_regression_labels_pytorch函数中被设置成0了）

        bbox_inside_weights.shape=(b,128,4) 4=[1,1,1,1]或者［0,0,0,0］ 正样本的权重是[1,1,1,1] 负样本的权重是[0,0,0,0]

        '''

        bbox_outside_weights = (bbox_inside_weights > 0).float()#好像bbox_inside_weights和bbox_outside_weights是相同的，并没有变化
        #bbox_outside_weights.shape=shape=(b,128,4) 4=[1,1,1,1]或者［0,0,0,0］ 正样本的权重是[1,1,1,1] 负样本的权重是[0,0,0,0]

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights
    '''
    rois = rois_batch.shape=(b, 128, 5) 记录了这128个roi的box（是从rpn预测出来的2020个proposal里面选出的128个box） 5:第一个数i代表该图片是该batch中的第i张图片
    labels=labels_batch.shape=(b, 128) 记录了每张图片上128个样本（正、负）的类别target（就是与其iou最大的gt的类别，精确到哪一类，而不是前背景类）
    bbox_targets.shape=(b,128,4) 存了正样本和负样本的回归target（负样本的回归目标在 self._get_bbox_regression_labels_pytorch函数中被设置成0了）
    box_inside_weights.shape=(b,128,4) 4=[1,1,1,1]或者［0,0,0,0］ 正样本的权重是[1,1,1,1] 负样本的权重是[0,0,0,0]
    bbox_outside_weights.shape=(b,128,4) 4=[1,1,1,1]或者［0,0,0,0］ 正样本的权重是[1,1,1,1] 负样本的权重是[0,0,0,0]


    
    
    
    '''

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

    def _get_bbox_regression_labels_pytorch(self, bbox_target_data, labels_batch, num_classes):
        '''
        这个函数的作用是将负样本的回归target抛弃（因为之前计算rpn输出的2020个porposal的回归值是2020个roi全都计算，然后选出128个正样本和负样本的回归target放入bbox_target_data，
        所以这里要把负样本的回归target抛弃），同时得到回归全权重。

        :param bbox_target_data: #bbox_target_data.shape=(b,128,4)得到了128个rois的回归值
        :param labels_batch: shape=(b, 128) 记录了每个roi匹配的gt的类别（具体到某一类）
        :param num_classes: 21
        :return:
        '''
        """Bounding-box regression targets (bbox_target_data) are stored in a
        compact form b x N x (class, tx, ty, tw, th)

        This function expands those targets into the 4-of-4*K representation used
        by the network (i.e. only one class has non-zero targets).

        Returns:
            bbox_target (ndarray): b x N x 4K blob of regression targets
            bbox_inside_weights (ndarray): b x N x 4K blob of loss weights
        """
        batch_size = labels_batch.size(0)#b
        rois_per_image = labels_batch.size(1)#128
        clss = labels_batch# shape=(b, 128) 记录了每个roi匹配的gt的类别（具体到某一类）
        bbox_targets = bbox_target_data.new(batch_size, rois_per_image, 4).zero_()#shape=(b,128,4) 全零
        bbox_inside_weights = bbox_target_data.new(bbox_targets.size()).zero_()#shape =（b,128,4）全零

        for b in range(batch_size): #b= 0,1,2....b-1
            # assert clss[b].sum() > 0
            if clss[b].sum() == 0:
                continue
            inds = torch.nonzero(clss[b] > 0).view(-1)#clss[b].shape=(k) =[0,2,6,...]代表128第0,2,6..个roi是正样本，存的是正样本的索引
            for i in range(inds.numel()):#inds.numel()=k 正样本的个数
                ind = inds[i]#
                bbox_targets[b, ind, :] = bbox_target_data[b, ind, :]#这里只将正样本的回归值保留了，也就是说将前面算出来的负样本的回归值记为0
                bbox_inside_weights[b, ind, :] = self.BBOX_INSIDE_WEIGHTS#BBOX_INSIDE_WEIGHTS=[1.0, 1.0, 1.0, 1.0] 这里只将正样本的回归权重记为1

        return bbox_targets, bbox_inside_weights
    #bbox_targets.shape=(b,128,4) 从bbox_target_data（（b,128,4）存了正样本和负样本的回归target（由于前面计算回归target是2020个proposal都计算，所以这里面负样本也有回归target，所以这里要把负样本的回归target清零））
    #                               将正样本的回归target取出放到bbox_targets里（bbox_targets原来为0）
    #bbox_inside_weights.shape=(b,128,4) 4=[1,1,1,1]或者［0,0,0,0］ 正样本的权重是[1,1,1,1] 负样本的权重是[0,0,0,0]



    def _compute_targets_pytorch(self, ex_rois, gt_rois):
        """Compute bounding-box regression targets for an image."""

        assert ex_rois.size(1) == gt_rois.size(1)
        assert ex_rois.size(2) == 4
        assert gt_rois.size(2) == 4

        batch_size = ex_rois.size(0)
        rois_per_image = ex_rois.size(1)

        targets = bbox_transform_batch(ex_rois, gt_rois)

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            targets = ((targets - self.BBOX_NORMALIZE_MEANS.expand_as(targets))
                        / self.BBOX_NORMALIZE_STDS.expand_as(targets))

        return targets


    def _sample_rois_pytorch(self, all_rois, gt_boxes, fg_rois_per_image, rois_per_image, num_classes):
        '''
        #todo:为什么要将gt加入大proposal里面（proposal ：2000， gt ： 20）---》看多实验结果，就是2020（）（讲解见上面）
        :param all_rois: shape=(b,2020,5) 将gt加入到候选rois里面  原来roi=（b,2000,5） gt=(b,20,5)
        :param gt_boxes: shape =（b, 20, 5）应该不是每个图片都有20个目标，可能20取的是一个比较大的值，并不是20里面都有gt的数据（固定的（b, 20, 5）维度）
        :param fg_rois_per_image: =128*0.25=32 每张图上应该取多少前景rois
        :param rois_per_image: =128 一个是一张图片上取多少roi给后面的rcnn
        :param num_classes: =21  类别数
        :return:
        '''
        '''
        此函数目的：计算rpn网络得到的2020（2000）个proposal（rois） 的回归target和类别target（后面训练rcnn用到）
        过程：
            1：先计算2020个proposal与 20个gt 相互间的iou，得到
                #max_overlaps.shape=(b,2020) 记录了每个rois 与所有gt最大的iou值
                #gt_assignment.shape=(b,2020) 记录了每个rois 与所有gt最大的iou的gt的索引
            2：然后根据最大iou信息得到每个porposal的类别target（与其iou最大的gt的类别就是此proposla的类别target）
            3：然后逐图片进行一下操作：
                1：得到这张图片上的前景样本（2020个proposal中，根据与gt最大iou 》 0.5 确定fg） 共k个（保存的是porposal的索引）
                2：得到这张图片上背景样本（2020个proposal中，根据gt最大iou 《0.5  》0.1 确定bg ） 共j个（保存的是porposal的索引）
                3：然后选出k个正样本（k《32）和128-k个负样本 （保存的是porposal的索引）
                4：将正样本和负样本合并 得到（128）张量，（保存的是porposal的索引，就是在2020个proposal中符合条件的proposal 的索引）
                5：将该张图片上正负样本的索引存在一个（b,128）的张量里面
                6：根据128个proposal 的索引取出这个128个porposla的类别信息
                7：将正负样本的索引对应的proposal的box存在一个（b，128,5）张量里面
                8：将选出来的128个正负样本各自最大iou的gt信息存储到gt_rois_batch里面，用来计算proposal box与gt的相对回归值
            4：计算128个选出来的porposal与他们对应的gt的回归值
            5：由于前面是通过正负样本的索引从2020个proposal取出128个roi（正负样本）的box，然后和所有的gt计算回归值，所以这里面负样本也计算了回归target，所以通过
                _get_bbox_regression_labels_pytorch函数将其中的负样本的回归target抛弃，只保留正样本的回归target，同时得到回归权重
        
        '''



        """Generate a random sample of RoIs comprising foreground and background
        examples.生成包含前景和背景示例的RoI的随机样本
        """
        # overlaps: (rois x gt_boxes)

        overlaps = bbox_overlaps_batch(all_rois, gt_boxes)#shape=(b,2020,20) 也就是2020个rois 与20个gt的iou值

        max_overlaps, gt_assignment = torch.max(overlaps, 2)
        #max_overlaps.shape=(b,2020) 记录了每个rois 与所有gt最大的iou值
        #gt_assignment.shape=(b,2020) 记录了每个rois 与所有gt最大的iou的gt的索引
        '''到此位置计算出了rpn提供的2020个（应该是 2000个proposal），，每个proposal里面与每个gt的最大iou， 以及最大iou的gt的索引'''

        batch_size = overlaps.size(0)#b
        num_proposal = overlaps.size(1)#2020
        num_boxes_per_img = overlaps.size(2)#20

        offset = torch.arange(0, batch_size)*gt_boxes.size(1)#gt_boxes.size(1)=20
        #offset=[0,20*1,20*2,20*3,....20*(b-1)]
        offset = offset.view(-1, 1).type_as(gt_assignment) + gt_assignment
        #offset.shape=(b,2020) ,里面存的是每个proposal 匹配的iou最大的gt的索引（只不过隔一张图片加上了20）

        '''下面的labels这句代码有错，根据girhub issue里面的回答更改'''
        #labels = gt_boxes[:,:,4].contiguous().view(-1).index((offset.view(-1),)).view(batch_size, -1)
        labels = gt_boxes[:, :, 4].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)
        #gt_boxes[:,:,4].shape=(b,20) 即将每一个gt的类别信息取出， 里面每一个数字代表该gt的类别
        #gt_boxes[:,:,4].contiguous().view(-1) .shape=(b*20)   offset.view(-1).shape=(b*2020)
        #gt_boxes[:, :, 4].contiguous().view(-1)[(offset.view(-1),)] .shape=(b*2020) 即得到每一个proposal匹配的gt的类别，也就是proposal的类别target
        #gt_boxes[:, :, 4].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1) .shape=(b,2020)即得到每一个proposal匹配的gt的类别，也就是proposal的类别target
        
        labels_batch = labels.new(batch_size, rois_per_image).zero_()#shape=(b, 128)全零  rois_per_image=128 是一张图片上取多少roi给后面的rcnn
        rois_batch  = all_rois.new(batch_size, rois_per_image, 5).zero_()#shape=(b, 128, 5) 全零
        gt_rois_batch = all_rois.new(batch_size, rois_per_image, 5).zero_()#shape=(b,128，5) 全零
        # Guard against the case when an image has fewer than max_fg_rois_per_image
        # foreground RoIs
        for i in range(batch_size):#i= 0,1,2,3,4,....(b-1)

            # max_overlaps.shape=(b,2020) 记录了每个rois 与所有gt最大的iou值
            #max_overlaps[i] = (2020) 记录了该张图片上每个rois 与所有gt最大的iou值
            fg_inds = torch.nonzero(max_overlaps[i] >= cfg.TRAIN.FG_THRESH).view(-1)   #FG_THRESH=0.5
            #fg_inds.shape=(k)  =[0,2,5,...]（表示该张图片上第0,2,5.。个proposal与gt的iou大于0.5） 将每张图片上proposal（2020个）找出其iou>0.5的proposal的索引
            fg_num_rois = fg_inds.numel()
            #fg_num_rois = k 也就是fg_inds里面元素的个数，也就是该张图片上符合iou>0.5 的peoposal的个数
            '''到此位置得到了每张图片 2020个proposal属于正样本的proposal索引（maxIoU >0.5）'''

            # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)  BG_THRESH_HI=0.5  BG_THRESH_LO=0.1
            bg_inds = torch.nonzero((max_overlaps[i] < cfg.TRAIN.BG_THRESH_HI) &
                                    (max_overlaps[i] >= cfg.TRAIN.BG_THRESH_LO)).view(-1)
            #bg_inds.shape=(j) =[0,2,5,...]（表示该张图片上第0,2,5.。个proposal与gt的iou<0.5 且>0.1） 将每张图片上proposal（2020个）找出其iou<0.5且>0.1的proposal的索引
            bg_num_rois = bg_inds.numel()
            # bg_num_rois = j 也就是bg_inds里面元素的个数，也就是该张图片上符合 的peoposal的个数
            '''到此位置得到了每张图片2020个proposal属于负样本的proposal索引（0.1<maxIoU <0.5）'''


            if fg_num_rois > 0 and bg_num_rois > 0:
                # sampling fg
                fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)#fg_rois_per_image=128.0.25=32   fg_num_rois=k
                #一般一张图片里面目标个数不会有32个 所以我们这里建设32较小  即32<k
                #fg_rois_per_this_image=k

                # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
                # See https://github.com/pytorch/pytorch/issues/1868 for more details.
                # use numpy instead.
                #rand_num = torch.randperm(fg_num_rois).long().cuda()
                rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).type_as(gt_boxes).long()#rand_num=[]  即产生tensor，里面是0-(k-1)这k个数打乱的结果
                # fg_inds.shape=(k)  =[0,2,5,...]（表示该张图片上第0,2,5.。个proposal与gt的iou大于0.5）
                fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]
                #fg_inds.shape = (k)=[0,2,5,...] （表示该张图片上第0,2,5.。个proposal与gt的iou大于0.5） 从原来的fg_inds里面随机抽出k个（前提k比32小，否则就是从fg_inds里面随机抽出32个）

                # sampling bg
                bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image#=128-k

                # Seems torch.rand has a bug, it will generate very large number and make an error.
                # We use numpy rand instead.
                #rand_num = (torch.rand(bg_rois_per_this_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                #rand_num = 0---(bg_num_rois-1)打乱顺序
                bg_inds = bg_inds[rand_num]
                #bg_inds.shape=(128-k) =[0,2,5,...]（表示该张图片上第0,2,5.。个proposal与gt的iou<0.5 且>0.1） 从原来的bfinds里面抽出 128-k个背景样本的索引

            elif fg_num_rois > 0 and bg_num_rois == 0:
                # sampling fg
                #rand_num = torch.floor(torch.rand(rois_per_image) * fg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * fg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
                fg_inds = fg_inds[rand_num]
                fg_rois_per_this_image = rois_per_image
                bg_rois_per_this_image = 0
            elif bg_num_rois > 0 and fg_num_rois == 0:
                # sampling bg
                #rand_num = torch.floor(torch.rand(rois_per_image) * bg_num_rois).long().cuda()
                rand_num = np.floor(np.random.rand(rois_per_image) * bg_num_rois)
                rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()

                bg_inds = bg_inds[rand_num]
                bg_rois_per_this_image = rois_per_image
                fg_rois_per_this_image = 0
            else:
                raise ValueError("bg_num_rois = 0 and fg_num_rois = 0, this should not happen!")

            # The indices that we're selecting (both fg and bg)
            keep_inds = torch.cat([fg_inds, bg_inds], 0)
            #keep_inds.shape=(128) 将fg_inds bg_inds 合并起来，， fg_inds共k个  bg_inds共128-k个  两者加起来共128个（也就是从共2020个proposal里面选出128个，这128
            # 个里面包含k个前景样本，128-k个背景样本）

            # Select sampled values from various arrays:
            #labels_batch.shape=(b, 128)全零  rois_per_image=128 是一张图片上取多少roi给后面的rcnn
            #labels.shape=(b,2020)即得到每一个proposal匹配的gt的类别，也就是proposal的类别target
            labels_batch[i].copy_(labels[i][keep_inds])


            # Clamp labels for the background RoIs to 0
            if fg_rois_per_this_image < rois_per_image:
                labels_batch[i][fg_rois_per_this_image:] = 0

            #all_rois: shape=(b,2020,5) 将gt加入到候选rois里面  原来roi=（b,2000,5） gt=(b,20,5)
            #rois_batch.shape=(b, 128, 5) 全零
            rois_batch[i] = all_rois[i][keep_inds]
            rois_batch[i,:,0] = i  #5：第一个数代表是batch里面的第几张图片

            #gt_boxes: shape =（b, 20, 5）应该不是每个图片都有20个目标，可能20取的是一个比较大的值，并不是20里面都有gt的数据（固定的（b, 20, 5）维度）
            #gt_assignment.shape=(b,2020) 记录了每个rois 与所有gt最大的iou的gt的索引
            #gt_rois_batch.shape=(b,128，5) 全零
            gt_rois_batch[i] = gt_boxes[i][gt_assignment[i][keep_inds]]#将选出来的128个正负样本各自最大iou的gt信息存储到gt_rois_batch里面

        bbox_target_data = self._compute_targets_pytorch(
                rois_batch[:,:,1:5], gt_rois_batch[:,:,:4])
        #bbox_target_data.shape=(b,128,4)得到了128个rois的回归值

        bbox_targets, bbox_inside_weights = \
                self._get_bbox_regression_labels_pytorch(bbox_target_data, labels_batch, num_classes)
        #bbox_targets.shape=(b,128,4)
        #bbox_inside_weights.shape=(b,128,4)

        # bbox_targets.shape=(b,128,4) 从bbox_target_data（（b,128,4）存了正样本和负样本的回归target（由于前面计算回归target是128个proposal都计算，所以这里面负样本也有回归target，所以这里要把负样本的回归target清零））
        #                               将正样本的回归target取出放到bbox_targets里（bbox_targets原来为0）
        # bbox_inside_weights.shape=(b,128,4) 4=[1,1,1,1]或者［0,0,0,0］ 正样本的权重是[1,1,1,1] 负样本的权重是[0,0,0,0]

        return labels_batch, rois_batch, bbox_targets, bbox_inside_weights
    '''
    labels_batch.shape=(b, 128) 记录了每张图片上128个样本（正、负）的类别target（就是与其iou最大的gt的类别，精确到哪一类，而不是前背景类）
    rois_batch.shape=(b, 128, 5) 记录了这128个roi的box（是从rpn预测出来的2020个proposal里面选出的128个box） 5:第一个数i代表该图片是该batch中的第i张图片
    bbox_targets.shape=(b,128,4) 从bbox_target_data（（b,128,4）存了正样本和负样本的回归target（负样本的回归目标在 self._get_bbox_regression_labels_pytorch函数中被设置成0了）
     
    bbox_inside_weights.shape=(b,128,4) 4=[1,1,1,1]或者［0,0,0,0］ 正样本的权重是[1,1,1,1] 负样本的权重是[0,0,0,0]
    
    '''
