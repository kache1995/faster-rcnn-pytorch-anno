import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
from lib.model.utils.config import cfg
from lib.model.rpn.rpn import _RPN
from lib.model.roi_pooling.modules.roi_pool import _RoIPooling
from lib.model.roi_crop.modules.roi_crop import _RoICrop
from lib.model.roi_align.modules.roi_align import RoIAlignAvg
from lib.model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from lib.model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta

class _fasterRCNN(nn.Module):
    """ faster RCNN """
    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE# = 7*2 =14
        '''
        # Size of the pooled region after RoI pooling
        __C.POOLING_SIZE = 7  roi pooling 之后得到的特征的尺寸
        CROP_RESIZE_WITH_MAX_POOL = True
        
        
        '''


        self.RCNN_roi_crop = _RoICrop()

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        '''

        :param im_data: shape=(b,3,W,H)
        :param im_info:shape=(b,3)  3=[W,H,2.2901]最后一个2.2901含义还不太清楚
        :param gt_boxes:shape=（b,20,5）应该不是每个图片都有20个目标，可能20取的是一个比较大的值，并不是20里面都有gt的数据（固定的（b,20,5）维度） 前n为没张图片上的gt，后面20-n全为0
        :param num_boxes:shape=(b) b=[k,j,...]应该是代表这个batch里面第一张图片上有k个gt，第二张图片上有j个gt
        :return:
        '''
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data

        # feed image data to base model to obtain base feature map
        base_feat = self.RCNN_base(im_data)
        #base_feat.shape=(b,1024,w,h)  w和h是原图的16分之一
        '''到此得到了前面特征提取网络的结果'''

        # feed base feature map tp RPN to obtain rois
        rois, rpn_loss_cls, rpn_loss_bbox = self.RCNN_rpn(base_feat, im_info, gt_boxes, num_boxes)
        '''
        #rois=output.shape=(b,2000,5)  5:[当前box在那个图片上（0-batchsize）,x1,y1,x2,y2]
        #rpn_loss_cls：rpn网络的分类loss   只计算了正样本和负样本的分类loss（根据匹配的标签，先得到正样本和负样本的索引，然后从预测的分数中取出
        #        正样本和负样本的分类分数，然后在从匹配的标签中取出正样本和负样本的标签（1,0），然后两者之间算交叉熵loss）
        #rpn_loss_box=(2.36)  是一个值，代表一个batch里面各个图片上的回归损失求的平均

        
        到此为止，rpn的功能就结束了，产生2000个proposal ，以及算所有anchor的回归损失和分类损失
        '''



        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes, num_boxes)
            #rois.shape=(b,2000,5) 5:[当前box在那个图片上（0-batchsize）,x1,y1,x2,y2]
            #gt_boxes: shape =（b, 20, 5）应该不是每个图片都有20个目标，可能20取的是一个比较大的值，并不是20里面都有gt的数据（固定的（b, 20, 5）维度）
            #num_boxes:shape=(b) b=[k,j,...]应该是代表这个batch里面第一张图片上有k个gt，第二张图片上有j个gt





            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            '''
            rois.shape=(b, 128, 5) 记录了这128个roi的box（是从rpn预测出来的2020个proposal里面选出的128个box） 5:第一个数i代表该图片是该batch中的第i张图片
            rois_label=labels.shape=(b, 128) 记录了每张图片上128个样本（正、负）的类别target（就是与其iou最大的gt的类别，精确到哪一类，而不是前背景类）
            rois_target=bbox_targets.shape=(b,128,4) 存了正样本和负样本的回归target（负样本的回归目标在 self._get_bbox_regression_labels_pytorch函数中被设置成0了）
            rois_inside_ws=box_inside_weights.shape=(b,128,4) 4=[1,1,1,1]或者［0,0,0,0］ 正样本的权重是[1,1,1,1] 负样本的权重是[0,0,0,0]
            rois_outside_ws=bbox_outside_weights.shape=(b,128,4) 4=[1,1,1,1]或者［0,0,0,0］ 正样本的权重是[1,1,1,1] 负样本的权重是[0,0,0,0]
            
            '''

            rois_label = Variable(rois_label.view(-1).long()) #shape=(b*128) 保存了送入rcnn网络的每张图片128个roi的类别标签
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))#shape=(b*128,4)保存了送入rcnn网络的每张图片128个roi的回归target
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))#shape=(b*128,4) 保存了每张图片128个roi的内权重（见损失公式）正样本的权重[1,1,1,1] f负样本[0,0,0,0]
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))#shape=[b*128,4]保存了每张图片128个roi的外权重（见损失公式）正样本的权重[1,1,1,1] f负样本[0,0,0,0]
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0

        rois = Variable(rois)
        #train : rois.shape=(b, 128, 5) 记录了这128个roi的box（是从rpn预测出来的2020个proposal里面选出的128个box） 5:第一个数i代表该图片是该batch中的第i张图片
        #test : rois.shape=(b,2000,5)  5:[当前box在那个图片上（0-batchsize）,x1,y1,x2,y2]


        # do roi pooling based on predicted rois
        '''
        这里从特征图上获得roi box的特征使用的方式是：align（虽然cfg.POOLING_MODE = ’crop‘，但是前面应该是哪里把它改成了align ）
        '''
        if cfg.POOLING_MODE == 'crop':
            # pdb.set_trace()
            # pooled_feat_anchor = _crop_pool_layer(base_feat, rois.view(-1, 5))
            grid_xy = _affine_grid_gen(rois.view(-1, 5), base_feat.size()[2:], self.grid_size)
            # base_feat.shape=(b,1024 ,w,h)  w和h是原图的16分之一    base_feat.size()[2:] = [w,h]


            grid_yx = torch.stack([grid_xy.data[:,:,:,1], grid_xy.data[:,:,:,0]], 3).contiguous()
            pooled_feat = self.RCNN_roi_crop(base_feat, Variable(grid_yx).detach())
            if cfg.CROP_RESIZE_WITH_MAX_POOL:
                pooled_feat = F.max_pool2d(pooled_feat, 2, 2)
        elif cfg.POOLING_MODE == 'align':#运行align
            #base_feat.shape=(b,1024 ,w,h)  w和h是原图的16分之一
            #train : rois.view(-1,5) .shape=(b*128,5)   test : rois.view(-1,5) .shape=(b*2000,5)
            pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))
            #train : pooled_feat.shape=(b*128,1024,7,7)  test : pooled_feat.shape=(b*2000,1024,7,7)
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat = self.RCNN_roi_pool(base_feat, rois.view(-1,5))

        # feed pooled features to top model
        '''这里只看训练时候：train : pooled_feat.shape=(b*128,1024,7,7)'''
        pooled_feat = self._head_to_tail(pooled_feat)#这里父类（class faster rcnn）调用了子类（class VGG）的head函数
        ##pooled_feat.shape=(b*128,2048)

        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)#bbox_pred.shape=(b*128,21*4)
        #看了训练的时候的输出：self.class_agnostic=false 那么就是需要针对每个类别预测回归值
        if self.training and not self.class_agnostic: #self.class_agnostic=false
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)#bbox_pred_view.shape=(b*128,21,4)
            #rois_labels.shape=(b*128) 保存了送入rcnn网络的每张图片128个roi的类别标签
            bbox_pred_select = torch.gather(bbox_pred_view, 1, rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            #bbox_pred_select.shape=(b*128,1,4) 从针对每一类预测的回归值（21,4）中取出该roi匹配的gt的类别（也就是该roi的类别target）的回归值
            bbox_pred = bbox_pred_select.squeeze(1)
            #bbox_pred.shape=(b*128,4)

        '''
        train: bbox_pred.shape=(b*128,4)
        test:  bbox_pred.shape=(b*2000, 21*4)
        '''

        '''到此位置得到的 128个roi经过rcnn预测的回归值（针对target类别的回归值，也就是说，如果这个roi的类别target是汽车，那么就从21*4中取出关于汽车这个类的预测回归值）'''


        # compute object classification probability

        ##pooled_feat.shape=(b*128,2048)
        cls_score = self.RCNN_cls_score(pooled_feat)#shape=(b*128,21) 预测属于每个类的分数
        cls_prob = F.softmax(cls_score, 1)#shape=(b*128,21) 得到属于每个类的概率（预测）

        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)

            # bounding box regression L1 loss
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            ##bbox_pred.shape=(b*128,4) 预测的每个roi的的回归值（针对他们的类别target的回归值）
            ##rois_target.shape=(b*128,4)保存了送入rcnn网络的每张图片128个roi的回归target
            #rois_inside_ws.shape=(b*128,4) 保存了每张图片128个roi的内权重（见损失公式）正样本的权重[1,1,1,1] f负样本[0,0,0,0]
            #rois_outside_ws.shape=[b*128,4]保存了每张图片128个roi的外权重（见损失公式）正样本的权重[1,1,1,1] f负样本[0,0,0,0]


        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)#shape(b,128,21)得到属于每个类的概率（预测） 如果是test：shape=(b,2000,21)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)#shape=(b,128,4)预测的每个roi的的回归值（针对他们的类别target的回归值） 如果是test：shape=(b,2000,4)
        #test： bbox_pred.shaep=(b,2000,21*4)

        return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label

    # train : rois.shape=(b, 128, 5) 记录了这128个roi的box（是从rpn预测出来的2020个proposal里面选出的128个box） 5:第一个数i代表该图片是该batch中的第i张图片
    # test : rois.shape=(b,2000,5)  5:[当前box在那个图片上（0-batchsize）,x1,y1,x2,y2]
    # cls_prob.shape(b,2000,21)得到属于每个类的概率（预测） 如果是test：shape=(b,2000,21)
    #bbox_pred.shape=(b,128,4)预测的每个roi的的回归值（针对他们的类别target的回归值） 如果是test：shape=(b,2000,21*4)
    #rpn_class_cls  rpn网络的分类loss（9*w*h个anchor里面正样本和负样本（共k个）的分类loss（不含ignore的分类loss））
    # rpn_loss_box=(2.36)  是一个值，代表一个batch里面各个图片上的回归损失求的平均
    #RCNN_loss_cls 是128个roi的分类loss
    #RCNN_loss_bbox  是128个roi的回归loss
    ##rois_labels.shape=(b*128) 保存了送入rcnn网络的每张图片128个roi的类别标签


    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean) # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
