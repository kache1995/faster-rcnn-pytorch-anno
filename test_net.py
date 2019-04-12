# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from lib.roi_data_layer.roidb import combined_roidb
from lib.roi_data_layer.roibatchLoader import roibatchLoader
from lib.model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from lib.model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False
  imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdbval_name, False)
  imdb.competition_mode(on=True)

  print('{:d} roidb entries'.format(len(roidb)))

  input_dir = args.load_dir + "/" + args.net + "/" + args.dataset
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))


  '''args.class_agnostic = False'''
  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100
  '''这个参数很重要，决定了最后每....'''

  vis = args.vis#是否要可视化

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  save_name = 'faster_rcnn_10'
  num_images = len(imdb.image_index)
  all_boxes = [[[] for _ in xrange(num_images)]
               for _ in xrange(imdb.num_classes)]

  output_dir = get_output_dir(imdb, save_name)
  dataset = roibatchLoader(roidb, ratio_list, ratio_index, 1, \
                        imdb.num_classes, training=False, normalize = False)
  '''注意看这里batchsize为1'''
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  data_iter = iter(dataloader)

  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  fasterRCNN.eval()
  empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
  for i in range(num_images):
      #todo:注意这里，现在是一张一张图片送入网络，而不是一个batch，，所以batchsize = 1

      data = next(data_iter)
      im_data.data.resize_(data[0].size()).copy_(data[0])
      im_info.data.resize_(data[1].size()).copy_(data[1])
      gt_boxes.data.resize_(data[2].size()).copy_(data[2])
      num_boxes.data.resize_(data[3].size()).copy_(data[3])

      '''
        :param im_data: shape=(1,3,W,H)
        :param im_info:shape=(1,3)  3=[W,H,2.2901]最后一个2.2901含义还不太清楚 --》pred_boxes /= data[1][0][2].item() 最后得到的box除以了这个2.29数
        :param gt_boxes:shape=（1,20,5）应该不是每个图片都有20个目标，可能20取的是一个比较大的值，并不是20里面都有gt的数据（固定的（b,20,5）维度）
        #:param num_boxes:shape=(b) b=[k,j,...]应该是代表这个batch里面第一张图片上有k个gt，第二张图片上有j个gt
      
      '''

      det_tic = time.time()
      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
      #rois.shape=(1,300,5)  5:[当前box在那个图片上（0-batchsize）,x1,y1,x2,y2] 也就是rpn得到的300roi，用来送入rcnn
      #cls_prob.shape(1,300,21)得到属于每个类的概率（预测） 如果是test：shape=(b,2000,21)
      #bbox_pred.shape=(1,300,4*21)预测的每个roi的的回归值（针对他们的类别target的回归值）

      # rpn_class_cls  rpn网络的分类loss（9*w*h个anchor里面正样本和负样本（共k个）的分类loss（不含ignore的分类loss））
      # rpn_loss_box=(2.36)  是一个值，代表一个batch里面各个图片上的回归损失求的平均
      # RCNN_loss_cls 是128个roi的分类loss
      # RCNN_loss_bbox  是128个roi的回归loss
      ##rois_labels = None
      #todo：注意，上面注释里面的b（batchsize）现在为1

      #测试时，上面所有的loss都是0


      scores = cls_prob.data#shape(1,300,21)得到属于每个类的概率（预测）
      boxes = rois.data[:, :, 1:5]#shape=(1,300,4)

      if cfg.TEST.BBOX_REG:#true
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data#shape=(1,300,21*4)预测的每个roi的的回归值
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:#True
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                '''args.class_agnostic=false  所以为跳到else执行，但是在fasterrcnn.py里面class_agnostic 默认false'''
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                #将预测的回归值做了一些改变（均值啥的）
                #box_deltas.shape=(1*2000*21,4)
                box_deltas = box_deltas.view(1, -1, 4)
                # box_deltas.shape=(1,2000*21,4)
            else:
                box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                           + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                #box_deltas.shape=(1*300*21,4)
                box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))
                # box_deltas.shape=(1,300,4*21)

          #boxes.shape=(1,300,4) rcnn的输入的框
          #box_deltas.shape=(1,300,4*21) rcnn预测的框的回归值
          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          #todo：查看测试时候（1,300,4）与（1,300,4*21）是怎么加起来的得到（1,300,4*21）
          #pred_boxes.shape=(1,300,4*21) 这是根据网络最后预测的回归值调整了送入rcnn的300个roi的位置，得到最终的box位置(test时候预测回归是针对每一类的回归4*21，所以
          # 调整位置的时候得到的结果也是针对每一类的位置)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)#修剪一下box（超出图片边界的做一下修剪）
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))
      #todo：为什么要将最后的位置除以一个数（难道是因为开始的时候图片进行了尺度变化？？？）
      pred_boxes /= data[1][0][2].item()

      scores = scores.squeeze()##shape(300,21)得到属于每个类的概率（预测） 因为现在batch为1，所有可以去掉batch维度
      pred_boxes = pred_boxes.squeeze()#shape=(300,4*21) 这是根据网络最后预测的回归值调整了送入rcnn的300个roi的位置，得到最终的box位置
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      if vis:
          im = cv2.imread(imdb.image_path_at(i))
          im2show = np.copy(im)
      for j in xrange(1, imdb.num_classes):#1,2,3，...20 针对每一个类别
          inds = torch.nonzero(scores[:,j]>thresh).view(-1)
          #scores[:, j].shape=(300) 代表2000个框是该类别的概率（该类的分数大于阈值则认为是属于该类别）
          #scores[:,j]>thresh.shape=(300) 里面值为0/1  0：代表这个box不是这个类别  1：代表该box是这个类别
          #torch.nonzero(scores[:, j] > thresh).shape=(k,1)
          #inds.shape=(k)  [0,4,6,19,...] 代表300个box中，第0,4,6,19，...个box属于该类别的分数大于阈值  k<=300

          # if there is det
          if inds.numel() > 0:#inds.numel()=k 计算inds里面元素的个数
            cls_scores = scores[:,j][inds]#shape=(k) 将属于该类的box的分数取出来 (假设k个box属于该类)
            _, order = torch.sort(cls_scores, 0, True)#将cls_socres(属于该类的框的概率)按分数从大到小排序  order.shape=(k)=[2,5,1,...] 代表2000个box中第2个box的概率最大,...
            if args.class_agnostic:#false
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]#pred_boxes[inds].shape=(k,4*21)将属于该类的box的位置拿出来
                # pred_boxes[inds][:, j * 4:(j + 1) * 4] .shape=(k,4) 在box的21个位置（针对每一类的回归位置）中将该类的位置取出
                #cls_boxes.shape=(k,4)
            '''300个roi，每个roi有21个分数， 有21个回归位置，前面通过分数大于阈值判断属于该类的roi k个，然后将这个k个roi对应该类的位置取出，得到cls_boxes.shape=(k,4) '''
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            #cls_scores.unsqueeze(1).shape=(k,1)  cls_boxes.shape=(k,4)
            #cls_dets .shape=(k,5) 将属于该类的box的位置和分数结合在一起  5：[x1,y1,x2,y2,score]



            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            #cls_dets.shape=(k,5) 按照分数高低排个序（从高到低）

            keep = nms(cls_dets, cfg.TEST.NMS)#TEST.NMS=0.3
            #keep.shape=(h,1),每个值代表要保留的bbox(k个box中要保留的box的索引，共h个, h<=k)
            cls_dets = cls_dets[keep.view(-1).long()]#shape=(h,5) k个box经过nms还剩下的h个box
            if vis:
              im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
            all_boxes[j][i] = cls_dets.cpu().numpy()#将数组存放在列表里面 all_boxes=[[[],[],[],[]...共n个]，[[],array[(h,5)],[],[]...共n个]，[[],[],[],[]...共n个]，。。共21个]
            #i代表第i张图片(假设共n张测试图片)，j代表第j类（j=1,2,,20共20类）
            # 第j类列表的第i个列表=数组（shape=（h，5））,如上一行  所以all_boxes[j][i]是一个numpy数组.shape=（h，5）
            # （可以这样理解：最终all_boxes.shape=(21,n,h,5)，allboxes是列表，用shape来说明不好。）
            #all_boxes是一个列表,第一维是21是固定的，第二维n代表有多少测试图片，一般不固定， 第三维h代表该张图片上有h个box输入该类，不固定 ，第四维5固定
            #由于allboxes是一个列表，除了21和5固定，其他n，h都不的固定
          else:
            all_boxes[j][i] = empty_array

      '''到这里，第i张图片上针对每个类（20个类）的box得到了， 这时all_boxes[:,i,:,:].shape=(20,h,5)  对于每个类，h可能有不同的值(h1,h2,h3,..h20)'''

      # Limit to max_per_image detections *over all classes*
      if max_per_image > 0:#max_per_image=100
          image_scores = np.hstack([all_boxes[j][i][:, -1]
                                    for j in xrange(1, imdb.num_classes)])
          #image_scores.shape=(h1+h2+...+h20) 将该张图片上所有类的box的分数拿出来
          if len(image_scores) > max_per_image:#len(image_scores)就代表该张图片检测出多少box，如果box个数大于100，那么就按分数从高到低取前100个
              image_thresh = np.sort(image_scores)[-max_per_image]
              for j in xrange(1, imdb.num_classes):
                  keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                  all_boxes[j][i] = all_boxes[j][i][keep, :]

      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
          .format(i + 1, num_images, detect_time, nms_time))
      sys.stdout.flush()

      if vis:
          cv2.imwrite('result.png', im2show)
          pdb.set_trace()
          #cv2.imshow('test', im2show)
          #cv2.waitKey(0)
  #最终all_boxes.shape = (21, n, h, 5)  这里all_boxes是一个列表，里面的h有不同的值，最多有20*n（不算背景类）个值，代表每张图片每个类检测出来的box个数
  '''
      all_box = [[[],[]],[[],[]],[[],[]]]
    all_box[0][0] = np.array([1,1,1,1,0.1])
    all_box[0][1] = np.array([[2,2,2,2,0.2],[2,2,2,2,0.02]])
    all_box[1][1] = np.array([3,3,3,3,0.3])
    all_box[2][0] = np.array([[4,4,4,4,0.4],[5,5,5,5,0.5]])
    print(all_box)
  
  [[array([1. , 1. , 1. , 1. , 0.1]), array([[2.  , 2.  , 2.  , 2.  , 0.2 ],
       [2.  , 2.  , 2.  , 2.  , 0.02]])], [[], array([3. , 3. , 3. , 3. , 0.3])], [array([[4. , 4. , 4. , 4. , 0.4],
       [5. , 5. , 5. , 5. , 0.5]]), []]]
  
  '''



  with open(det_file, 'wb') as f:
      pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)#将所有测试图片的结果保存到文件

  print('Evaluating detections')
  imdb.evaluate_detections(all_boxes, output_dir)

  end = time.time()
  print("test time: %0.4fs" % (end - start))
