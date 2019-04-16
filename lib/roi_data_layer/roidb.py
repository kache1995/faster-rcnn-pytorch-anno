"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import lib.datasets
import numpy as np
from lib.model.utils.config import cfg
from lib.datasets.factory import get_imdb
import PIL
import pdb

def prepare_roidb(imdb):
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """

  roidb = imdb.roidb
  # 猜的：self.roidb= [{}，{}，。。。{}] 共10022个
  '''
      {'boxes': array([[ 47, 239, 194, 370],
     [  7,  11, 351, 497]], dtype=uint16), 'gt_classes': array([12, 15]), 'gt_ishard': array([0, 0]), 'gt_overlaps': <2x21 sparse matrix of type '<class 'numpy.float32'>'
  with 2 stored elements in Compressed Sparse Row format>, 'flipped': False, 'seg_areas': array([ 19536., 168015.], dtype=float32)}

  box.shape=(n,4) n代表这张图片上有几个目标
  gt_classes.shape=(n)   例如：array([12, 15]) 代表第一个目标是第12类（猫），第二个目标是第15类（人）
  gt_overlaps.   overlaps.shape=(n,21) 值为0或1  默认0， 如果[2,20]=1代表该张图片上第2个目标是第20类的分割
  flipped= False(前5011个为false ，后5011个为True)
  seg_areas.shape=(n) 例如：array([ 19536., 168015.] 代表第一个目标box的面积是19536， 第二个目标box的面积是168015

  '''



  if not (imdb.name.startswith('coco')):
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
         for i in range(imdb.num_images)]#imdb.num_images=5011*2 做了数据扩增之后
      #imdb.num_images=5011 imdb.image_path_at(i)获取第i张图片的路径
      #size=[（353,400）,（200,300），...] 存储了10022张图片的尺寸
         
  for i in range(len(imdb.image_index)):
    roidb[i]['img_id'] = imdb.image_id_at(i)
    roidb[i]['image'] = imdb.image_path_at(i)
    if not (imdb.name.startswith('coco')):
      roidb[i]['width'] = sizes[i][0]
      roidb[i]['height'] = sizes[i][1]
    # need gt_overlaps as a dense array for argmax
    gt_overlaps = roidb[i]['gt_overlaps'].toarray()
    # max overlap with gt over classes (columns)
    max_overlaps = gt_overlaps.max(axis=1)#[1,1,1,..] 如果这张图片有n个obj ，那么就有n个1
    # gt class that had the max overlap
    max_classes = gt_overlaps.argmax(axis=1)#[12,15]  表示第一个目标是第12类，第二个目标是滴15类
    roidb[i]['max_classes'] = max_classes
    roidb[i]['max_overlaps'] = max_overlaps
    # sanity checks
    # max overlap of 0 => class should be zero (background)
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    # max overlap > 0 => class should not be zero (must be a fg class)
    nonzero_inds = np.where(max_overlaps > 0)[0]
    assert all(max_classes[nonzero_inds] != 0)

  # 猜的：self.roidb= [{}，{}，。。。{}] 共10022个
  '''
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
  '''



def rank_roidb_ratio(roidb):
    # rank roidb based on the ratio between width and height.
    ratio_large = 2 # largest ratio to preserve.
    ratio_small = 0.5 # smallest ratio to preserve.    
    
    ratio_list = []
    for i in range(len(roidb)):
      width = roidb[i]['width']
      height = roidb[i]['height']
      ratio = width / float(height)

      if ratio > ratio_large:
        roidb[i]['need_crop'] = 1
        ratio = ratio_large
      elif ratio < ratio_small:
        roidb[i]['need_crop'] = 1
        ratio = ratio_small        
      else:
        roidb[i]['need_crop'] = 0

      ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)#[4 1 5 0 2 3,...] 代表第四个最小，其次第一个最小，。。（从小到大的索引）
    return ratio_list[ratio_index], ratio_index
#ratio_list[ratio_index]将ratio_list里面的比率按照从小到大排序（根据ratio_index排好的）
#ratio_index[4 1 5 0 2 3,...] 代表第四个最小，其次第一个最小，。。（从小到大的索引）

def filter_roidb(roidb):
    # filter the image without bounding box.
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
      if len(roidb[i]['boxes']) == 0:
        del roidb[i]
        i -= 1
      i += 1

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb

def combined_roidb(imdb_names, training=True):
  '''

  :param imdb_names: #imdb_name=voc_2007_trainval
  :param training: True 先看训练的时候
  :return:
  '''
  """
  Combine multiple roidbs
  """

  def get_training_roidb(imdb):
    '''

    :param imdb:  imdb = pascal_voc(trainval, 2007)  pascal_voc是imdb(类)的子类  imdb（对象）是pascal（类）的对象
    :return:
    '''
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:#True 在训练期间使用水平翻转图像
      print('Appending horizontally-flipped training examples...')
      imdb.append_flipped_images()
      print('done')

    print('Preparing training data...')

    prepare_roidb(imdb)
    #ratio_index = rank_roidb_ratio(imdb)
    print('done')

    return imdb.roidb
  # 猜的：self.roidb= [{}，{}，。。。{}] 共10022个
  '''
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
  '''
  
  def get_roidb(imdb_name):
    #imdb_name=voc_2007_trainval
    imdb = get_imdb(imdb_name)#imdb = pascal_voc(trainval, 2007)  pascal_voc是imdb的子类
    print('Loaded dataset `{:s}` for training'.format(imdb.name))#imdb.name= voc_2007_trainval
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)  #TRAIN.PROPOSAL_METHOD='gt'
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))#'gt'
    roidb = get_training_roidb(imdb)
    return roidb
  # 猜的：self.roidb= [{}，{}，。。。{}] 共10022个
  '''
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
  '''

  roidbs = [get_roidb(s) for s in imdb_names.split('+')]  #s = [voc_2007_trainval]
  roidb = roidbs[0]
  # 猜的：self.roidb= [{}，{}，。。。{}] 共10022个
  '''
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
  '''

  if len(roidbs) > 1:#roidbs=1
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)#这里imdb好像和roidb是一样的

  if training:
    roidb = filter_roidb(roidb)

  ratio_list, ratio_index = rank_roidb_ratio(roidb)
  # ratio_list[ratio_index]将ratio_list里面的比率按照从小到大排序（根据ratio_index排好的）
  # ratio_index[4 1 5 0 2 3,...] 代表第四个最小，其次第一个最小，。。（从小到大的索引）
  '''
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
  ----------------------
  imdb：=[{},{},{},....{}]
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


  return imdb, roidb, ratio_list, ratio_index
