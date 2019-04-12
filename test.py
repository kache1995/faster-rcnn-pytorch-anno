import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import pickle
import PIL
from PIL import Image
import os
import xml
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import scipy

def fun1():
    a = torch.Tensor(5,2,3,3)
    b = F.softmax(a, 1)
    print(b.shape)

def fun2():
    scores = torch.Tensor(5,9,32,32)
    feat_width = 32
    feat_height = 32
    _feat_stride = 16
    shift_x = np.arange(0, feat_width) * _feat_stride  # [0,16,16*2,16*3,...16*h]
    shift_y = np.arange(0, feat_height) * _feat_stride  # [0,16,16*2,16*3,...16*w]
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                         shift_x.ravel(), shift_y.ravel())).transpose())
    shifts = shifts.contiguous().type_as(scores).float()


    print(shifts.shape)
    print(shifts[1000,:])

def fun3():
    batchsize=3
    scores_keep = torch.Tensor([[1,2,3,0],[1,8,6,7],[6,1,5,3]])
    _, order = torch.sort(scores_keep, 1, True)
    print(scores_keep)
    print(order)

    output = scores_keep.new(3,2,5).zero_()
    print(output)

def fun4():
    a = torch.Tensor(20,5)
    b = a[:,0]
    print(b.shape)

def fun5():
    output = torch.zeros(3,4,5)
    for i in range(3):
        output[i,:,0]=i
        print(output[i,:,0].shape)
    print(output)

def fun6():
    rpn_cls_score = torch.Tensor(2,18,40,32)
    feat_width = 32
    feat_height = 40
    feat_stride = 16
    shift_x = np.arange(0, feat_width) * feat_stride
    shift_y = np.arange(0, feat_height) * feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    print(shift_x.shape)
    print(shift_y.shape)
    shifts = torch.from_numpy(np.vstack((shift_x.ravel(), shift_y.ravel(),
                                         shift_x.ravel(), shift_y.ravel())).transpose())
    shifts = shifts.contiguous().type_as(rpn_cls_score).float()
    print(shifts.shape)

def fun7():
    keep = torch.Tensor([0,3,2,1,0])
    inds_inside = torch.nonzero(keep).view(-1)
    print(inds_inside)

def fun8():
    overlaps = torch.Tensor([[[1,2,3,4],[5,3,1,5]],[[4,3,2,1],[3,6,1,5]]])#(2,2,4)
    print(overlaps)
    max_overlaps, argmax_overlaps = torch.max(overlaps, 2)
    gt_max_overlaps, _ = torch.max(overlaps, 1)
    print(max_overlaps)
    print(argmax_overlaps)
    print(gt_max_overlaps)

    # gt_max_overlaps[gt_max_overlaps == 0] = 1e-5
    # print(gt_max_overlaps)

    batch_size = 2
    print(overlaps.eq(gt_max_overlaps.view(batch_size, 1, -1).expand_as(overlaps)))
    keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size, 1, -1).expand_as(overlaps)), 2)
    print(keep)

    print(torch.sum(keep))
    print(keep>0)

def fun9():
    labels = torch.Tensor([[1,1,0,0],[1,1,1,0]])
    sum_fg = torch.sum((labels == 1).int(), 1)
    sum_bg = torch.sum((labels == 0).int(), 1)
    print(sum_fg)

def fun10():
    labels = torch.Tensor([[1,1,0,1,0,-1,1],[0,1,1,0,-1,0,0]])
    print("ori_labels:",labels)
    num_fg = 2
    for i in range(2):
        fg_inds = torch.nonzero(labels[i] == 1).view(-1)
        print(fg_inds)
        # torch.randperm seems has a bug on multi-gpu setting that cause the segfault.
        # See https://github.com/pytorch/pytorch/issues/1868 for more details.
        # use numpy instead.
        # rand_num = torch.randperm(fg_inds.size(0)).type_as(gt_boxes).long()
        rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).long()
        print(rand_num)
        disable_inds = fg_inds[rand_num[:fg_inds.size(0) - num_fg]]
        print(disable_inds)
        labels[i][disable_inds] = -1  # 去掉一些正样本（将其标记为-1）
        break
    print(labels)

def fun11():
    a = torch.Tensor([1,1,1,0,0,1,0])
    b = torch.nonzero(a==1)
    print(b)

    c = torch.arange(0, 3) * 20  # gt_boxes.shape=(b,20,5)
    print(c)

def fun12():
    argmax_overlaps = torch.Tensor(5,10)
    offset = torch.Tensor(5)
    out = argmax_overlaps + offset.view(5,1).type_as(argmax_overlaps)
    print(out.shape)

def fun13():
    labels = torch.Tensor([[1,1,0],[0,1,-1]])
    num_examples = torch.sum(labels[1] >= 0)
    positive_weights = 1.0 / num_examples.item()
    negative_weights = 1.0 / num_examples.item()
    print(num_examples)
    print(positive_weights)

def fun14():
    rpn_label = torch.Tensor([[1,1,0,1,-1,0,-1,1,0],[0,0,0,-1,1,1,-1,0,0]])
    rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
    print(rpn_label)
    print(rpn_keep)
    print(rpn_label.view(-1).ne(-1))

def fun15():
    rpn_cls_score = torch.Tensor([4,3,7,1,8,4,5])
    rpn_keep = torch.Tensor([2,5,6]).long()
    rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
    print(rpn_cls_score)

def fun16():
    abs_in_box_diff = torch.Tensor([1,5,3,4,8])
    smoothL1_sign = (abs_in_box_diff < 4).detach().float()
    print(smoothL1_sign)

def fun17():
    loss_box = torch.Tensor([[[[1,2,3],[4,5,6]]],[[[7,8,9],[10,11,12]]]])
    print(loss_box.shape)
    print(loss_box)
    dim = [1,2,3]

    for i in sorted(dim, reverse=True): #dim=[1,2,3]   i=3,2,1
        loss_box = loss_box.sum(i)
        print(loss_box)


    loss_box = loss_box.mean()
    print(loss_box)


def fun18():
    all_rois = torch.Tensor(2,10,5)
    gt_boxes_append = torch.Tensor(2,2,5)
    all_rois = torch.cat([all_rois, gt_boxes_append], 1)
    print(all_rois.shape)

def fun19():
    rois_per_image = 128
    FG_FRACTION = 0.25
    fg_rois_per_image = int(np.round(FG_FRACTION * rois_per_image))
    print(fg_rois_per_image)

def fun20():
    overlaps = torch.Tensor([[[1,2,1,5,4],[2,8,3,8,5],[3,2,5,6,1]]])
    print(overlaps.shape)
    max_overlaps, gt_assignment = torch.max(overlaps, 2)
    print(overlaps)
    print(max_overlaps)
    print(gt_assignment)

def fun21():
    batch_size = 3
    gt_assignment = torch.Tensor([[1,2,3],[4,5,6],[7,8,9]])

    offset = torch.arange(0, batch_size) * 20  # gt_boxes.size(1)=20
    print(offset)
    offset = offset.view(-1,1).type_as(gt_assignment) + gt_assignment
    print(offset)

def fun22():
    batch_size = 3
    gt_boxes = torch.Tensor([[[1,1,1,1],[2,2,2,2]],[[3,3,3,3],[4,4,4,4]],[[5,5,5,5],[6,6,6,6]]])
    print(gt_boxes.shape)
    print(gt_boxes[:, :, 1])

    offset = torch.Tensor([[0,1,0],[1,1,0],[0,0,1]]).long()
    #labels = gt_boxes[:, :, 0].contiguous().view(-1).index((offset.view(-1),)).view(batch_size, -1)
    labels = gt_boxes[:, :, 1].contiguous().view(-1)[(offset.view(-1),)].view(batch_size, -1)
    print(labels)
    print(labels.shape)

def fun23():
    for i in range(3):
        max_overlaps = torch.Tensor([[0.8,0.2,0.6,0.4],[0.1,0.2,0.3,0.8],[0.5,0.6,0.1,0.3]])
        print(max_overlaps.shape)
        fg_inds = torch.nonzero(max_overlaps[i] >= 0.5).view(-1)
        fg_num_rois = fg_inds.numel()
        print(fg_inds)
        print(fg_num_rois)
        break

def fun24():
    s='google'
    if s == '':
        return -1
    str_dict = {}
    for i in s:
        if i in str_dict.keys():
            str_dict[i] += 1
        else:
            str_dict[i] = 1
    for i in range(len(s)):
        if str_dict[s[i]] == 1:
            print(i)
            exit()
    print(-1)

def fun25():
    fg_num_rois = 10
    rand_num = torch.from_numpy(np.random.permutation(fg_num_rois)).long()
    print(rand_num)

    bg_rois_per_this_image = 128 -10
    bg_num_rois = 60
    rand_num = np.floor(np.random.rand(bg_rois_per_this_image) * bg_num_rois)
    print(rand_num)

def fun26():
    fg_inds = torch.Tensor([1,2,3,4])
    bg_inds = torch.Tensor([7,8,9,0])
    keep_inds = torch.cat([fg_inds, bg_inds], 0)
    print(keep_inds)

def fun27():
    clss = torch.Tensor([3,2,3,1 ,5])
    inds = torch.nonzero(clss > 2).view(-1)
    print(inds)

def fun28():
    def FindNumbersWithSum(array, tsum):
        # write code here
        two_sum = array[-1] * array[-2]
        a, b = 0, 0
        p1 = 0
        p2 = len(array) - 1
        while p2 > p1:
            cur_sum = array[p1] + array[p2]
            if cur_sum == tsum:
                if array[p1]*array[p2] < two_sum:
                    a, b = array[p1], array[p2]
                    two_sum = a * b
                p1 += 1
            elif cur_sum < tsum:
                p1 += 1
            elif cur_sum > tsum:
                p2 -= 1

        return a, b

    array = [3,4,5,6,7,8,9,10,11,12,13]
    print(FindNumbersWithSum(array,10))


def fun29():
    bbox_inside_weights = torch.Tensor([[1,1,1,1],[0,0,0,0],[1,1,1,1]])
    bbox_outside_weights = (bbox_inside_weights > 0).float()
    print(bbox_outside_weights)

def fun30():
    class B():
        def __init__(self):
            pass
        def forward(self):
            self.head()

    class A(B):
        def __init__(self):
            pass
        def head(self):
            print("head")

    a = A()
    a.forward()

def fun31():
    n = 4
    m = 5
    num = list(range(n))
    while len(num) > 1:
        for i in range(m - 1):
            num.append(num.pop(0))
        num.pop(0)
    print(num)


def fun32():
    rois_label = torch.Tensor([1,1,3]).long()

    bbox_pred_view = torch.Tensor([[[0,0],[1,1],[2,2],[3,3]],[[4,4],[5,5],[6,6],[7,7]],[[8,8],[9,9],[10,10],[11,11]]])#[1*3,4,2]

    bbox_pred_select = torch.gather(bbox_pred_view, 1,rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 2))
    print(bbox_pred_select)


def fun33():
    loss_box = torch.Tensor([[1,2,3,4],[5,6,7,8]])  # shape=(b*128,4)
    dim = [1]
    for i in sorted(dim, reverse=True):  # dim=[1]   i=1
        loss_box = loss_box.sum(i)
    print(i)
    print(loss_box)
    print(loss_box.mean())

def bbox_transform_inv(boxes, deltas, batch_size):
    '''
    rpn:（rpn预测出所有anchor的回归值之后根据回归值对原始anchor调整）
    :param boxes: anchors.shape=(b,w*h*9,4)就是所有的anchor
    :param deltas: #shape=(b,w*h*9,4)预测的所有的anchor的回归值
    :param batch_size: b
    :return:

    rcnn:(测试时候最后用到)
    #boxes.shape=(b,2000,4) rcnn的输入的框
    #box_deltas.shape=(1,b*2000,4) rcnn预测的框的回归值
    b=1


    '''
    widths = boxes[:, :, 2] - boxes[:, :, 0] + 1.0
    heights = boxes[:, :, 3] - boxes[:, :, 1] + 1.0
    ctr_x = boxes[:, :, 0] + 0.5 * widths
    ctr_y = boxes[:, :, 1] + 0.5 * heights

    dx = deltas[:, :, 0::4]
    dy = deltas[:, :, 1::4]
    dw = deltas[:, :, 2::4]
    dh = deltas[:, :, 3::4]

    pred_ctr_x = dx * widths.unsqueeze(2) + ctr_x.unsqueeze(2)
    pred_ctr_y = dy * heights.unsqueeze(2) + ctr_y.unsqueeze(2)
    pred_w = torch.exp(dw) * widths.unsqueeze(2)
    pred_h = torch.exp(dh) * heights.unsqueeze(2)

    pred_boxes = deltas.clone()
    # x1
    pred_boxes[:, :, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, :, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, :, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, :, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes

def fun34():
    boxes = torch.Tensor(3,100,4)
    deltas = torch.Tensor(3,100,4*21)
    pred_boxes = bbox_transform_inv(boxes, deltas, 1)
    print(pred_boxes.shape)

def fun35():
    thresh = 2
    scores = torch.Tensor([[5,2,3],[4,1,0],[1,5,2]])
    print(scores[:, 0] > thresh)
    print(torch.nonzero(scores[:, 0] > thresh))
    inds = torch.nonzero(scores[:, 0] > thresh).view(-1)
    print(inds)

def fun36():
    cls_scores = torch.Tensor([4,2,6,1,5,3])
    _, order = torch.sort(cls_scores, 0, True)
    print(order)

def fun37():
    a = None
    if a:
        print(1)
    else:
        print(2)

def fun38():
    cls_dets = torch.Tensor([3,1,8,2])
    order = torch.Tensor([2,0,3,1]).long()
    cls_dets = cls_dets[order]
    print(cls_dets)

def fun39():
    bbox = torch.Tensor(21,3,5)
    image_scores = np.hstack([bbox[j][:, -1]
                              for j in range(1,21 )])
    print(image_scores.shape)

def fun40():
    all_boxes = [[[] for _ in range(10)]
                 for _ in range(21)]
    print(all_boxes)
    print(type(all_boxes))

    all_boxes = [[],[]]
    all_boxes[0]=np.array([[1,2],[3,4]])
    print(all_boxes)

def fun41():
    all_box = [[[],[]],[[],[]],[[],[]]]
    all_box[0][0] = np.array([1,1,1,1,0.1])
    all_box[0][1] = np.array([[2,2,2,2,0.2],[2,2,2,2,0.02]])
    all_box[1][1] = np.array([3,3,3,3,0.3])
    all_box[2][0] = np.array([[4,4,4,4,0.4],[5,5,5,5,0.5]])
    print(all_box)


    # det_file = '../../temp_file/faster_rcnn_pytorch/test.pkl'
    # with open(det_file, 'wb') as f:
    #     pickle.dump(all_box, f, pickle.HIGHEST_PROTOCOL)  # 将所有测试图片的结果保存到文件

def fun42():
    num_classes = 21
    _classes = ('__background__',  # always index 0
                     'aeroplane', 'bicycle', 'bird', 'boat',
                     'bottle', 'bus', 'car', 'cat', 'chair',
                     'cow', 'diningtable', 'dog', 'horse',
                     'motorbike', 'person', 'pottedplant',
                     'sheep', 'sofa', 'train', 'tvmonitor')
    _class_to_ind = dict(zip(_classes, range(num_classes)))
    print(_class_to_ind)

def _load_pascal_annotation( index):
    """
    读取图片id为index的图片的标注内容
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    num_classes = 21
    _classes = ('__background__',  # always index 0
                'aeroplane', 'bicycle', 'bird', 'boat',
                'bottle', 'bus', 'car', 'cat', 'chair',
                'cow', 'diningtable', 'dog', 'horse',
                'motorbike', 'person', 'pottedplant',
                'sheep', 'sofa', 'train', 'tvmonitor')
    _class_to_ind = dict(zip(_classes, range(num_classes)))


    data_path = 'F:/Python Workspace/datasets/VOCdevkit/VOC2007'
    filename = os.path.join(data_path, 'Annotations', index + '.xml')#filename=../../00026.xml
    tree = ET.parse(filename)
    objs = tree.findall('object')
    # if not self.config['use_diff']:
    #     # Exclude the samples labeled as difficult
    #     non_diff_objs = [
    #         obj for obj in objs if int(obj.find('difficult').text) == 0]
    #     # if len(non_diff_objs) != len(objs):
    #     #     print 'Removed {} difficult objects'.format(
    #     #         len(objs) - len(non_diff_objs))
    #     objs = non_diff_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, num_classes), dtype=np.float32)
    # "Seg" area for pascal is just the box area
    seg_areas = np.zeros((num_objs), dtype=np.float32)
    ishards = np.zeros((num_objs), dtype=np.int32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text) - 1
        y1 = float(bbox.find('ymin').text) - 1
        x2 = float(bbox.find('xmax').text) - 1
        y2 = float(bbox.find('ymax').text) - 1

        diffc = obj.find('difficult')
        difficult = 0 if diffc == None else int(diffc.text)
        ishards[ix] = difficult

        cls = _class_to_ind[obj.find('name').text.lower().strip()]
        boxes[ix, :] = [x1, y1, x2, y2]
        gt_classes[ix] = cls
        overlaps[ix, cls] = 1.0
        seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

    overlaps = scipy.sparse.csr_matrix(overlaps)

    return {'boxes': boxes,
            'gt_classes': gt_classes,
            'gt_ishard': ishards,
            'gt_overlaps': overlaps,
            'flipped': False,
            'seg_areas': seg_areas}

def fun43():
    indx = '000001'
    obj_dict = _load_pascal_annotation(indx)
    print(obj_dict)
    print('--------------')
    print(obj_dict["gt_overlaps"])


    # need gt_overlaps as a dense array for argmax
    gt_overlaps = obj_dict['gt_overlaps'].toarray()
    # max overlap with gt over classes (columns)
    max_overlaps = gt_overlaps.max(axis=1)
    # gt class that had the max overlap
    max_classes = gt_overlaps.argmax(axis=1)
    print(gt_overlaps)
    print(max_overlaps)
    print(max_classes)

def fun44():
    ratio_list = [0.5,0.2,0.66,0.8,0.1,0.4]


    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    print(ratio_index)

    print(ratio_list[ratio_index])

def fun45():
    import numpy.random as npr
    random_scale_inds = npr.randint(0, high=len((600,)),  # cfg.TRAIN.SCALES=(600,)
                                    size=1)
    print(random_scale_inds)

def fun46():
    gt_classes = np.array([12,0,18])
    gt_inds = np.where(gt_classes != 0)[0]
    print(gt_inds)
if __name__ == "__main__":
    fun46()