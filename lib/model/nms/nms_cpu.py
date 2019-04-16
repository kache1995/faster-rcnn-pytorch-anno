from __future__ import absolute_import

import numpy as np
import torch

def nms_cpu(dets, thresh):
    '''

    :param dets: shape=(2000,5) 5=[x1,y1,x2,y2,score]
    :param thresh:
    :return:
    '''
    dets = dets.numpy()
    x1 = dets[:, 0]#shape=(2000)
    y1 = dets[:, 1]#shape=(2000)
    x2 = dets[:, 2]#shape=(2000)
    y2 = dets[:, 3]#shape=(2000)
    scores = dets[:, 4]#shape=(2000)

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order.item(0)
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return torch.IntTensor(keep)

#
# if __name__ == "__main__":
#     a = torch.rand(100,5)
#     keep = nms_cpu(a,0.1)
#     print(keep.shape)
#     print(keep)