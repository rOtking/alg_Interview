import cv2 as cv
import numpy as np
import torch
import random
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def disEclud(v1, v2):
    # 两个向量计算欧式距离
    return np.sqrt(np.sum(np.power(v1 - v2, 2)))


def kmeans(dataset, k):
    '''
    :param dataset: (m,n) m个样本，n维
    :param k: 中心个数
    :return:
    '''
    m, n = dataset.shape
    # 1.随机初始化k个样本为聚类中心
    # replace为是否可取相同元素
    clusterCenter = dataset[np.random.choice([i for i in range(m)], k, replace=False)]

    clusterAssment = np.zeros((m, 2)) # 第一类是该样本属于哪个cluster；第二列是到该中心的距离
    iteration = 100
    while(iteration > 0):
        # 2.计算每个样本到各个cluster的距离
        for i in range(m):
            minIndex = 0
            minDist = np.inf
            for j in range(k):
                # 计算距离
                dist = disEclud(dataset[i, :], clusterCenter[j, :])
                if dist < minDist:
                    minIndex = j
                    minDist = dist
            clusterAssment[i, :] = minIndex, minDist

        # 3.重新分配每个簇的中心
        for center in range(k):
            mask = clusterAssment[:, 0] == center
            dateInCluster = dataset[mask]   # 取出属于本类的所有点
            # 计算本类新的聚类中心
            if dateInCluster.shape[0] > 0:
                clusterCenter[center, :] = np.mean(dateInCluster, axis=0)
        iteration -= 1

    return clusterCenter, clusterAssment


def getIou(preds, gt):
    '''
    貌似不能直接实现gts多维的直接计算，顶多是多预测与一个gt
    :param preds: (m, 4)
    :param gts: (4,)
    :return:
    '''
    left_top_x = np.maximum(preds[:, 0], gt[0])
    left_top_y = np.maximum(preds[:, 1], gt[1])
    right_bottom_x = np.minimum(preds[:, 2], gt[2])
    right_bottom_y = np.minimum(preds[:, 3], gt[3])

    # 注意不相交会出现 左上的坐标 > 右下的坐标
    # todo 这里是做精彩的，把不相交的情况巧妙的解决了
    inner_w = np.maximum(right_bottom_x - left_top_x, 0.)
    inner_h = np.maximum(right_bottom_y - left_top_y, 0.)
    inner_area = inner_w * inner_h   # 不相交面积=0
    # union area = s1 + s2 - inner
    # 广播
    union_area = (gt[2] - gt[0]) * (gt[3] - gt[1]) + \
                 ((preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])) - \
                 inner_area
    iou = inner_area / union_area
    print(iou)
    # 获取最大
    maxIndex = np.argmax(iou)
    maxIou = np.max(iou)
    print(maxIndex)
    print(maxIou)

    return iou

def getIouForAnchor(box, centers):
    '''
    多个centers与每个box计算
    :param box:
    :param centers:
    :return:
    '''
    inner_w = np.minimum(box[0], centers[:,0])
    inner_h = np.minimum(box[1], centers[:,1])

    inner_area = inner_w * inner_h

    union_area = box[0] * box[1] + centers[:,0] * centers[:,1] - inner_area
    iou = inner_area / union_area
    return iou

def kmeansForAnchor(boxes, k):
    '''

    :param boxes:(m,2)   wh
    :param k:
    :return:
    '''
    m = boxes.shape[0]
    # 1.初始化center
    clusterCenter = boxes[np.random.choice([i for i in range(m)], k, replace=False)]
    clusterAssment = np.zeros((m,))
    iteration = 100
    while(iteration > 0):
        for i in range(m):
            dists = 1 - getIouForAnchor(boxes[i], clusterCenter)
            minIndex = np.argmin(dists)
            clusterAssment[i] = minIndex

        # 重分配每个中心
        for center in range(k):
            mask = clusterAssment == center
            boxInCluster = boxes[mask]
            if boxInCluster.shape[0] >0:
                clusterCenter[center] = np.mean(boxInCluster, axis=0)
        iteration -= 1

    print(clusterCenter)

def nms(dets, iou_threshold):
    '''
    原始预测肯定是 box为(m,4),score为(m,n)的多分类
    可得到每个box的最大类，也就是它的label，因为它就算它的max_score再低，它也是这个box预测的所有类最大的

    那就可以整理成
    class1:[(x,y,x,y,score1),...]
    class2:[(x,y,x,y,score2),...]
    ...

    所有box都有具体的最大分类与分数，按类别划分好了

    所以nms是对每个类别分别进行的，但是每个类内的逻辑是一样的，所以只写一个类的即可，多类就是forloop
    且 肯定不会出现一个box多个类！很爽！！！！

    :param dets:(m,5)   [x,y,x,y,conf]
    :param iou_threshold:
    :return:
    '''

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    score = dets[:, 4]

    order = np.argsort(score)[::-1]   # score从高到低排序
    keep = []   # 保留的box的index

    areas = (x2 - x1) * (y2 - y1)   # 每个box的面积
    while(len(order)):
        i = order[0]
        keep.append(i)

        # 计算当前box与剩余其他box的iou
        left_top_x = np.maximum(x1[i], x1[i+1:])
        left_top_y = np.maximum(y1[i], y1[i+1:])
        right_bottom_x = np.minimum(x2[i], x2[i+1:])
        right_bottom_y = np.minimum(y2[i], y2[i+1:])

        inner_w = np.maximum(right_bottom_x - left_top_x, 0.)
        inner_h = np.maximum(right_bottom_y - left_top_y, 0.)
        inner_area = inner_w * inner_h

        union_area = areas[i] + areas[i+1:] - inner_area    # 广播
        iou = inner_area / union_area
        # 得到满足条件的order
        idx = np.where(iou < iou_threshold)[0]
        # size属性：不管array是几维，size是所有元素的个数
        # shape就是维度信息
        if idx.size == 0:
            break
        order = order[idx + 1]   # todo： 关键！iou的条件求出得idx是从0开始的，而order是有首元素的，+1得到真正的order顺序

    return keep




# 浮点数开方，保留3位小数
def sqrt(x):
    left, right = 0, x
    while(left <= right):
        mid = left + (right - left) / 2
        if -1e-1 < mid ** 2 - x <= 1e-3:
            print(mid)
            return mid
        elif mid ** 2 < x:
            left = mid
        elif mid ** 2 > x:
            right = mid
        else:
            pass


if __name__ == '__main__':
    # m = 8
    # n = 2
    # np.random.seed(42)
    # dataset = np.random.random((m, n))
    # #
    # k = 2
    # res = kmeansForAnchor(dataset, k)
    # res = kmeans(dataset, k)
    # print(res)
    # preds = np.array([[0,0,2,3],[2,1,3,4],[0,1,2,3]])
    # gt = np.array([1,1,3,4])
    # getIou(preds, gt)

    # dets = np.array([[1,2,3,4,0.9],[2,2,3,4,0.8],[1,2,3,4.5,0.8]])
    # keep = nms(dets, 0.7)
    # print(keep)

    # B, C, H, W = 8, 256, 52, 52
    # heads = 4
    # imgs = torch.randn((B,C,H,W))
    # embedding = imgs.flatten(2).permute((0,2,1))
    # print(embedding.shape)
    #
    # qkv_op = nn.Linear(C, 3 * C)
    # qkv = qkv_op(embedding).reshape(B, H*W, 3, heads, C // heads)
    # print(qkv.shape)
    # qkv = qkv.permute(2, 0, 3, 1, 4)
    # print(qkv.shape)
    # q, k, v = qkv[0], qkv[1], qkv[2]
    # print(q.shape)
    #
    # attn = q@k.transpose(-1,-2)
    # attn = attn.softmax(dim=-1)
    # x = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
    # print(x.shape)
    # layer = nn.LayerNorm(C)
    # out = layer(x)
    # print(out.shape)

    sqrt(15)

