import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# 1. maxpool
def maxpooling(feature, kernel, pad, stride):
    c, h, w = feature.shape
    kh, kw = kernel.shape
    feature_pad = np.pad(feature, ((0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=(0, 0))
    new_h, new_w = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1

    new_feature = np.zeros((3, new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            roi = feature_pad[:, i * stride: i * stride + kh, j * stride: j * stride + kw]
            point = np.max(roi, axis=(1, 2))
            new_feature[:, i, j] = point
    return

# 2. avepooling
def average_pooling(feature, kernel, pad, stride):
    c, h, w = feature.shape
    kh, kw = kernel.shape
    feature_pad = np.pad(feature, ((0,0), (pad, pad),(pad, pad)), mode='constant', constant_values=(0,0))
    new_h, new_w = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1
    new_feature = np.zeros((c, new_h, new_w))
    for i in range(new_h):
        for j in range(new_w):
            roi = feature_pad[:, i * stride: i * stride + kh, j * stride: j * stride + kw]
            point = np.mean(roi, axis=(1, 2))
            new_feature[:, i, j] = point
    print(new_feature)
    return new_feature

# 3.conv
def conv(feature, kernel, pad, stride):
    c, h, w = feature.shape
    c_out, c_in, kh, kw = kernel.shape
    feature_pad = np.pad(feature, ((0,0),(pad, pad),(pad,pad)), mode='constant', constant_values=(0,0))
    new_h, new_w = (h + 2 * pad - kh) // stride + 1, (w + 2 * pad - kw) // stride + 1
    new_feature = np.zeros((c_out, new_h, new_w))

    for i in range(new_h):
        for j in range(new_w):
            for p in range(c_out):
                roi = feature_pad[:, i * stride: i * stride + kh, j * stride: j * stride + kw]
                new_feature[p, i, j] = np.sum(roi * kernel[p, :, :, :].squeeze())
    print(new_feature)
    return new_feature


# todo np.maximum两个响像逐位去大小，np.max一个向量沿着某个轴取最大
def iou(preds, gt):
    # preds -> (n, 4)  gt (4,)
    left_top_x = np.maximum(preds[:, 0], gt[0])
    left_top_y = np.maximum(preds[:, 1], gt[1])
    right_bottom_x = np.minimum(preds[:, 2], gt[2])
    right_bottom_y = np.minimum(preds[:, 3], gt[3])

    inner_w = np.maximum(right_bottom_x - left_top_x, 0)
    inner_h = np.maximum(right_bottom_y - left_top_y, 0)
    inner_area = inner_w * inner_h

    union_area = (gt[3] - gt[1]) * (gt[2] - gt[0]) + (preds[:, 3] - preds[:, 1]) * (preds[:, 2] - preds[:, 0]) - inner_area
    IOU = inner_area / union_area
    maxIndex = np.argmax(IOU)
    maxIOU = np.max(IOU)
    print(IOU)
    print(maxIndex)
    print(maxIOU)
    return IOU, maxIndex, maxIOU

def nms(preds, threshold):
    # preds  (n,5)xyxy score
    x1, y1, x2, y2, score = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3], preds[:, 4]
    order = np.argsort(score)[::-1]   # 高-低
    keep = []    # 保留的idx
    areas = (y2 - y1) * (x2 - x1)
    while(order.size > 0):
        maxIndex = order[0]
        keep.append(maxIndex)
        left_top_x = np.maximum(x1[maxIndex], x1[order[1:]])
        left_top_y = np.maximum(y1[maxIndex], y1[order[1:]])
        right_bottom_x = np.minimum(x2[maxIndex], x2[order[1:]])
        right_bottom_y = np.minimum(y2[maxIndex], y2[order[1:]])

        inner_w = np.max(right_bottom_x - left_top_x, 0)
        inner_h = np.max(right_bottom_y - left_top_y, 0)
        inner_area = inner_w * inner_h

        union_area = areas[maxIndex] + areas[order[1:]] - inner_area
        IOU = inner_area / union_area
        idx = np.where(IOU <= threshold)[0]
        if idx.size <= 0:
            break
        order = order[idx + 1]  # 从1开始
    return preds[keep]



# 手写个CNN网络
class Net(nn.Module):
    def __init__(self, h, w, nc):
        super(Net, self).__init__()
        self.layer = nn.Sequential(*[
            nn.Conv2d(3, 64, kernel_size=3, padding=1,stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(h * w * 128, nc)
        ])
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1)
        self.fc = nn.Linear(112 * 112 * 128, nc)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        print(x.shape)
        x = x.flatten(start_dim=1, end_dim=3)
        print(x.shape)
        x = self.fc(x)

        return x





if __name__ == '__main__':
    # feature = np.array([x for x in range(48)]).reshape(3, 4, 4)
    # print(feature)
    # kernel = np.zeros((2,3,3,3))
    # conv(feature, kernel, pad=1, stride=1)

    # preds = np.array([[1,2,3,4], [2,2,3,3],[2,3,4,5]])
    # gt = np.array([1,1,2,3])
    # iou(preds, gt)

    img = torch.ones((1,3,224,224))
    B,C,H,W = img.shape
    nc = 10
    model = Net(h=H, w=W, nc=nc)
    res = model(img)
    print(res.shape)






