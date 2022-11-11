import torch
from torch.autograd import Function
from torch import nn
import torch.nn.functional as F
import numpy as np
from KKKMeans import kmeans, kmeans_predict
from sklearn.cluster import KMeans

class LinearAverage(nn.Module):
    def __init__(self, inputSize, outputSize, T=0.05, momentum=0.0):
        """
        :param inputSize: 2048, dim of resnet last output layer
        :param outputSize: target domain sample num
        :param T: temp
        """
        super(LinearAverage, self).__init__()
        self.nLem = outputSize
        self.T = T
        self.momentum = momentum
        self.register_buffer('params', torch.tensor([T, momentum]))
        self.register_buffer('memory', torch.zeros(outputSize, inputSize))
        self.register_buffer('labels', torch.tensor([0 for _ in range(outputSize)], dtype=torch.long))
        self.flag = 0
        self.memory = self.memory.cuda()
        self.labels = self.labels.cuda()

    def forward(self, x, y=None):
        out = torch.mm(x, self.memory.t()) / self.T
        return out

    def update_weight(self, features, pseudo_labels, index):
        pseudo_labels = torch.from_numpy(pseudo_labels).cuda()
        if not self.flag:
            weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
            weight_pos.mul_(0.0)
            weight_pos.add_(torch.mul(features.data, 1.0))
            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            self.memory.index_copy_(0, index, updated_weight)

            self.labels.index_copy_(0, index, pseudo_labels)
            self.flag = 1
        else:
            weight_pos = self.memory.index_select(0, index.data.view(-1)).resize_as_(features)
            weight_pos.mul_(self.momentum)
            weight_pos.add_(torch.mul(features.data, 1 - self.momentum))

            w_norm = weight_pos.pow(2).sum(1, keepdim=True).pow(0.5)
            updated_weight = weight_pos.div(w_norm)
            self.memory.index_copy_(0, index, updated_weight)
            self.labels.index_copy_(0, index, pseudo_labels)
        self.memory = F.normalize(self.memory)  # .cuda()

    def set_weight(self, features, index):
        self.memory.index_copy_(0, index, features)


    def get_target_centroid(self, device, num_clusters, way=1):
        """
        prototypes = torch.zeros((self.memory.shape[1], num_class)).cuda()
        label_num = torch.zeros(num_class).cuda()

        for cls in range(num_class):
            idx = torch.where(self.labels == cls)[0]
            if idx.shape[0] == 0:
                continue
            label_num[cls] += idx.shape[0]
            feat_cls = self.memory[idx, :]
            prototype = torch.sum(feat_cls, dim=0, keepdim=False)
            prototypes[:, cls] = prototype
        return prototypes, label_num
        """
        if way == 1:
            X = self.memory.clone().detach().cpu().numpy()
            km = KMeans(num_clusters).fit(X)
            cc = km.cluster_centers_
            cluster_centers = torch.from_numpy(cc).to(device)
            return cluster_centers
        else:
            cluster_ids_x, cluster_centers = kmeans(
                X=self.memory.clone().detach(), num_clusters=num_clusters, distance='euclidean', device=device
            )
            return cluster_centers
