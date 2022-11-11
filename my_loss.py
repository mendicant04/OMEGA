import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os

def entropy(p):
    p = F.softmax(p, dim=1)
    return -torch.mean(torch.sum(p * torch.log(p + 1e-5), 1))


def entropy_margin(p, value, margin=0.2, weight=None):
    p = F.softmax(p, dim=1)
    return -torch.mean(hinge(torch.abs(-torch.sum(p * torch.log(p + 1e-5), 1) - value), margin))


def hinge(input, margin=0.2):
    return torch.clamp(input, min=margin)


def KLdiv(a, b, eps=1e-5):
    tmp = a * (torch.log(a + eps) - torch.log(b + eps))
    ans = tmp.sum()
    return ans


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets,
                                  reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


def get_target_centroid(generator, classifier, target_loader, num_class, device, threshold):
    generator.eval()
    classifier.eval()
    feat_all = None
    labels_all = None
    feat_size = 0
    with torch.no_grad():
        for idx, data_t in enumerate(target_loader):
            img_t = data_t[0].to(device)
            feat_t = generator(img_t)
            feat_size = feat_t.shape[1]
            out_t = classifier(feat_t)

            out_t_s = F.softmax(out_t, dim=1)
            entr = -torch.sum(out_t_s * torch.log(out_t_s), 1).data.cpu().numpy()
            pred = out_t_s.data.max(1)[1].cpu().numpy()
            pred_unk = np.where(entr > threshold)
            pred[pred_unk[0]] = num_class
            pred = torch.from_numpy(pred)

            if feat_all is None:
                feat_all = feat_t
                labels_all = pred
            else:
                feat_all = torch.cat((feat_all, feat_t), dim=0)
                labels_all = torch.cat((labels_all, pred), dim=0)

    prototypes = torch.zeros((feat_size, num_class))

    label_num = np.zeros(num_class)
    for cls in range(num_class):
        idx = np.where(labels_all == cls)[0]
        if idx.shape[0] == 0:
            continue
        label_num[cls] += idx.shape[0]
        feat_cls = feat_all[idx, :]
        prototype = torch.sum(feat_cls, dim=0, keepdim=False)
        prototypes[:, cls] = prototype

    return prototypes, labels_all
