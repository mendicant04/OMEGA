from torchvision import models
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function, Variable


class ResBase(nn.Module):
    def __init__(self, option='resnet50', pret=True):
        super(ResBase, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        elif option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        elif option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        elif option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        else:
            model_ft = models.resnet50(pretrained=pret)
        mod = list(model_ft.children())
        mod.pop()  # 应该是为了最后一个全连接不要了
        self.features = nn.Sequential(*mod)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.dim)
        return x


class ResClassifier_MME(nn.Module):
    """
        存放的内容是源域原型，因此维度是2048 * 10
    """
    def __init__(self, num_classes=10, input_size=2048, temp=0.05):
        super(ResClassifier_MME, self).__init__()
        self.fc = nn.Linear(input_size, num_classes, bias=False)
        self.tmp = temp

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False, reverse=False):
        if return_feat:
            return x
        x = F.normalize(x)
        x = self.fc(x) / self.tmp
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))

    def weights_init(self, m):
        m.weight.data.normal_(0.0, 0.1)

    def get_source_prototypes(self):
        tmp = self.fc.weight.clone().detach()
        tmp = tmp.T
        return tmp
