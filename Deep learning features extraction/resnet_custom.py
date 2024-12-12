# modified from Pytorch official resnet.py
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch
#from torchsummary import summary
import torch.nn.functional as F
from torchvision import models
# import clip


available_policies = {"resnet18": models.resnet18,"resnet50": models.resnet50, "vgg16": models.vgg16, "vgg19": models.vgg19,
                      "alexnet": models.alexnet, "inception": models.inception_v3}



__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def model_baseline(model_name = 'resnet50',pretrained=False):
    """Constructs a Modified ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # model = ResNet_Baseline(Bottleneck_Baseline, [3, 4, 6, 3])
    model = available_policies[model_name](pretrained=pretrained)
    model=Net(model)

    return model



def load_pretrained_weights(model, name):
    pretrained_dict = model_zoo.load_url(model_urls[name])
    model.load_state_dict(pretrained_dict, strict=False)
    return model

'''
class Net(nn.Module):  # 没有临床信息
    def __init__(self, model):
        super(Net, self).__init__()
        self.layer = nn.Sequential(*list(model.children()))
        self.conv = self.layer[:-1]
        self.dense = None
        if type(self.layer[-1]) == type(nn.Sequential()):
            self.feature = self.layer[-1][-1].in_features
            self.dense = self.layer[-1][:-1]
        else:
            self.feature = self.layer[-1].in_features
        self.linear = nn.Linear(self.feature, 2)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        if self.dense is not None:
            x = self.dense(x)
        #print(x.size())
        #x = self.linear(x)
        return x
'''
class Net(nn.Module):  
    def __init__(self, model):
        super(Net, self).__init__()
        self.layer = nn.Sequential(*list(model.children()))
        self.conv = self.layer[:-1]#获取除最后一层外的所有子模块

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        

        return x
