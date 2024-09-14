import torch.nn as nn
import torch
import torch.nn.functional as F
from SpikingNeuron_AAS import AASNeuron


def isActivation(name):
    if 'relu' in name.lower() or 'clip' in name.lower() or 'floor' in name.lower() or 'tcl' in name.lower():
        return True
    return False

class Wrapper(nn.Module):
    def __init__(self, pretrained_model, T):
        super(Wrapper, self).__init__()
        self.T = T
        self.pretrained_model = pretrained_model

    def forward(self, x):
        output = self.pretrained_model(x)
        for t in range(1, self.T):
            output+=self.pretrained_model(x)
            torch.cuda.empty_cache()
        return output/self.T


class AAS_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.AAS = nn.Parameter(torch.Tensor([4.]), requires_grad=True)
    def forward(self, x):
        x = F.relu(x, inplace='True')
        x = self.AAS - x
        x = F.relu(x, inplace='True')
        x = self.AAS - x
        return x


def replace_activation_by_AAS(model, t):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_AAS(module, t)
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "AAS"):
                model._modules[name] = AAS_layer()
    return model


def replace_activation_by_AASneuron(model):
    for name, module in model._modules.items():
        if hasattr(module,"_modules"):
            model._modules[name] = replace_activation_by_AASneuron(module)
        if isActivation(module.__class__.__name__.lower()):
            if hasattr(module, "AAS"):
                model._modules[name] = AASNeuron(scale=module.AAS)
            else:
                model._modules[name] = AASNeuron(scale=1.)
    return model

