import torch.nn as nn
import torch

class OnlineIFNode(IFNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = None,
            surrogate_function = surrogate.Sigmoid(), detach_reset: bool = True,
             **kwargs):

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v.detach() + x

    def forward_init(self, x: torch.Tensor):
        self.v = torch.ones_like(x)*0.5
        self.rate_tracking = None

    def forward(self, x: torch.Tensor, **kwargs):
        init = kwargs.get('init', False)
        save_spike = kwargs.get('save_spike', False)
        if init:
            self.forward_init(x)

        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)

        if save_spike:
            self.spike = spike

        with torch.no_grad():
            if self.rate_tracking == None:
                self.rate_tracking = spike.clone().detach()
            else:
                self.rate_tracking = self.rate_tracking + spike.clone().detach()
        return torch.cat((spike, self.rate_tracking), dim=0)


class AASNeuron(nn.Module):
    '''
    AAS neuron
    '''
    def __init__(self, AAS=1.):
        super(AASNeuron, self).__init__()
        self.AAS = AAS
        self.t = 0
        self.neuron = OnlineIFNode(v_reset=None)

    def forward(self, x, **kwargs):
        x = x / self.AAS
        x = self.neuron(x, **kwargs)
        self.t += 1
        return x * self.AAS
    def reset(self):
        self.t = 0
        self.neuron.reset()
