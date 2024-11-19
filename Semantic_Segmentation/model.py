import torch
import torch.nn as nn
from torch.nn import init
from torchvision.models.resnet import BasicBlock, ResNet
from torchvision import models
from torch import Tensor

# Returns 2D convolutional layer with space-preserving padding
def conv(in_planes, out_planes, kernel_size=3, stride=1, dilation=1, bias=False, transposed=False):
  if transposed:
    layer = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1, output_padding=1, dilation=dilation, bias=bias)
    # Bilinear interpolation init
    w = torch.Tensor(kernel_size, kernel_size)
    centre = kernel_size % 2 == 1 and stride - 1 or stride - 0.5
    for y in range(kernel_size):
      for x in range(kernel_size):
        w[y, x] = (1 - abs((x - centre) / stride)) * (1 - abs((y - centre) / stride))
    layer.weight.data.copy_(w.div(in_planes).repeat(in_planes, out_planes, 1, 1))
  else:
    padding = (kernel_size + 2 * (dilation - 1)) // 2
    layer = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
  if bias:
    init.constant(layer.bias, 0)
  return layer


# Returns 2D batch normalisation layer
def bn(planes):
  layer = nn.BatchNorm2d(planes)
  # Use mean 0, standard deviation 1 init
  init.constant(layer.weight, 1)
  init.constant(layer.bias, 0)
  return layer

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
  """3x3 convolution with padding"""
  return nn.Conv2d(
    in_planes,
    out_planes,
    kernel_size=3,
    stride=stride,
    padding=dilation,
    groups=groups,
    bias=False,
    dilation=dilation,
  )

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
  """1x1 convolution"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class FeatureResNet(ResNet):
  def __init__(self):
    super().__init__(BasicBlock, [3, 4, 6, 3], 1000)
    # self.layer1 = nn.Sequential(
    #   nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),

  def forward(self, x):
    x1 = self.conv1(x)
    x = self.bn1(x1)
    x = self.relu(x)
    x2 = self.maxpool(x)
    x = self.layer1(x2)
    x3 = self.layer2(x)
    x4 = self.layer3(x3)
    x5 = self.layer4(x4)
    return x1, x2, x3, x4, x5

class SegResNet(nn.Module):
  def __init__(self, num_classes, pretrained_net):
    super().__init__()

    self.pretrained_net = pretrained_net
    self.relu1 = nn.ReLU(inplace=True)
    self.relu2 = nn.ReLU(inplace=True)
    self.relu3 = nn.ReLU(inplace=True)
    self.relu4 = nn.ReLU(inplace=True)
    self.relu5 = nn.ReLU(inplace=True)
    self.conv5 = conv(512, 256, stride=2, transposed=True)
    self.bn5 = bn(256)
    self.conv6 = conv(256, 128, stride=2, transposed=True)
    self.bn6 = bn(128)
    self.conv7 = conv(128, 64, stride=2, transposed=True)
    self.bn7 = bn(64)
    self.conv8 = conv(64, 64, stride=2, transposed=True)
    self.bn8 = bn(64)
    self.conv9 = conv(64, 32, stride=2, transposed=True)
    self.bn9 = bn(32)
    self.conv10 = conv(32, num_classes, kernel_size=7)
    init.constant(self.conv10.weight, 0)  # Zero init

  def forward(self, x):
    x1, x2, x3, x4, x5 = self.pretrained_net(x) # x的尺寸为[batch_size, 3, w, h],
    # x1的尺寸为[batch_size, 64, w/2, h/2],
    # x2的尺寸为[batch_size, 64, w/4, h/4]
    # x3的尺寸为[batch_size, 128, w/8, h/8],
    # x4的尺寸为[batch_size, 256, w/16, h/16],
    # x5的尺寸为[batch_size, 512, w/32, h/32]
    x = self.relu1(self.bn5(self.conv5(x5))) # x的尺寸为[batch_size, 256, w/16, h/16]
    x = self.relu2(self.bn6(self.conv6(x + x4))) # x的尺寸为[batch_size, 128, w/8, h/8]
    x = self.relu3(self.bn7(self.conv7(x + x3)))  # x的尺寸为[batch_size, 64, w/4, h/4]
    x = self.relu4(self.bn8(self.conv8(x + x2)))
    x = self.relu5(self.bn9(self.conv9(x + x1)))
    x = self.conv10(x)
    return x




class OnlineSegResNet(nn.Module):
  def __init__(self, num_classes, singleneuron, **kwargs):
    super(OnlineSegResNet, self).__init__()
    self.conv1 = SequentialModule(singleneuron,nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),singleneuron(**kwargs)
    )

    self.layer2 = SequentialModule( # 经过首层卷积后得到x2
     singleneuron,WrapedSNNOp(nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)), nn.BatchNorm2d(64),
      singleneuron(**kwargs),
      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

    self.layer3 = self.make_layers(64, 128, stride=2, neurons=singleneuron, **kwargs)
    self.layer4 = self.make_layers(128, 256, stride=2, neurons=singleneuron, **kwargs)
    self.layer5 = self.make_layers(256, 512, stride=2, neurons=singleneuron, **kwargs)


    self.layer6 = self.make_layers_step_2([
      WrapedSNNOp(nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1, bias=False)),
      nn.BatchNorm2d(256)],
      neurons=singleneuron, **kwargs
    )
    self.layer7 =self.make_layers_step_2([
      WrapedSNNOp(
        nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1, bias=False)),
        nn.BatchNorm2d(128)],
        neurons=singleneuron, **kwargs
    )
    self.layer8 = self.make_layers_step_2([WrapedSNNOp(nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1, bias=False)),
        nn.BatchNorm2d(64)],
        neurons=singleneuron, **kwargs
    )
    self.layer9 = self.make_layers_step_2([
        WrapedSNNOp(nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1, bias=False)),
        nn.BatchNorm2d(64)],
        neurons=singleneuron, **kwargs
    )
    self.layer10 = self.make_layers_step_2([
        WrapedSNNOp(nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, dilation=1, bias=False)),
        nn.BatchNorm2d(32)],
        neurons=singleneuron, **kwargs
    )
    self.layer11 = WrapedSNNOp(nn.Conv2d(32, num_classes, kernel_size=7, stride=1, padding=3, dilation=1, bias=False))

  def make_layers_step_2(self, layerlist, neurons, **kwargs):
      layers = []
      for A in layerlist:
        layers.append(A)
      layers.append(neurons(**kwargs))
      return SequentialModule(neurons, *layers)


  def make_layers(self, in_channels, out_channels, neurons, stride=1, **kwargs):
    layers = []
    downsample = None
    if stride != 1 or in_channels != out_channels:
      downsample = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
    conv2d1 = WrapedSNNOp(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False))
    layers += [conv2d1, bn(out_channels), neurons(**kwargs)]
    conv2d2 = WrapedSNNOp(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
    layers += [conv2d2, bn(out_channels), neurons(**kwargs)]
    conv2d2 = WrapedSNNOp(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False))
    layers += [conv2d2, bn(out_channels), neurons(**kwargs)]
    return ResSequentialModule(neurons, downsample, *layers)

  def forward(self, x, **kwargs):
    x1 = self.conv1(x, **kwargs)  # x1的尺寸为[2*batch_size, 64, w/2, h/2],

    x2 = self.layer2(x1, **kwargs)  # x2的尺寸为[2 * batch_size, 64, w/4, h/4]

    x3 = self.layer3(x2, **kwargs)  # x3的尺寸为[batch_size, 128, w/8, h/8],
    x4 = self.layer4(x3, **kwargs)  # x4的尺寸为[batch_size, 256, w/16, h/16],
    x5 = self.layer5(x4, **kwargs)  # x5的尺寸为[batch_size, 512, w/32, h/32]

    x6 = self.layer6(x5, **kwargs)  # x的尺寸为[batch_size, 256, w/16, h/16]
    x7 = self.layer7(x6 + x4, **kwargs)  # x的尺寸为[batch_size, 128, w/8, h/8]
    x8 = self.layer8(x7 + x3, **kwargs)  # x的尺寸为[batch_size, 64, w/4, h/4]
    x9 = self.layer9(x8 + x2, **kwargs)  # x的尺寸为[batch_size, 64, w/2, h/2]
    x10 = self.layer10(x9 + x1, **kwargs)  # x的尺寸为[batch_size, 32, w, h]
    x11 = self.layer11(x10, **kwargs)  # x的尺寸为[batch_size, num_classes, w, h]
    return x11


class ResSequentialModule(nn.Sequential):

  def __init__(self, single_step_neuron, downsample=None, *args):
    super(ResSequentialModule, self).__init__(*args)
    self.single_step_neuron = single_step_neuron
    self.downsample = downsample



  def forward(self, input, **kwargs):
    identity = input
    for i, module in enumerate(self._modules.values()):
      if module==self.downsample:
        break
      else:
        if isinstance(module, WrapedSNNOp) or isinstance(module,self.single_step_neuron):
          input = module(input, **kwargs)
        else:
          input = module(input)

    if self.downsample is not None:
      identity = self.downsample(identity)
    input += identity
    return input

  def get_spike(self):
    spikes = []
    for module in self._modules.values():
      if isinstance(module, self.single_step_neuron):
        spike = module.spike.cpu()
        spikes.append(spike.reshape(spike.shape[0], -1))
    return spikes


class SequentialModule(nn.Sequential):

  def __init__(self, single_step_neuron, *args):
    super(SequentialModule, self).__init__(*args)
    self.single_step_neuron = single_step_neuron

  def forward(self, input, **kwargs):
    for i, module in enumerate(self._modules.values()):
      if isinstance(module, self.single_step_neuron) or isinstance(module, WrapedSNNOp):
        input = module(input, **kwargs)
      else:
        input = module(input)
    return input

  def get_spike(self):
    spikes = []
    for module in self._modules.values():
      if isinstance(module, self.single_step_neuron):
        spike = module.spike.cpu()
        spikes.append(spike.reshape(spike.shape[0], -1))
    return spikes

class ScaledNeuron(nn.Module):
  def __init__(self, scale=1.):
    super(ScaledNeuron, self).__init__()
    self.scale = scale
    self.t = 0
    self.neuron = neuron.IFNode(v_reset=None)

  def forward(self, x):
    x = x / self.scale
    if self.t == 0:
      self.neuron(torch.ones_like(x) * 0.5)  # 初始膜电位为Vth的1/2，有助于大幅减少转换损失
    x = self.neuron(x)
    self.t += 1
    return x * self.scale

  def reset(self):
    self.t = 0
    self.neuron.reset()



class Replace(Function):
  @staticmethod
  def forward(ctx, z1, z1_r):
    return z1_r

  @staticmethod
  def backward(ctx, grad):
    return (grad, grad)


class WrapedSNNOp(nn.Module):

  def __init__(self, op):
    super(WrapedSNNOp, self).__init__()
    self.op = op

  def forward(self, x, **kwargs):
    require_wrap = kwargs.get('require_wrap', True)
    if require_wrap:
      B = x.shape[0] // 2
      spike = x[:B]
      rate = x[B:]
      with torch.no_grad():
        out = self.op(spike).detach()
      in_for_grad = Replace.apply(spike, rate)
      out_for_grad = self.op(in_for_grad)
      output = Replace.apply(out_for_grad, out)
      return output
    else:
      return self.op(x)