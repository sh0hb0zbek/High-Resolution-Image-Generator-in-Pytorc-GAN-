import torch
import torch.nn as nn
import torch.nn.functional as F
from tensor_ops import pixel_norm, upsample, downsample, minibatch_stddev, blur

import math
from enum import Enum


class PixelNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pixel_norm(x)


class Upsample(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return upsample(x)


class Downsample(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return downsample(x)


class MinibatchStddev(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return minibatch_stddev(x)


class ScaledLeakyRelu(nn.Module):
    def __init__(self, alpha: float = 0.2,gain: float = math.sqrt(2.)):
        super(ScaledLeakyRelu, self).__init__()
        self.alpha = alpha
        self.gain = gain

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.leaky_relu(x, self.alpha) * self.gain


class ImageConversionMode(Enum):
    TORCH_TO_MODEL = 0 # [ 0, 1] to [-1, 1]
    MODEL_TO_TORCH = 1 # [-1, 1] to [ 0, 1]


class ImageConversion(nn.Module):
    def __init__(self, conversion_mode: 'ImageConversionMode', **kwargs):
        super(ImageConversion, self).__init__()
        self.conversion_mode = conversion_mode
    
    def forward(self, image):
        if self.conversion_mode == ImageConversionMode.TORCH_TO_MODEL:
            return image * 2. - 1.   # [ 0, 1] to [-1, 1]
        elif self.conversion_mode == ImageConversionMode.MODEL_TO_TORCH:
            return image * 0.5 + 0.5 # [-1, 1] to [ 0, 1]
        else:
            assert False, f"Unknown conversion mode: {self.conversion_mode}"


class ScaledAdd(nn.Module):
    def __init__(self, scale: float = 1. / math.sqrt(2.)):
        super(ScaledAdd, self).__init__()
        self.scale_value = scale
        self.scale = nn.Parameter(torch.tensor(scale), requires_grad=False)

    def forward(self, a ,b):
        assert a.shape[1:] == b.shape[1:], f"{a.shape} != {b.shape}"
        return (a + b) * self.scale
    

class ScaledConv2d(nn.Module):
    def __init__(
        self,
        out_channel_count: int,
        in_channel_count: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 1,
        pre_blur: bool = False):
        super(ScaledConv2d, self).__init__()
        self.out_channel_count = out_channel_count
        self.in_channel_count = in_channel_count
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pre_blur = pre_blur

        self.kernel = nn.Parameter(torch.normal(0., 1., (self.out_channel_count, self.in_channel_count, self.kernel_size, self.kernel_size)))
        self.bias = nn.Parameter(torch.zeros(self.out_channel_count))
        self.scale = nn.Parameter(torch.tensor(1. / math.sqrt(self.kernel_size ** 2 * self.out_channel_count)), requires_grad=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        if self.pre_blur:
            y = blur(y)
        y = F.conv2d(y, self.kernel * self.scale, bias=self.bias, stride=self.stride, padding="same" if self.stride==1 else self.padding)
        return y
    
class ToRGB(nn.Module):
    def __init__(self, input_channel: int):
        super(ToRGB, self).__init__()
        self.to_rgb = ScaledConv2d(3, input_channel, kernel_size=1)
    
    def forward(self, x):
        return self.to_rgb(x)


class UpsampleConv2d(nn.Module):
    def __init__(
            self,
            out_channel_count: int,
            in_channel_count: int,
            kernel_size: int = 3):
        super(UpsampleConv2d, self).__init__()
        self.out_channel_count = out_channel_count
        self.in_channel_count = in_channel_count
        self.kernel_size = kernel_size
        self.kernel_shape = torch.tensor((self.out_channel_count, self.in_channel_count, self.kernel_size, kernel_size))
        self.stride = 2
        self.padding = 1
        self.output_padding = 1

        self.kernel = nn.Parameter(torch.normal(0, 1., (self.in_channel_count, self.out_channel_count, self.kernel_size, self.kernel_size)))
        self.bias = nn.Parameter(torch.zeros(self.out_channel_count))
        self.scale = nn.Parameter(torch.tensor(1. / torch.sqrt(torch.prod(self.kernel_shape) // self.out_channel_count)), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.conv_transpose2d(x, self.kernel * self.scale, bias=None, stride=self.stride, padding=self.padding, output_padding=self.output_padding)
        y = y + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        y = blur(y)
        return y
    

class ScaledDense(nn.Module):
    def __init__(self,
                 output_count: int,
                 input_size: int):
        super(ScaledDense, self).__init__()
        self.output_count = output_count
        self.input_count = input_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.kernel = nn.Parameter(torch.normal(0., 1., (self.input_count, self.output_count)))#.to(self.device)
        self.bias = nn.Parameter(torch.zeros(self.output_count))#.to(self.device)
        self.scale = nn.Parameter(torch.tensor(1. / math.sqrt(self.input_count)))#.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.matmul(x, self.kernel * self.scale)
        return y + self.bias


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class ToRGB(nn.Module):
    def __init__(self, input_channel: int):
        super(ToRGB, self).__init__()
        self.to_rgb =ScaledConv2d(3, input_channel, kernel_size=1)
    
    def forward(self, x):
        return self.to_rgb(x)


class Add(nn.Module):
    def __init__(self):
        super(Add, self).__init__()
    
    def forward(self, a, b):
        return a + b