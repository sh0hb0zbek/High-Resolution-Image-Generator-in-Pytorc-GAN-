from layers import Add, ToRGB, Reshape, ImageConversion, ImageConversionMode, MinibatchStddev, PixelNorm, ScaledAdd, ScaledConv2d, ScaledDense, Upsample, Downsample, ScaledLeakyRelu, UpsampleConv2d
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple


def stat_str(x):
    mean = torch.mean(x)
    std = torch.std(x)
    examples = list(torch.reshape(x, [-1])[:10])
    examples = " ".join([f"{example:6.2f}" for example in examples])
    return f"{mean:6.2f} +/- {std:6.2f} [{examples}]"


def validate_resolution(resolution: int) -> None:
    assert resolution in [4, 8, 16, 32, 64, 128, 256, 512, 1024]


def activate(
        x,
        activation = lambda: ScaledLeakyRelu()):
    if activation is not None:
        x = activation()(x)
    return x


def upsample_conv_2d(
        out_channel_count: int,
        in_channel_count: int):
    return nn.Sequential(
        UpsampleConv2d(out_channel_count=out_channel_count,
                       in_channel_count=in_channel_count),
        ScaledLeakyRelu())


def conv_2d(
        out_channel_count: int,
        in_channel_count: int,
        kernel_size: int = 3,
        activation = ScaledLeakyRelu(),
        stride: int = 1,
        pre_blur: bool = False):
    conv_block = nn.Sequential(
        ScaledConv2d(out_channel_count=out_channel_count,
                        in_channel_count=in_channel_count,
                        kernel_size=kernel_size,
                        stride=stride,
                        pre_blur=pre_blur))
    if activation is not None:
        conv_block.add_module("activation", activation)
    return conv_block


class Generator(nn.Module):
    def __init__(self,
                 resolution: int,
                 latent_size):
        super(Generator, self).__init__()
        validate_resolution(resolution)

        self.resolution_to_channel_counts = {
            4  : 512,   8  : 512,   16  : 512,
            32 : 512,   64 : 512,   128 : 256,
            256: 128,   512:  64,   1024:  32  }
        self.resolution = resolution
        self.input_shape = [None, latent_size]
        
        self.module_1 = nn.Sequential(
            nn.Identity(latent_size),
            PixelNorm(),
            ScaledDense(output_count=512*4*4, input_size=latent_size),
            Reshape(-1, 512, 4, 4),
            ScaledLeakyRelu()
        )
        self.upsample = Upsample()
        self.add = Add()
        self.image_conversion = ImageConversion(ImageConversionMode.MODEL_TO_TORCH)

        self.generator_body()

    def generator_body(self):
        self.conv_modules = nn.ModuleDict()
        res = 4
        self.conv_modules.update({f"{res}_1": conv_2d(512, 512)})
        self.conv_modules.update({f"{res}_2": ToRGB(512)})
        while res < self.resolution:
            in_channel_count = self.resolution_to_channel_counts[res]
            out_channnel_count = self.resolution_to_channel_counts[res*2]
            
            res *= 2
            self.conv_modules.update({f"{res}_1": nn.Sequential(
                upsample_conv_2d(out_channel_count=out_channnel_count,
                                 in_channel_count=in_channel_count),
                conv_2d(out_channel_count=out_channnel_count,
                        in_channel_count=out_channnel_count))})
            self.conv_modules.update({f"{res}_2": ToRGB(out_channnel_count)})


    def forward(self, x):
        x = self.module_1(x)
        res = 4
        conv_out = self.conv_modules[f"{res}_1"](x)
        rgb_out = self.conv_modules[f"{res}_2"](conv_out)
        while res < self.resolution:
            res *= 2
            conv_out = self.conv_modules[f"{res}_1"](conv_out)
            rgb_out_prev = self.upsample(rgb_out)
            rgb_out = self.conv_modules[f"{res}_2"](conv_out)
            rgb_out = self.add(rgb_out_prev, rgb_out)
        rgb = self.image_conversion(rgb_out)
        return rgb


def downsample(out_channel_count, in_channel_count):
    downsample_block = nn.Sequential(Downsample())
    if out_channel_count != in_channel_count:
        downsample_block.add_module("downsample", conv_2d(out_channel_count=out_channel_count,
                                                          in_channel_count=in_channel_count,
                                                          kernel_size=1,
                                                          activation=None))
    return downsample_block
    

class Discriminator(nn.Module):
    def __init__(self, resolution):
        super(Discriminator, self).__init__()
        validate_resolution(resolution=resolution)
        self.resolution_to_feature_counts: Dict[int, Tuple[int, int]] = {
            1024: (32, 64),
            512: (64, 128),
            256: (128, 256),
            128: (256, 512),
            64: (512, 512),
            32: (512, 512),
            16: (512, 512),
            8: (512, 512),
            4: (512, 512)}
        self.resolution = resolution
        
        self.m_modules = nn.ModuleDict()
        self.m_modules.update({"input", nn.Identity(self.resolution, self.resolution, 3)})
        self.m_modules.update({"image_conversion", ImageConversion(ImageConversionMode.TORCH_TO_MODEL)})
        self.m_modules.update({f"from_rgb_{self.resolution}x{self.resolution}", conv_2d(out_channel_count=self.resolution_to_feature_counts[self.resolution][0],
                                                                                     in_channel_count=3,
                                                                                     kernel_size=1)})
        self.m_modules.update({"minibatch_stddev": MinibatchStddev()})
        self.m_modules.update({"conv_4x4_1": conv_2d(out_channel_count=512,
                                                   in_channel_count=513)})
        self.m_modules.update({"conv_4x4_2": conv_2d(out_channel_count=512,
                                                   in_channel_count=512,
                                                   kernel_size=4,
                                                   stride=4)})
        self.m_modules.update({"flatten": nn.Flatten()})
        self.m_modules.update({"to_classification": ScaledDense(output_count=1, input_size=512)})

        res = 4
        while res < self.resolution:
            res *= 2
            self.m_modules.update({f"conv_{res}x{res}_1": conv_2d(out_channel_count=self.resolution_to_feature_counts[res][0],
                                                                in_channel_count=self.resolution_to_feature_counts[res][0])})
            self.modm_modulesules.update({f"conv_{res}x{res}_2": conv_2d(out_channel_count=self.resolution_to_feature_counts[res][1],
                                                                in_channel_count=self.resolution_to_feature_counts[res][0],
                                                                stride=2)})
            self.m_modules.update({f"downsample_{res}_to_{res//2}": downsample(out_channel_count=self.resolution_to_feature_counts[res][1],
                                                                             in_channel_count=self.resolution_to_feature_counts[res][0])})
            self.m_modules.update({f"scaled_add_{res//2}": ScaledAdd()})
    
    def forward(self, x):
        input = self.m_modules["input"](x)
        image_conversion = self.m_modules["image_conversion"](input)
        from_rgb = self.modulm_moduleses[f"from_rgb_{self.resolution}x{self.resolution}"](image_conversion)
        x = from_rgb

        res = self.resolution
        while res > 4:
            conv_1 = self.m_modules[f"conv_{res}_{res}_1"](x)
            conv_2 = self.m_modules[f"conv_{res}_{res}_2"](conv_1)
            downsample = self.m_modules[f"downsample_{res}_to_{res//2}"](x)
            x = self.m_modules[f"scaled_add_{res//2}"](conv_2, downsample)
            res //= 2
        minibatch_stddev = self.m_modules["minibatch_stddev"](x)
        conv_4x4_1 = self.m_modules["conv_4x4_1"](minibatch_stddev)
        conv_4x4_2 = self.m_modules["conv_4x4_2"](conv_4x4_1)
        flatten = self.m_modules["flatten"](conv_4x4_2)
        to_classification = self.m_modules["to_classification"](flatten)
        return to_classification

