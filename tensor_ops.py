import torch
import torch.nn as nn
import torch.nn.functional as F


def create_blur_filter(dtype) -> torch.Tensor:
    return torch.tensor(
        [[0.015625, 0.046875, 0.046875, 0.015625],
         [0.046875, 0.140625, 0.140625, 0.046875],
         [0.046875, 0.140625, 0.140625, 0.046875],
         [0.015625, 0.046875, 0.046875, 0.015625]],
        dtype=dtype)


def blur(x: torch.Tensor, stride: int = 1) -> torch.Tensor:
    channel_count = x.shape[1]
    filter = create_blur_filter(x.dtype).unsqueeze(0).unsqueeze(0)
    filter = filter.repeat(channel_count, 1, 1, 1).to(x.device)
    return F.conv2d(x, filter, stride=stride, padding="same" if stride == 1 else 1, groups=channel_count)


def upsample(x: torch.Tensor) -> torch.Tensor:
    def upsample_with_zeros(x: torch.Tensor) -> torch.Tensor:
        in_height = x.shape[2]
        in_width = x.shape[3]
        channel_count = x.shape[1]

        out_height = in_height * 2
        out_width = in_width * 2
        x = x.view(-1, channel_count, in_height, 1, in_width, 1)
        x = F.pad(x, (0, 1, 0, 0, 0, 1))
        return x.view(-1, channel_count, out_height, out_width)

    return blur(upsample_with_zeros(x * 4.))


def downsample(x: torch.Tensor) -> torch.Tensor:
    return blur(x, stride=2)


def pixel_norm(x: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    original_dtype = x.dtype
    x = x.float()
    normalized = x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + epsilon)
    return normalized.to(original_dtype)


def lerp(start: torch.Tensor, end: torch.Tensor, factor: torch.Tensor) -> torch.Tensor:
    return start + (end - start) * factor


def reduce_std_nan_safe(x: torch.Tensor, axis=None, keepdim=False, epsilon=1e-7) -> torch.Tensor:
    y = x.float()
    mean = y.mean(axis=axis, keepdim=True)
    variance = ((y - mean) ** 2).mean(axis=axis, keepdim=keepdim)
    sqrt = torch.sqrt(variance + epsilon)
    return sqrt.to(x.dtype)


def minibatch_stddev(x, group_size=4):
    original_shape = x.shape
    original_dtype = x.dtype
    global_sample_count = original_shape[0]
    group_size = min(group_size, global_sample_count)
    group_count = global_sample_count // group_size
    assert group_size * group_count == global_sample_count, 'Sample count was not divisible by group size'
    # Shape definitions:
    # N = global sample count
    # G = group count
    # M = sample count within group
    # H = height
    # W = width
    # C = channel count
    # [NHWC] Input shape
    y = x.reshape(-1, group_size, *original_shape[1:])  # [GMHWC] Split into groups
    y = y.float()
    stddevs = reduce_std_nan_safe(y, axis=1, keepdim=True)  # [G1HWC]
    avg = stddevs.mean(dim=list(range(1, stddevs.ndim)), keepdim=True)  # [G1111]
    new_feature_shape = (*y.shape[:2], 1, *y.shape[3:])  # [GMHW1]
    new_feature = avg.expand(new_feature_shape)
    y = torch.cat([y, new_feature], dim=2)
    y = y.reshape(-1, *y.shape[2:])
    y = y.to(original_dtype)
    return y