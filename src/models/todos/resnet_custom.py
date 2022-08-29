from typing import Callable, List, Optional, Tuple

import torch
from torch import nn, Tensor


# Base layer types

class _Conv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        **kwargs,
    ) -> None:
      assert [_ > 0 for _ in (kernel_size, stride, dilation)]
      assert [_ not in kwargs for _ in ('padding', 'padding_mode')]
      
      padding = (kernel_size - 1) * dilation
      assert padding % 2 == 0, 'Half padding requires either kernel_size % 2 == 1 or dilation % 2 == 0'

      super().__init__(
          in_channels, out_channels, kernel_size,
          stride=stride,
          dilation=dilation,
          padding=padding // 2,
          padding_mode='zeros',
          **kwargs,
      )

class _MaxPool2d(nn.MaxPool2d):
    def __init__(
        self,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        dilation: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

class _Upsample(nn.Module):
    def __init__(
        self,
        factors: List[int],
        axes: List[int],
    ) -> None:
        super().__init__()

        assert len(factors) == len(axes)
        self.factors = factors
        self.axes = axes

    def forward(self, x: Tensor) -> Tensor:
        for factor, axis in zip(self.factors, self.axes):
            x = x.repeat_interleave(factor, dim=axis)
        return x

class _Upsample2d(_Upsample):
    def __init__(
        self,
        factor: int = 1,
    ) -> None:
        super().__init__([factor] * 2, [2, 3])

    def extra_repr(self):
        return f'factor={self.factors[0]}'


# Layers

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        act_layer: Optional[Callable[[], nn.Module]] = None,
        pool_layer: Optional[Callable[[], nn.Module]] = None,
        pre: bool = False,
        **kwargs,
    ) -> None:
        modules = list()
        conv = _Conv2d(in_channels, out_channels, kernel_size, stride=stride, **kwargs,)
        if norm_layer is not None:
            modules.append(norm_layer(in_channels if pre else out_channels))
        if act_layer is not None:
            modules.append(act_layer())
        
        if not pre:
            modules = [conv] + modules
        else:
            modules = modules + [conv]
        
        if pool_layer is not None:
            assert not pre
            modules.append(pool_layer())

        super().__init__(*modules)

class ConvStem(ConvLayer):    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        act_layer: Optional[Callable[[], nn.Module]] = None,
        pool_layer: Optional[Callable[[], nn.Module]] = None,
        **kwargs,
    ) -> None:
        if pool_layer is None:
            pool_layer = _MaxPool2d

        super().__init__(in_channels, out_channels, 7,
            stride=2,
            norm_layer=norm_layer,
            act_layer=act_layer,
            pool_layer=pool_layer,
            pre=False,
            **kwargs,
        )


# Blocks

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        expansion: int = 4,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        act_layer: Optional[Callable[[], nn.Module]] = None,
        pre: bool = False,
        shortcut: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if in_channels != channels * expansion or stride != 1 or shortcut:
            self.shortcut = ConvLayer(
                in_channels, channels * expansion, 1,
                stride=stride,
                norm_layer=norm_layer,
                act_layer=None if not pre else act_layer,
                pre=pre,
                **kwargs,
            )
        else:
            self.shortcut = nn.Identity()
        
        if not pre:
            self.act = act_layer()
        else:
            self.act = nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.block(x) + self.shortcut(x))
        

class BasicBlock(ResidualBlock):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        expansion: int = 4,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        act_layer: Optional[Callable[[], nn.Module]] = None,
        pre: bool = False,
        shortcut: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels, channels, stride, expansion,
            norm_layer=norm_layer,
            act_layer=act_layer,
            pre=pre,
            shortcut=shortcut,
        )

        block = list()
        block.append(ConvLayer(
            in_channels, channels, 3,
            stride=stride,
            norm_layer=norm_layer,
            act_layer=act_layer,
            pre=pre,
            **kwargs,
        ))
        block.append(ConvLayer(
            channels, channels * expansion, 3,
            stride=1,
            norm_layer=norm_layer,
            act_layer=None if not pre else act_layer,
            pre=pre,
            **kwargs,
        ))
        self.block = nn.Sequential(*block)


class BottleneckBlock(ResidualBlock):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        expansion: int = 4,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        act_layer: Optional[Callable[[], nn.Module]] = None,
        pre: bool = False,
        shortcut: bool = False,
        v1: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            in_channels, channels, stride, expansion,
            norm_layer=norm_layer,
            act_layer=act_layer,
            pre=pre,
            shortcut=shortcut,
        )

        block = list()
        block.append(ConvLayer(
            in_channels, channels, 1,
            stride=1 if not v1 else stride,
            norm_layer=norm_layer,
            act_layer=act_layer,
            pre=pre,
            **kwargs,
        ))
        block.append(ConvLayer(
            channels, channels, 3,
            stride=stride if not v1 else 1,
            norm_layer=norm_layer,
            act_layer=act_layer,
            pre=pre,
            **kwargs,
        ))
        block.append(ConvLayer(
            channels, channels * expansion, 1,
            stride=1,
            norm_layer=norm_layer,
            act_layer=None if not pre else act_layer,
            pre=pre,
            **kwargs,
        ))
        self.block = nn.Sequential(*block)


class ResidualLayer(nn.Sequential):
    def __init__(
        self,
        block: Optional[Callable[..., nn.Module]],
        size: int,
        in_channels: int,
        channels: int,
        stride: int = 1,
        expansion: int = 4,
        **kwargs,
    ) -> None:
        blocks = list()
        blocks.append(block(in_channels, channels, stride, expansion, **kwargs,))
        for _ in range(1, size):
            blocks.append(block(channels * expansion, channels, 1, expansion, **kwargs,))
        
        super().__init__(*blocks)



class ResnetFPN(nn.Module):
    def __init__(
        self,
        block: Optional[Callable[..., nn.Module]],
        sizes: List[int],
        in_channels: int,
        channels: List[int],
        strides: List[int],
        expansion: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()

        self.stem = ConvStem(in_channels, channels[0], **kwargs)

        self.module_list = nn.ModuleList()
        for size, channel, stride in zip(sizes, list(zip(channels[:1] + [_ * expansion for _ in channels[:-1]], channels)), strides):
            self.module_list.append(ResidualLayer(block, size, *channel, stride=stride, expansion=expansion, **kwargs,))

        self.inner_list = nn.ModuleList([_Conv2d(channel * expansion, channels[0] * expansion, 1) for channel in channels])
        self.upsample_list = nn.ModuleList([_Upsample2d(stride) for stride in strides])
        self.outer_list = nn.ModuleList([_Conv2d(channels[0] * expansion, channels[0] * expansion, 3) for _ in channels])
    
    def _forward(self, x: Tensor) -> Tuple[List[Tensor]]:
        features = list()
        inner_features = list()
        upsampled_features = list()
        outer_features = list()

        x = self.stem(x)

        for module in self.module_list:
            if len(features) == 0:
                features.append(module(x))
            else:
                features.append(module(features[-1]))
        
        for inner, upsample, outer, feature in list(zip(self.inner_list, self.upsample_list, self.outer_list, features))[::-1]:
            if len(inner_features) == 0:
                inner_features.append(inner(feature))
                upsampled_features.append(upsample(inner_features[-1]))
            else:
                inner_features.append(inner(feature) + upsampled_features[-1])
                upsampled_features.append(upsample(inner_features[-1]))
            outer_features.append(outer(inner_features[-1]))
        
        inner_features, upsampled_features, outer_features = [_[::-1] for _ in [inner_features, upsampled_features, outer_features]]
        
        return features, inner_features, upsampled_features, outer_features

    def forward(self, x: Tensor) -> List[Tensor]:
        return self._forward(x)[-1]


