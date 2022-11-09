from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from collections import OrderedDict
from dataclasses import dataclass, field, asdict, is_dataclass, replace, InitVar

import torch
from torch import nn, Tensor

import torchvision


def replace_keys(
    self: Any,
    field: str,
    keys: List[str],
) -> object:
    assert is_dataclass(self)
    assert field not in keys
    assert is_dataclass(self.__getattribute__(field))
    self.__setattr__(field, replace(self.__getattribute__(field), **dict([(k, self.__getattribute__(k)) for k in keys])))


def load_submodule_params(
    src_dict: Mapping[str, Tensor], # state_dict which contains weights to be copied
    dest_dict: Mapping[str, Tensor], # state_dict into which weights are copied
    submodules: List[str], # list of fully qualified submodule names which whose weights will be copied
) -> nn.Module:
    submodules = [submodule.split('.') for submodule in submodules]
    is_submodule_param = lambda param_name: (lambda names: len([i for i in range(len(names)) if names[:i + 1] in submodules]) > 0)(param_name.split('.'))
    return OrderedDict(list(dest_dict.items()) + [item for item in src_dict.items() if is_submodule_param(item[0])])


@dataclass
class anchor_conf:
    size: int = 32
    scales: List[float] = field(default_factory=lambda: [2 ** i for i in range(0, 0 + 1)])
    ratios: List[float] = field(default_factory=lambda: [2 ** i for i in range(-1, 1 + 1)])
    levels: int = 5

    def module(self) -> nn.Module:
        return torchvision.models.detection.anchor_utils.AnchorGenerator(
            tuple([tuple([self.size * (2 ** level) * scale for scale in self.scales]) for level in range(self.levels)]),
            tuple([tuple(self.ratios) for _ in range(self.levels)])
        )


@dataclass
class rpn_conf:
    num_channels: int = 256

    anchor_generator: anchor_conf = field(default_factory=anchor_conf)

    conv_depth: int = 1
    fg_iou_thresh: float = 0.7
    bg_iou_thresh: float = 0.3
    batch_size_per_image: int = 256
    positive_fraction: float = 0.5
    pre_nms_top_n: Dict[str, int] = field(default_factory=lambda: dict(training=2000, testing=1000))
    post_nms_top_n: Dict[str, int] = field(default_factory=lambda: dict(training=2000, testing=1000))
    nms_thresh: float = 0.7
    score_thresh: float = 0.0

    def module(self) -> nn.Module:
        names = 'fg_iou_thresh bg_iou_thresh batch_size_per_image positive_fraction pre_nms_top_n post_nms_top_n nms_thresh score_thresh'.split()
        kwargs = dict([(k, v) for k, v in asdict(self).items() if k in names])
        return torchvision.models.detection.rpn.RegionProposalNetwork(
            anchor_generator=self.anchor_generator.module(),
            head=torchvision.models.detection.rpn.RPNHead(
                self.num_channels,
                len(self.anchor_generator.scales) * len(self.anchor_generator.ratios),
                conv_depth=self.conv_depth,
            ),
            **kwargs,
        )


@dataclass
class pooler_conf:
    featmap_names: List[str] = field(default_factory=lambda: list(map(str, range(4))))
    output_size: int = 7
    sampling_ratio: int = 2
    canonical_scale: int = 224
    canonical_level: int = 4

    def module(self) -> nn.Module:
        return torchvision.ops.poolers.MultiScaleRoIAlign(**asdict(self))


@dataclass
class box_head_conf:
    num_classes: int = 91
    num_channels: int = 256

    pooler: pooler_conf = field(default_factory=lambda: pooler_conf())
    conv_layers: Optional[List[int]] = None
    fc_layers: List[int] = field(default_factory=lambda: [1024])
    norm_layer: Optional[nn.Module] = None

    def module(self) -> nn.Module:
        return nn.Sequential(OrderedDict([
            ('box_roi_pool', self.pooler.module()),
            ('box_head', torchvision.models.detection.faster_rcnn.FastRCNNConvFCHead(
                input_size=(self.num_channels, self.pooler.output_size, self.pooler.output_size),
                conv_layers=self.conv_layers,
                fc_layers=self.fc_layers,
                norm_layer=self.norm_layer,
            ) if self.conv_layers is not None else torchvision.models.detection.faster_rcnn.TwoMLPHead(
                self.num_channels * self.pooler.output_size * self.pooler.output_size,
                self.fc_layers[-1],
            )),
            ('box_predictor', torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                self.fc_layers[-1],
                self.num_classes,
            )),
        ]))


@dataclass
class mask_head_conf:
    num_classes: int = 91
    num_channels: int = 256

    pooler: pooler_conf = field(default_factory=lambda: pooler_conf(output_size=14))
    layers: List[int] = field(default_factory=lambda: [256 for _ in range(4)])
    dilation: int = 1
    channels: int = 256
    norm_layer: Optional[nn.Module] = None

    def module(self) -> nn.Module:
        return nn.Sequential(OrderedDict([
            ('mask_roi_pool', self.pooler.module()),
            ('mask_head', torchvision.models.detection.mask_rcnn.MaskRCNNHeads(
                self.num_channels,
                self.layers,
                self.dilation,
                norm_layer=self.norm_layer,
            )),
            ('mask_predictor', torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
                self.layers[-1],
                self.channels,
                self.num_classes,
            )),
        ]))


@dataclass
class heads_conf:
    num_classes: int = 91
    num_channels: int = 256

    box_head: box_head_conf = field(default_factory=box_head_conf)
    mask_head: Optional[mask_head_conf] = field(default_factory=mask_head_conf)

    fg_iou_thresh: float = 0.5
    bg_iou_thresh: float = 0.5
    batch_size_per_image: int = 512
    positive_fraction: float = 0.25
    bbox_reg_weights: Tuple[float, float, float, float] = (10., 10., 5., 5.,)
    score_thresh: float = 0.05
    nms_thresh: float = 0.5
    detections_per_img: int = 100

    def __post_init__(self):
        keys = 'num_classes num_channels'.split()
        replace_keys(self, 'box_head', keys)
        if self.mask_head is not None:
            replace_keys(self, 'mask_head', keys)

    def module(self) -> nn.Module:
        names = 'fg_iou_thresh bg_iou_thresh batch_size_per_image positive_fraction bbox_reg_weights score_thresh nms_thresh detections_per_img'.split()
        kwargs = dict([(k, v) for k, v in asdict(self).items() if k in names])
        return torchvision.models.detection.roi_heads.RoIHeads(
            **dict(self.box_head.module().named_children()),
            **kwargs,
            **(dict(self.mask_head.module().named_children()) if self.mask_head is not None else dict()),
        )


@dataclass
class backbone_conf:
    num_channels: int = 256
    backbone_norm_layer: nn.Module = None
    fpn_norm_layer: nn.Module = None

    def module(self) -> nn.Module:
        backbone = torchvision.models.resnet.resnet50(norm_layer=self.backbone_norm_layer)
        layer_names = OrderedDict([(f'layer{i + 1}', f'{i}') for i in range(4)])
        in_channels_list = [backbone.get_submodule(name)[-1].conv3.out_channels for name in layer_names.keys()]
        extra_blocks = torchvision.ops.feature_pyramid_network.LastLevelMaxPool()
        return torchvision.models.detection.backbone_utils.BackboneWithFPN(
            backbone,
            layer_names,
            in_channels_list,
            self.num_channels,
            extra_blocks=extra_blocks,
            norm_layer=self.fpn_norm_layer,
        )


@dataclass
class rcnn_conf:
    num_classes: int = 91
    num_channels: int = 256
    pretrained: bool = False

    backbone: backbone_conf = field(default_factory=backbone_conf)
    rpn: rpn_conf = field(default_factory=rpn_conf)
    heads: heads_conf = field(default_factory=heads_conf)

    weights: object = torchvision.models.detection.mask_rcnn.MaskRCNN_ResNet50_FPN_Weights.COCO_V1

    def __post_init__(self):
        if self.pretrained:
            self.backbone.backbone_norm_layer = torchvision.ops.misc.FrozenBatchNorm2d

        keys = 'num_classes num_channels'.split()
        replace_keys(self, 'backbone', keys[1:])
        replace_keys(self, 'rpn', keys[1:])
        replace_keys(self, 'heads', keys)

    def _module(self) -> nn.Module:
        return nn.Sequential(OrderedDict([
            ('backbone', self.backbone.module()),
            ('rpn', self.rpn.module()),
            ('roi_heads', self.heads.module()),
        ]))

    def module(self, freeze_submodules=None, skip_submodules=None) -> nn.Module:
        model = self._module()
        if self.pretrained:
            state_dict = self.weights.get_state_dict(progress=True)
            if skip_submodules is not None:
                state_dict = load_submodule_params(model.state_dict(), state_dict, skip_submodules)
            model.load_state_dict(state_dict)
        if freeze_submodules is not None:
            for submodule in map(model.get_submodule, freeze_submodules):
                for param in submodule.parameters():
                    param.requires_grad_(False)
        return model


@dataclass
class rcnn_v2_conf(rcnn_conf):
    backbone: backbone_conf = field(default_factory=lambda: backbone_conf(fpn_norm_layer=nn.BatchNorm2d))
    rpn: rpn_conf = field(default_factory=lambda: rpn_conf(conv_depth=2))
    heads: heads_conf = field(default_factory=lambda: heads_conf(
        box_head=box_head_conf(
            conv_layers=[256 for _ in range(4)],
            norm_layer=nn.BatchNorm2d,
        ),
        mask_head=mask_head_conf(norm_layer=nn.BatchNorm2d),
    ))

    weights: object = torchvision.models.detection.mask_rcnn.MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1

    def __post_init__(self):
        super().__post_init__()
        self.backbone.backbone_norm_layer = None
