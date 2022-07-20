import pdb

from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.ops.poolers import MultiScaleRoIAlign

from torchvision.models.detection import backbone_utils
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform

class _default_mrcnn_configs:
    def __init__(
        self,
        num_classes=91,
        backbone_num_features=5,
        backbone_out_channels=256,
    ):

        # self.num_classes = 91 if num_classes is None else num_classes
        # self.backbone_num_features = 5 if backbone_num_features is None else backbone_num_features
        # self.backbone_out_channels = 256 if backbone_out_channels is None else backbone_out_channels


        anchor_config = dict(
            sizes=[2 ** i for i in range(5, 5 + backbone_num_features)],
            # scales=[2 ** i for i in range(-1, 1 + 1)], # based on original paper variant
            scales=[2 ** i for i in range(0, 0 + 1)], # default torch implementation variant
            ratios=[2 ** i for i in range(-1, 1 + 1)],
        )
        rpn_head_config = dict(
            in_channels=backbone_out_channels,
            num_anchors=len(anchor_config['scales']) * len(anchor_config['ratios']), #
            # conv_depth=1, # option unsupported by some versions
        )
        rpn_config = dict(
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n=dict(
                training=2000,
                testing=1000,
            ),
            post_nms_top_n=dict(
                training=2000,
                testing=1000,
            ),
            nms_thresh=0.7,
            score_thresh=0.0,
        )
        box_roi_pool_config = dict(
            featmap_names=[str(i) for i in range(4)],
            output_size=7,
            sampling_ratio=2,
            canonical_scale=224, #
            canonical_level=4, #
        )
        box_head_config = dict(
            in_channels=backbone_out_channels * (box_roi_pool_config['output_size'] ** 2),
            representation_size=1024,
        )
        box_predictor_config = dict(
            in_channels=box_head_config['representation_size'],
            num_classes=num_classes,
        )
        mask_roi_pool_config = dict(
            featmap_names=box_roi_pool_config['featmap_names'],
            output_size=14,
            sampling_ratio=2,
            canonical_scale=box_roi_pool_config['canonical_scale'],
            canonical_level=box_roi_pool_config['canonical_level'],
        )
        mask_head_config = dict(
            in_channels=backbone_out_channels,
            layers=tuple([256 for _ in range(4)]),
            dilation=1,
        )
        mask_predictor_config = dict(
            in_channels=mask_head_config['layers'][-1],
            dim_reduced=256,
            num_classes=num_classes,
        )
        roi_heads_config = dict(
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=512,
            positive_fraction=0.25,
            bbox_reg_weights=None, # appears to use (10., 10., 5., 5.,) by default?
            score_thresh=0.05,
            nms_thresh=0.5,
            detections_per_img=100,
        )

        self.config_dict = dict(
            rpn_config=dict(
                anchor_config=anchor_config,
                rpn_head_config=rpn_head_config,
                rpn_config=rpn_config,
            ),
            roi_heads_config=dict(
                box_configs=dict(
                    box_roi_pool_config=box_roi_pool_config,
                    box_head_config=box_head_config,
                    box_predictor_config=box_predictor_config,
                ),
                mask_configs=dict(
                    mask_roi_pool_config=mask_roi_pool_config,
                    mask_head_config=mask_head_config,
                    mask_predictor_config=mask_predictor_config,
                ),
                roi_heads_config=roi_heads_config,
            ),
        )

def defaults(d, keys, list=False, prefix=None, suffix=None):
    if prefix is not None:
        keys = [f'{prefix}{k}' for k in keys]
    if suffix is not None:
        keys = [f'{k}{suffix}' for k in keys]

    assert set(d.keys()).issuperset(set(keys))
    values = [(k, d[k]) for k in keys]
    if list:
        return [k for _, k in values]
    return dict(values)


def build_anchor_generator(config):
    keywords = 'sizes scales ratios'
    sizes, scales, ratios = defaults(config, keywords.split(), list=True)

    sizes_t = tuple([tuple([size * scale for scale in scales]) for size in sizes])
    ratios_t = tuple([tuple(ratios) for _ in sizes])

    return AnchorGenerator(
        sizes=sizes_t,
        aspect_ratios=ratios_t,
    )

def build_rpn_head(config):
    keywords = 'in_channels num_anchors' # 'in_channels num_anchors conv_depth'
    config = defaults(config, keywords.split())

    return RPNHead(**config)

def build_rpn(anchor_config, rpn_head_config, rpn_config):
    keywords = 'fg_iou_thresh bg_iou_thresh batch_size_per_image positive_fraction pre_nms_top_n post_nms_top_n nms_thresh score_thresh'
    rpn_config = defaults(rpn_config, keywords.split())
    for k in 'pre post'.split(): # sub-config check to ensure pre, post nms configs properly structured, as expected by RegionProposalNetwork()
        k = f'{k}_nms_top_n'
        rpn_config[k] = defaults(rpn_config[k], 'training testing'.split())
        # rpn_config[k] = defaults(rpn_config[k], 'train test'.split())


    return RegionProposalNetwork(
        anchor_generator=build_anchor_generator(anchor_config),
        head=build_rpn_head(rpn_head_config),
        **rpn_config,
    )

def build_roi_pool(roi_pool_config):
    keywords = 'featmap_names output_size sampling_ratio canonical_scale canonical_level'
    roi_pool_config = defaults(roi_pool_config, keywords.split())

    return MultiScaleRoIAlign(**roi_pool_config)

def build_box_head(box_head_config):
    keywords = 'in_channels representation_size'
    box_head_config = defaults(box_head_config, keywords.split())

    return TwoMLPHead(**box_head_config)

def build_box_predictor(box_predictor_config):
    keywords = 'in_channels num_classes'
    box_predictor_config = defaults(box_predictor_config, keywords.split())

    return FastRCNNPredictor(**box_predictor_config)

def build_mask_head(mask_head_config):
    keywords = 'in_channels layers dilation'
    mask_head_config = defaults(mask_head_config, keywords.split())

    return MaskRCNNHeads(**mask_head_config)

def build_mask_predictor(mask_predictor_config):
    keywords = 'in_channels dim_reduced num_classes'
    mask_predictor_config = defaults(mask_predictor_config, keywords.split())

    return MaskRCNNPredictor(**mask_predictor_config)

def build_box_heads(box_roi_pool_config, box_head_config, box_predictor_config):
    return dict(
        box_roi_pool=build_roi_pool(box_roi_pool_config),
        box_head=build_box_head(box_head_config),
        box_predictor=build_box_predictor(box_predictor_config),
    )

def build_mask_heads(mask_roi_pool_config, mask_head_config, mask_predictor_config):
    return dict(
        mask_roi_pool=build_roi_pool(mask_roi_pool_config),
        mask_head=build_mask_head(mask_head_config),
        mask_predictor=build_mask_predictor(mask_predictor_config),
    )

def build_roi_heads(box_configs, mask_configs, roi_heads_config):
    roi_keywords = 'fg_iou_thresh bg_iou_thresh batch_size_per_image positive_fraction bbox_reg_weights score_thresh nms_thresh detections_per_img'
    head_keywords = lambda t: [f'{t}_{k}_config' for k in 'roi_pool head predictor'.split()]

    roi_heads_config = defaults(roi_heads_config, roi_keywords.split())

    box_configs = defaults(box_configs, 'roi_pool head predictor'.split(), prefix='box_', suffix='_config')
    box_heads = build_box_heads(**box_configs)

    mask_heads = dict()
    if mask_configs is not None:
        mask_configs = defaults(mask_configs, 'roi_pool head predictor'.split(), prefix='mask_', suffix='_config')
        mask_heads = build_mask_heads(**mask_configs)

    return RoIHeads(
        **box_heads,
        **roi_heads_config,
        **mask_heads,
    )


if __name__ == '__main__':
    configs = _default_mrcnn_configs().config_dict

    # pdb.set_trace()


    rpn_configs = defaults(configs['rpn_config'], 'anchor rpn_head rpn'.split(), suffix='_config')
    box_configs = defaults(configs['roi_heads_config']['box_configs'], 'roi_pool head predictor'.split(), prefix='box_', suffix='_config')
    mask_configs = defaults(configs['roi_heads_config']['mask_configs'], 'roi_pool head predictor'.split(), prefix='mask_', suffix='_config')
    roi_heads_config, = defaults(configs['roi_heads_config'], 'roi_heads'.split(), suffix='_config', list=True)

    # pdb.set_trace()

    rpn = build_rpn(**rpn_configs)
    roi_heads = build_roi_heads(box_configs, mask_configs, roi_heads_config)

    backbone = backbone_utils.resnet_fpn_backbone('resnet50', pretrained=False, trainable_layers=5)

    im_size = 1024
    transform = GeneralizedRCNNTransform(
        min_size=im_size,
        max_size=im_size,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
    )

    # TODO: test using pretrained weights to begin training, vs. starting from scratch i.e. pretrained=False
    mrcnn_model = GeneralizedRCNN(backbone, rpn, roi_heads, transform)
