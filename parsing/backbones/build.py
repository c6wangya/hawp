from .registry import MODELS
from .stacked_hg import HourglassNet, Bottleneck2D
from .refinenet_deformable import RefineNetDeform
from .multi_task_head import MultitaskHead

@MODELS.register("Hourglass")
def build_hg(cfg):
    inplanes = cfg.MODEL.HGNETS.INPLANES
    num_feats = cfg.MODEL.OUT_FEATURE_CHANNELS//2
    depth = cfg.MODEL.HGNETS.DEPTH
    num_stacks = cfg.MODEL.HGNETS.NUM_STACKS
    num_blocks = cfg.MODEL.HGNETS.NUM_BLOCKS
    head_size = cfg.MODEL.HEAD_SIZE

    out_feature_channels = cfg.MODEL.OUT_FEATURE_CHANNELS


    num_class = sum(sum(head_size, []))
    model = HourglassNet(
        block=Bottleneck2D,
        inplanes = inplanes,
        num_feats= num_feats,
        depth=depth,
        head=lambda c_in, c_out: MultitaskHead(c_in, c_out, head_size=head_size),
        num_stacks = num_stacks,
        num_blocks = num_blocks,
        num_classes = num_class)

    model.out_feature_channels = out_feature_channels

    return model

@MODELS.register("Refinenet")
def build_rf(cfg):
    model = RefineNetDeform(
            2, 
            cuda=cfg.MODEL.CUDA, 
            attn=cfg.MODEL.ATTN, 
            attn_only=cfg.MODEL.ATTN_ONLY, 
            attn_dim=cfg.MODEL.ATTN_DIM, 
            n_head=cfg.MODEL.ATTN_N_HEAD, 
            use_contrastive=cfg.MODEL.ATTN_USE_CTL,
            share_weights=cfg.MODEL.ATTN_SHARE_W, 
            attn_bottleneck=cfg.MODEL.ATTN_BN
    )
    model.out_feature_channels = 2
    return model

def build_backbone(cfg, type=None):
    assert cfg.MODEL.NAME in MODELS,  \
        "cfg.MODELS.NAME: {} is not registered in registry".format(cfg.MODELS.NAME)
    if type is None:
        return MODELS[cfg.MODEL.NAME](cfg)
    elif type == 'Refinenet':
        return MODELS["Refinenet"](cfg)
    else:
        return MODELS["Hourglass"](cfg)
