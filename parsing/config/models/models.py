from yacs.config import CfgNode as CN
from .shg import HGNETS
from .head import PARSING_HEAD

MODELS = CN()

MODELS.NAME = "Hourglass"
MODELS.HGNETS = HGNETS
MODELS.DEVICE = "cuda"
MODELS.WEIGHTS = ""
MODELS.HEAD_SIZE  = [[3], [1], [1], [2], [2]] 
MODELS.OUT_FEATURE_CHANNELS = 256

MODELS.LOSS_WEIGHTS = CN(new_allowed=True)

MODELS.PARSING_HEAD   = PARSING_HEAD

MODELS.CUDA = True
MODELS.ATTN = False
MODELS.ATTN_ONLY = False
MODELS.ATTN_DIM = ''
MODELS.ATTN_N_HEAD = 1

MODELS.SCALE = 1.0
