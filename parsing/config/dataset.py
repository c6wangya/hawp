from yacs.config import CfgNode as CN
# ---------------------------------------------------------------------------- #
# Dataset options
# ---------------------------------------------------------------------------- #
DATASETS = CN()
DATASETS.TRAIN = ("building_256_train",)
DATASETS.VAL   = ("building_256_test",)
DATASETS.TEST  = ("building_256_test",)
DATASETS.IMAGE = CN()
DATASETS.IMAGE.HEIGHT = 256
DATASETS.IMAGE.WIDTH  = 256

DATASETS.IMAGE.PIXEL_MEAN = [109.730, 103.832, 98.681]
DATASETS.IMAGE.PIXEL_STD  = [22.275, 22.124, 23.229]
DATASETS.IMAGE.TO_255 = True
DATASETS.TARGET = CN()
DATASETS.TARGET.HEIGHT= 128
DATASETS.TARGET.WIDTH = 128
DATASETS.DISTANCE_TH = 0.02
DATASETS.NUM_STATIC_POSITIVE_LINES = 300
DATASETS.NUM_STATIC_NEGATIVE_LINES = 40
#
