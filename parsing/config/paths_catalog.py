import os
import os.path as osp

class DatasetCatalog(object):

    DATA_DIR = osp.abspath(osp.join(osp.dirname(__file__),
                '..','..','data'))
    
    DATASETS = {
        'wireframe_train': {
            'img_dir': 'wireframe/images',
            'ann_file': 'wireframe/train.json',
        },
        'road_train': {
            'img_dir': 'road/images',
            'ann_file': 'road/train.json',
        }, 
        'building_train': {
            'img_dir': 'building/images',
            'ann_file': 'building/train.json',
        }, 
        'building_256_train': {
            'img_dir': 'building_patches-256/images',
            'ann_file': 'building_patches-256/train.json',
        }, 
        'wireframe_test': {
            'img_dir': 'wireframe/images',
            'ann_file': 'wireframe/test.json',
        },
        'road_test': {
            'img_dir': 'road/images',
            'ann_file': 'road/test.json',
        }, 
        'building_test': {
            'img_dir': 'building/images',
            'ann_file': 'building/test.json',
        }, 
        'building_256_test': {
            'img_dir': 'building_patches-256/images',
            'ann_file': 'building_patches-256/test.json',
        },
        'york_test': {
            'img_dir': 'york/images',
            'ann_file': 'york/test.json',
        }
    }

    @staticmethod
    def get(name):
        assert name in DatasetCatalog.DATASETS
        data_dir = DatasetCatalog.DATA_DIR
        attrs = DatasetCatalog.DATASETS[name]

        args = dict(
            root = osp.join(data_dir,attrs['img_dir']),
            ann_file = osp.join(data_dir,attrs['ann_file'])
        )

        if 'train' in name:
            return dict(factory="TrainDataset",args=args)
        if 'test' in name and 'ann_file' in attrs:
            return dict(factory="TestDatasetWithAnnotations",
                        args=args)
        raise NotImplementedError()
