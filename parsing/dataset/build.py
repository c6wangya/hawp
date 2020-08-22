import torch
from .transforms import *
from . import train_dataset, HRJ_dataset, HRJ_test_dataset, test_dataset, building_train_dataset, building_test_dataset
from parsing.config.paths_catalog import DatasetCatalog

def build_transform(cfg):
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                                           cfg.DATASETS.IMAGE.PIXEL_STD,
                                           cfg.DATASETS.IMAGE.TO_255)
        ]
    )
    return transforms
def build_train_dataset(cfg):
    assert len(cfg.DATASETS.TRAIN) == 1
    name = cfg.DATASETS.TRAIN[0]
    dargs = DatasetCatalog.get(name)

    factory = getattr(HRJ_dataset,dargs['factory'])
    args = dargs['args']
    args['transform'] = Compose(
                                [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                                        cfg.DATASETS.IMAGE.WIDTH),
                                 ToTensor(),
                                 Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                                           cfg.DATASETS.IMAGE.PIXEL_STD,
                                           cfg.DATASETS.IMAGE.TO_255)])
    args['transform_target'] = Compose(
                                [ToTensor()])
    dataset = factory(**args)
    
    dataset = torch.utils.data.DataLoader(dataset,
                                          batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                          collate_fn=HRJ_dataset.collate_fn,
                                          shuffle = True,
                                          num_workers = cfg.DATALOADER.NUM_WORKERS)
    return dataset

def build_test_dataset(cfg):
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                                           cfg.DATASETS.IMAGE.PIXEL_STD,
                                           cfg.DATASETS.IMAGE.TO_255)
        ]
    )

    datasets = []
    for name in cfg.DATASETS.TEST:
        dargs = DatasetCatalog.get(name)
        factory = getattr(test_dataset,dargs['factory'])
        args = dargs['args']
        args['transform'] = transforms
        dataset = factory(**args)
        dataset = torch.utils.data.DataLoader(
            dataset,  batch_size = 1,
            collate_fn = dataset.collate_fn,
            num_workers = cfg.DATALOADER.NUM_WORKERS,
        )
        datasets.append((name,dataset))
    return datasets

def build_hrj_test_dataset(cfg):
    transforms = Compose(
        [ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
         ToTensor(),
         Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                                           cfg.DATASETS.IMAGE.PIXEL_STD,
                                           cfg.DATASETS.IMAGE.TO_255)
        ]
    )
    transforms_target = Compose(
        [
            ResizeTarget(cfg.DATASETS.IMAGE.HEIGHT,
                     cfg.DATASETS.IMAGE.WIDTH),
            ToTensor()
        ]
    )

    datasets = []
    for name in cfg.DATASETS.TEST:
        dargs = DatasetCatalog.get(name)
        factory = HRJ_test_dataset.HRJTestDataset
        args = dargs['args']
        args['transform'] = transforms
        args['transform_target'] = transforms_target
        dataset = factory(**args)
        dataset = torch.utils.data.DataLoader(
            dataset,  batch_size = 1,
            collate_fn = HRJ_dataset.collate_fn,
            num_workers = cfg.DATALOADER.NUM_WORKERS,
        )
        datasets.append((name,dataset))
    return datasets

def build_building_train_dataset(cfg):
    args = {'root': './data/building_patches-raster-128/'}
    args['transform'] = Compose(
                                [
                                    ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                                        cfg.DATASETS.IMAGE.WIDTH),
                                    ToTensor(),
                                    Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                                            cfg.DATASETS.IMAGE.PIXEL_STD,
                                            cfg.DATASETS.IMAGE.TO_255)
                                ])
    args['transform_target'] = Compose(
                                [
                                    ResizeTarget(cfg.DATASETS.IMAGE.HEIGHT,
                                        cfg.DATASETS.IMAGE.WIDTH),
                                    ToTensor()
                                ])
    dataset = building_train_dataset.BuildingTrainDataset(**args)
    
    dataset = torch.utils.data.DataLoader(dataset,
                                          batch_size=cfg.SOLVER.IMS_PER_BATCH,
                                          collate_fn=building_train_dataset.collate_fn,
                                          shuffle = True,
                                          num_workers = cfg.DATALOADER.NUM_WORKERS)
    return dataset

def build_building_test_dataset(cfg):
    args = {'root': './data/building_patches-raster-128/'}
    args['transform'] = Compose(
                                [
                                    ResizeImage(cfg.DATASETS.IMAGE.HEIGHT,
                                        cfg.DATASETS.IMAGE.WIDTH),
                                    ToTensor(),
                                    Normalize(cfg.DATASETS.IMAGE.PIXEL_MEAN,
                                            cfg.DATASETS.IMAGE.PIXEL_STD,
                                            cfg.DATASETS.IMAGE.TO_255)
                                ])
    args['transform_target'] = Compose(
                                [
                                    ResizeTarget(cfg.DATASETS.IMAGE.HEIGHT,
                                        cfg.DATASETS.IMAGE.WIDTH),
                                    ToTensor()
                                ])

    dataset = building_test_dataset.BuildingTestDataset(**args)
    
    dataset = torch.utils.data.DataLoader(dataset,
                                          batch_size=1,
                                          collate_fn=building_test_dataset.collate_fn,
                                          num_workers = cfg.DATALOADER.NUM_WORKERS)
    return dataset