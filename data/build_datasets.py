from torch.utils import data
from torch.utils.data import DataLoader

from .SemevalDataset import SemevalDataset
from yacs.config import CfgNode


def build_dataset(csv: str) -> SemevalDataset:
    return SemevalDataset(csv)


def build_data_loader(cfg: CfgNode, csv: str, is_train: bool = True) -> DataLoader:
    if is_train:
        batch_size = cfg.SOLVER.BATCH_SIZE
        shuffle = True
    else:
        batch_size = cfg.TEST.BATCH_SIZE
        shuffle = False

    datasets = build_dataset(csv)

    num_workers = cfg.DATALOADER.NUM_WORKERS

    data_loader = data.DataLoader(
        dataset=datasets,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers)

    return data_loader
