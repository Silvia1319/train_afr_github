import copy
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from torchvision.transforms import transforms
import torch
from water_birds_dataset import CustomizedWaterbirdsDataset as WaterbirdsDataset


class WaterBirdsDataModule(pl.LightningDataModule):
    def __init__(self, name: str, root_dir: str, input_size: int, batch_size: int, num_workers: int, **kwargs):
        super().__init__()
        self._name = name
        self._root_dir = root_dir
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._train = None
        self._validation = None
        self._test = None
        self._d_erm = None
        self._d_rw = None
        self._stage = 1
        self._model=None
        # https://github.com/kohpangwei/group_DRO/blob/cbbc1c5b06844e46b87e264326b56056d2a437d1/data/celebA_dataset.py#L81
        self._transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._dataset = None

    def change_to_2nd_stage(self, model):
        self._stage = 2
        # self._model = model
        WaterbirdsDataset.first_stage_model = copy.deepcopy(model)

    def prepare_data(self):
        self._dataset = WaterbirdsDataset(root_dir=self._root_dir, download=True)

    def train_dataloader(self) -> DataLoader:

        _train_data = self._dataset.get_subset("train", transform=self._transform)
        d_erm_size = int(len(_train_data) * 0.8)
        d_rw_size = len(_train_data) - d_erm_size
        _d_erm, _d_rw = random_split(_train_data, [d_erm_size, d_rw_size])
        #        train_loader = get_train_loader("standard", self._d_erm,
        #                                        batch_size=self._batch_size,
        #                                         num_workers=self._num_workers)
        if self._stage == 1:
            train_loader = DataLoader(_d_erm,
                                      batch_size=self._batch_size,
                                      num_workers=self._num_workers,
                                      shuffle=True)
        else:
            train_loader = DataLoader(_d_rw,
                                      batch_size=self._batch_size,
                                      num_workers=self._num_workers,
                                      shuffle=True)
        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_data = self._dataset.get_subset("val", transform=self._transform)
        val_loader = get_eval_loader("standard", val_data,
                                     batch_size=self._batch_size,
                                     num_workers=self._num_workers)

        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_data = self._dataset.get_subset("test", transform=self._transform)
        test_loader = get_eval_loader("standard", test_data,
                                      batch_size=self._batch_size,
                                      num_workers=self._num_workers)

        return test_loader

    @property
    def dataset(self):
        return self._dataset

    @property
    def train_dataset(self):
        train_data = self._dataset.get_subset("train", transform=self._transform)

        return train_data

    @property
    def val_dataset(self):
        train_data = self._dataset.get_subset("val", transform=self._transform)

        return train_data

    @property
    def test_dataset(self):
        train_data = self._dataset.get_subset("test", transform=self._transform)

        return train_data
