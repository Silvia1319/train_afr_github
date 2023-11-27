import torch
import pytorch_lightning as pl

from enum import Enum
from torch.utils.data import DataLoader, random_split
from wilds.common.data_loaders import get_eval_loader
from torchvision.transforms import transforms
from water_birds_dataset import CustomizedWaterbirdsDataset as WaterbirdsDataset
import config

class TrainingStage(Enum):
    ERM = 1
    REWEIGTING = 2


class WaterBirdsDataModule(pl.LightningDataModule):
    def __init__(self, name: str, root_dir: str, input_size: int, batch_size: int, num_workers: int, **kwargs):
        super().__init__()
        self._name = name
        self._root_dir = root_dir
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._stage = TrainingStage.ERM
        # https://github.com/kohpangwei/group_DRO/blob/cbbc1c5b06844e46b87e264326b56056d2a437d1/data/celebA_dataset.py#L81
        self._transform = transforms.Compose([
            transforms.CenterCrop(178),
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self._train_erm_data = None
        self._train_rw_data = None
        self._val_data = None
        self._test_data = None

    @staticmethod
    def compute_afr_weights(erm_logits, class_label, gamma):
        with torch.no_grad():
            p = erm_logits.softmax(-1)
        y_onehot = torch.zeros_like(erm_logits).scatter_(-1, class_label.unsqueeze(-1), 1)
        p_true = (p * y_onehot).sum(-1)
        weights = (-gamma * p_true).exp()
        n_classes = torch.unique(class_label).numel()
        class_count = []
        for y in range(n_classes):
            class_count.append((class_label == y).sum())
        for y in range(0, n_classes):
            weights[class_label == y] *= 1 / class_count[y]
        weights /= weights.sum()
        return weights

    def change_to_2nd_stage(self, model):
        self._stage = TrainingStage.REWEIGTING

        model.eval()
        model.cuda()
        logits = []
        ys = []
        for x, y, c in self.train_dataloader(shuffle=False):
            y_hat = model(x.to("cuda")).detach().cpu()
            logits.append(y_hat)
            ys.append(y)
        logits = torch.cat(logits)
        ys = torch.cat(ys)

        weights = self.compute_afr_weights(logits, ys, config.gamma)
        WaterbirdsDataset.weights = {self._train_rw_data.indices[i]: weights[i] for i in
                                     range(len(self._train_rw_data))}


    def setup(self, *args, **kwargs):
        dataset = WaterbirdsDataset(root_dir=self._root_dir, download=True)

        self._train_erm_data = dataset.get_subset("train", transform=self._transform)
        self._train_rw_data = dataset.get_subset("train_rw", transform=self._transform)
        self._val_data = dataset.get_subset("val", transform=self._transform)
        self._test_data = dataset.get_subset("test", transform=self._transform)

    def train_dataloader(self, shuffle=True) -> DataLoader:
        data = self._train_erm_data if self._stage == TrainingStage.ERM else self._train_rw_data

        train_loader = DataLoader(data,
                                  batch_size=self._batch_size,
                                  num_workers=self._num_workers,
                                  shuffle=shuffle)

        return train_loader

    def val_dataloader(self) -> DataLoader:
        val_loader = get_eval_loader("standard", self._val_data,
                                     batch_size=self._batch_size,
                                     num_workers=self._num_workers)

        return val_loader

    def test_dataloader(self) -> DataLoader:
        test_loader = get_eval_loader("standard", self._test_data,
                                      batch_size=self._batch_size,
                                      num_workers=self._num_workers)

        return test_loader
