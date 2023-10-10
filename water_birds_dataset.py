from wilds.datasets.waterbirds_dataset import WaterbirdsDataset
import torch
import config


class CustomizedWaterbirdsDataset(WaterbirdsDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        self.first_stage_model = None

    def __getitem__(self, idx):

        x, y, metadata = super().__getitem__(idx)
        x, y, c = x, y, metadata[0]
        w = 1
        if self.first_stage_model is None:
            return x, y, c
        else:
            if idx in self.cache:
                w = self.cache[idx]
            else:
                w = CustomizedWaterbirdsDataset.compute_afr_weights(x, y, config.gamma)
                self.cache[idx] = w

            return x, y, c, w

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
