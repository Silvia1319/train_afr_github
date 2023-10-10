import torch
import torch.nn
from water_birds_module import WaterBirdsDataModule
from model_modified import ModelErm, ModelAfr
import config
import pytorch_lightning as pl


def train():
    erm = ModelErm()
    datamodule = WaterBirdsDataModule(name="Waterbirds", root_dir=config.root_dir,
                                      input_size=config.input_size,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers)
    first_trainer = pl.Trainer(max_epochs=config.first_epoch, accelerator=config.accelerator)
    first_trainer.fit(erm, datamodule)

    datamodule.change_to_2nd_stage(erm.model)
    afr = ModelAfr(erm.model)
    second_trainer = pl.Trainer(max_epochs=config.second_epoch, accelerator=config.accelerator)
    second_trainer.fit(afr, datamodule)


if __name__ == "__main__":
    train()
