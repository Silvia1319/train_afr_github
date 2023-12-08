import torch
import torch.nn
from water_birds_module import WaterBirdsDataModule
from model_modified import ModelErm, ModelAfr
import config
import pytorch_lightning as pl
import os
from torchvision.utils import save_image


def save_test_images(loader, save_folder):
    os.makedirs(save_folder, exist_ok=True)

    for batch_idx, batch in enumerate(loader):
        images = batch[0]  # Assuming the first element in the batch is the image tensor
        for i in range(images.shape[0]):
            image = images[i].cpu()  # Ensure it's a PyTorch tensor, not a NumPy array
            save_path = os.path.join(save_folder, f'image_{batch_idx * images.shape[0] + i}.png')
            save_image(image, save_path)
def test_models():

    erm = ModelErm()
    erm.load_state_dict(torch.load('erm_weights.pth'))
    erm.eval()


    datamodule = WaterBirdsDataModule(name="Waterbirds", root_dir=config.root_dir,
                                      input_size=config.input_size,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers)

    first_trainer = pl.Trainer(accelerator=config.accelerator, devices=[0])
    first_trainer.test(erm, datamodule=datamodule)


    datamodule.change_to_2nd_stage(erm.model)


    afr = ModelAfr(erm.model)
    afr.load_state_dict(torch.load('afr_weights.pth'))
    afr.eval()

    second_trainer = pl.Trainer(accelerator=config.accelerator, devices=[0])
    second_trainer.test(afr, datamodule=datamodule)
    save_test_images(datamodule.test_dataloader(), 'afr_test_images')

if __name__ == "__main__":
    test_models()
