import yaml
import os
import torch
from models import Generator, Discriminator
from train import training
from load_data import load_data
from writer import writer
CONFIG_PATH = "/shared/home/iseiot/12184741/GAN/GAN_2024/custom_3/"
config ={"dataset_dir": "/shared/home/iseiot/12184741/GAN/GAN_2024/dataset/",
         "num_epoch": 50,
         "batch_size": 64,
         "resolution": 128,
         "latent_size": 100,
         "writer_dir": "/shared/home/iseiot/12184741/GAN/GAN_2024/runs/"}

def load_config(config_name):
    with open(os.path.join(CONFIG_PATH, config_name)) as file:
        config = yaml.safe_load(file)
    return config


def main():
    # config = load_config("config.yaml")
    discriminator = Discriminator(config["resolution"])
    generator = Generator(config["resolution"], config["latent_size"])
    if torch.cuda.is_available():
        discriminator.cuda()
        generator.cuda()
    writer_ = writer(config["writer_dir"])
    real_image_loader = load_data(dataset_dir=config["dataset_dir"],
                                  batch_size=config["batch_size"])

    training(generator=generator,
             discriminator=discriminator,
             real_image_loader=real_image_loader,
             num_epoch=config["num_epoch"],
             batch_size=config["batch_size"],
             writer=writer_,
             device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))


if __name__ == "__main__":
    main()