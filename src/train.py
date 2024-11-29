#!/bin/python3

from model import diffusion
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
import torch

if __name__ == 'main':
    training_iterations = 1000
    model = diffusion.DDPM(input_channels=6, image_size=64).to('cuda')
    num_epochs = 100
    dataloader = DataLoader(ImageFolder(os.environ.get("IMAGES_PATH")), shuffle=True)

    
    for epoch in tqdm(range(num_epochs), "Epochs"):
        epoch_loss = 0
        for batch in tqdm(dataloader, "Batch No."):
            epoch_loss += model(batch)
        print(f"Epoch loss was {epoch_loss/len(dataloader)}")
    print("Training Complete")

    torch.save(model, '/usr/data/weights.pth')

