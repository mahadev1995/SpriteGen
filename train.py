import os
import yaml
import torch
import random
import numpy as np
import pandas as pd
from dataset import *
import torch.nn as nn
from unet import UNet
from ddpm import DDPM
from torchvision import transforms as v2
from torchvision.utils import save_image

#read yaml file
with open('config.yaml') as file:
  config = yaml.safe_load(file)

# Setting reproducibility
SEED = config['seed']
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Create directories
os.makedirs(config['log_path'], exist_ok=True)
os.makedirs(config['ckpt_path'], exist_ok=True)
os.makedirs(config['img_path'], exist_ok=True)

# Data transforms
transform = v2.Compose([
                        v2.Lambda(lambda x: (x/255.)),
                        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                       ])

dataloader = create_dataloader(config, transform)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = UNet(in_ch=config['in_ch'], 
             out_ch=config['out_ch'], 
             time_dim=config['time_dim'],
             device=device)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
criterion = nn.MSELoss()
ddpm = DDPM(beta_start=config['beta_start'], 
            beta_end=config['beta_end'], 
            steps=config['steps'], 
            color_channels=config['color_channels'], 
            image_size=config['image_size'], 
            device=device)

loss_history = []

for epoch in range(config['num_epochs']):
    total_loss=0
    for step, data in enumerate(dataloader):
        data = data.to(device)

        optimizer.zero_grad()
        t = ddpm.time_step_sampler(data.shape[0]).to(device)
        x_t, e = ddpm.forward_diffusion(data, t)
        e_hat = model(x_t, t)
        loss = criterion(e, e_hat)

        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
    loss_history.append(total_loss/len(dataloader))

    sampled_images = ddpm.backward_diffusion(model, config['num_images_to_sample'])
    save_image(sampled_images, config['img_path']+ f'/sampled_image_epoch_{epoch}.png')

    print(f'Epoch: {epoch}, Loss: {round(loss.item(), 4)}')
    
    torch.save(model.state_dict(), config['ckpt_path'] + '/unet.pth')

df = pd.DataFrame(loss_history, columns=['Loss'])
df.to_csv(config['log_path'] + '/loss_history.csv')







