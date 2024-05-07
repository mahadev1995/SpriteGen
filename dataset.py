import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class SpritesDataset(Dataset):
    def __init__(self, sprites, transform=None):
        super().__init__()
        self.sprites = sprites
        self.transform = transform

    def __len__(self):
        return self.sprites.shape[0]
    
    def __getitem__(self, idx):
        sample = self.sprites[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample

def create_dataloader(config, transform=None):
    data = np.load(config['data_path'])
    data = torch.tensor(data).permute(0, 3, 1, 2)

    dataset = SpritesDataset(data, transform)

    dataloader = DataLoader(dataset, 
                        batch_size=config['batch_size'], 
                        shuffle=config['shuffle'], 
                        drop_last=config['drop_last'],
                        ) 
    return dataloader
