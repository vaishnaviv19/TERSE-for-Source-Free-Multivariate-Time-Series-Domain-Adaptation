import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class LoadDataset(Dataset):
    def __init__(self, dataset, dataSetConfig):
        super.__init__()
        self.numChannels = dataSetConfig.input_channels
        x_data = dataset["samples"]

        if self.numChannels == 2:
            self.data = x_data.unsqueeze(1)
        elif self.numChannels == 3 and x_data.shape[1] != self.numChannels:
            self.data = x_data.transpose(0, 2, 1)

        if isinstance(x_data, np.ndarray):
            x_data = torch.from_numpy(x_data)

        y_data = dataset["labels"]

        if y_data is not None and isinstance(y_data, np.ndarray):
            self.labels = torch.from_numpy(y_data)
        
        if dataSetConfig.normalize:
            data_mean = torch.mean(x_data, dim=(0,2))
            data_std = torch.std(x_data, dim=(0,2))
            self.transform = transforms.Normalize(mean=data_mean, std=data_std)
        
        self.x_data = x_data.float()
        self.y_data = y_data.long() if y_data is not None else None
        self.len = x_data.shape[0]


    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.x_data[idx]
        if self.transform:
            x = self.transform(self.x_data[idx].reshape(self.numChannels, -1, 1)).reshape(self.x_data[idx].shape)
        y = self.y_data[idx] if self.y_data is not None else None

        return x, y, idx

def data_generator(file_path, domain, dtype, dataSetConfig, hparams):
    data_file = torch.load(os.path.join(file_path, f"{dtype}_{domain}.pt"))

    data_set = LoadDataset(data_file, dataSetConfig)

    if dtype == "test":
        suffle = False
        drop_last = False
    else:
        shuffle = dataSetConfig.shuffle
        drop_last = dataSetConfig.drop_last
    
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=hparams["batch_size"], shuffle=shuffle, drop_last=drop_last)
    return data_loader

