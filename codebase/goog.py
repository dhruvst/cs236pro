import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, functional
from sklearn.preprocessing import MinMaxScaler

class GOOG(Dataset):
    def __init__(self, param='High', train=True, transform=None):
        self.filePath = './vae_data/GOOG-year.csv'
        self.param = param
        self.original, self.minmax = self.loadData(self.param)
        self.transform = transform

    def __len__(self):
        return len(self.original)

    def __getitem__(self, item):
        sixDays = self.original[item]
        sample = {"sixDays":sixDays}
        if self.transform:
            sample = self.transform(sample)

        # breakpoint()

        return sample
    
    def loadData(self, param):
        df = pd.read_csv(self.filePath, index_col=0)
        df_param = df[param]

        # to do it in one shot.
        # df_norm = pd.DataFrame(MinMaxScaler().fit_transform(df))

        minmax = MinMaxScaler().fit(df_param.values.reshape(-1,1))
        df_norm = pd.DataFrame(minmax.transform(df_param.values.reshape(-1,1)))
        
        df_norm.index = df_param.index
        df_norm.columns = [param]

        dataset = []
        for i in range(len(df_norm) - 6):
            dataset.append(df_norm.iloc[i:i+6,:])

        print('GOOG dataset length: {}'.format(len(dataset)))
        return dataset, minmax
    
    def reverseMinMax(self, pred):
        return self.minmax.inverse_transform(pred)

class ToTensor:
    def __call__(self, sample):
        sample['sixDays'] = torch.from_numpy(sample['sixDays'].values).float()
        return sample

class MaskData:
    """This torchvision image transformation prepares the MNIST digits to be
    used in the tutorial. Depending on the number of quadrants to be used as
    inputs (1, 2, or 3), the transformation masks the remaining (3, 2, 1)
    quadrant(s) setting their pixels with -1. Additionally, the transformation
    adds the target output in the sample dict as the complementary of the input
    """

    def __init__(self, pos=1, mask_with=-1):
        self.mask_with = mask_with
        self.pos = -pos

    def __call__(self, sample):
        tensor = sample['sixDays']
        inp = tensor.detach().clone()

        # remove the last one
        inp[self.pos] = self.mask_with

        # now, sets the input as complementary
        out = tensor.clone()
        out[inp != -1] = self.mask_with

        sample["input"] = inp
        sample["output"] = out
        return sample

def get_data(pos, batch_size):
    transforms = Compose([ToTensor(), MaskData(pos=pos)])

    datasets, dataloaders, dataset_sizes = {}, {}, {}
    for mode in ["train", "val"]:
        datasets[mode] = GOOG(
            transform=transforms, train=mode == "train"
        )
        dataloaders[mode] = DataLoader(
            datasets[mode],
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=0,
        )
        dataset_sizes[mode] = len(datasets[mode])

    return datasets, dataloaders, dataset_sizes