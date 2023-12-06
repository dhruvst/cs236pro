import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, functional
from sklearn.preprocessing import StandardScaler

class STOCK(Dataset):
    def __init__(self, pd_df, input_size, n_cols, mask_size=5, train = True):
        self.ncols = n_cols
        self.original, self.test_low, self.test_high, self.minmax = self.loadData(pd_df, input_size, mask_size)
        self.transform =  Compose([ToTensor(), MaskData(pos=mask_size)])
        self.train = train
    def __len__(self):
        return len(self.original)

    def __getitem__(self, item):
        sample = {"original":self.original[item]}
        test_low = {"original":self.test_low[0]}
        test_high = {"original":self.test_high[0]}
        if self.transform:
            sample = self.transform(sample)
            test_low = self.transform(test_low)
            test_high = self.transform(test_high)
        if self.train:
            return sample
        else:
            return {'input_low': test_low['input'], 'input_high': test_high['input']}
    
    def loadData(self, df, input_size, mask_size):
    
        minmax = StandardScaler().fit(df[['close','high','low']].values.reshape(-1,3))
        df_norm = pd.DataFrame(minmax.transform(df[['close','high','low']].values).reshape(-1,3))
        
        df_norm.index = df.index
        df_norm.columns = df.columns

        dataset = []
        for i in range(len(df_norm) - (input_size+mask_size) - 1):
            dataset.append(df_norm.iloc[i:i + (input_size+mask_size),:])
        
        testset_low = [pd.concat([df_norm.iloc[-(input_size+mask_size-1) : -1],pd.DataFrame(
            [dict(zip(df_norm.columns,[0]*len(df_norm.columns)))]*2)])]
        
        testset_high = [pd.concat([df_norm.iloc[-input_size : ]
                        ,pd.DataFrame([dict(zip(df_norm.columns
                                                ,[0]*len(df_norm.columns)))]*mask_size)])]

        return dataset, testset_low, testset_high, minmax
    
    def reverseMinMax(self, pred):
        return self.minmax.inverse_transform(pred)

class ToTensor:
    def __call__(self, sample):
        sample['original'] = torch.from_numpy(sample['original'].values).float()
        return sample

class MaskData:
    """This transformation masks the values to be predicted with -1 and
    adds the target output in the sample dict as the complementary of the input
    """

    def __init__(self, pos=1, mask_with=-1):
        self.mask_with = mask_with
        self.pos = -pos

    def __call__(self, sample):
        tensor = sample['original']
        inp = tensor.detach().clone()

        # remove the last ones
        inp[self.pos:] = self.mask_with

        # now, sets the input as complementary
        out = tensor.clone()
        out[inp != -1] = self.mask_with

        sample["input"] = inp
        sample["output"] = out
        
        
        return sample
    
def get_data(df,input_size,ncols, batch_size=16):
    data, dl, ds = {}, {}, {}
    data['train'] = STOCK(df, input_size=input_size, n_cols=ncols)
    data['test'] = STOCK(df, input_size=input_size, n_cols=ncols, train = False)
    dl['train'] = DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=0)
    dl['val'] = DataLoader(data['train'], batch_size=batch_size, shuffle=False, num_workers=0)
    ds['train'] = ds['val'] = len(dl['train'])
    return data, dl, ds

def get_all_data(input_size, ncols, filepath):
    datasets, dataloaders, dataset_sizes = defaultdict(dict), defaultdict(dict), defaultdict(dict)
    d_all = pd.read_pickle(filepath)
    for ticker, dates in d_all.items():    
        for date,df in dates.items():
            prev_close = df['close'].shift(fill_value=0)
            df['high'] = df['high'] - prev_close
            df['low'] = df['low'] - prev_close
            df['close'] = df['close'] - prev_close
            df = df.iloc[1:]
            datasets[ticker][date], dataloaders[ticker][date], dataset_sizes[ticker][date] = get_data(df,input_size,ncols)
    return datasets, dataloaders, dataset_sizes
