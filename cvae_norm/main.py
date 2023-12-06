import torch
from torchvision.transforms import Compose, functional
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import dill
import pyro
from pyro.infer import Predictive

from collections import defaultdict
from datetime import datetime
import argparse

from models import train_baseline, train_cvae
from stock import get_all_data


def train_predict(ticker, date
                    ,datasets, dataloaders, dataset_sizes, device 
                     ,input_size, ncols, input_shape, early_stop_patience, num_epochs, learning_rate):
    
    mask_shape=[input_size,ncols]
    d_baseline, d_cvae = {}, {}
    dataset = datasets[ticker][date]
    dataloader_test = DataLoader(dataset["test"], batch_size=1, shuffle=False)
    batch = next(iter(dataloader_test))
    
    def get_pred(model, input_type, prelim_high_prediction=False):
        ### input_type: input_low or input_high
        prediction = model(batch[input_type].to(device))
        if type(prediction) == dict:
            prediction = prediction['y']
        prediction = prediction.reshape(-1,ncols).detach()
        
        prediction = torch.from_numpy(dataset['test'].reverseMinMax(prediction[:,:3]))
        index = {'input_low':2, 'input_high':1}
        if input_type == 'input_low' and prelim_high_prediction:
            return prediction[mask_shape[0], index['input_low']].item()\
                    , prediction[mask_shape[0]+mask_shape[1]-1, index['input_high']].item()
        return prediction[mask_shape[0], index[input_type]].item(),0
    
    baseline_net = train_baseline(device=device,
            dataloaders=dataloaders[ticker][date],
            dataset_sizes=dataset_sizes[ticker][date],
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            early_stop_patience=early_stop_patience,
            input_shape =  input_shape)

    d_baseline['low'], d_baseline['estimated_high'] = get_pred(baseline_net, 'input_low', prelim_high_prediction = True)
    d_baseline['high'] = get_pred(baseline_net, 'input_high')[0]
    
    cvae_net = train_cvae(device,
            dataloaders=dataloaders[ticker][date],
            dataset_sizes=dataset_sizes[ticker][date],
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            early_stop_patience=early_stop_patience,
            input_shape = input_shape,
            pre_trained_baseline_net=baseline_net,)
    predictive = Predictive(cvae_net.model, guide=cvae_net.guide, num_samples=1)

    d_cvae['low'], d_cvae['estimated_high'] = get_pred(predictive, 'input_low' , prelim_high_prediction = True)
    d_cvae['high'] = get_pred(predictive, 'input_high')[0]

    return d_baseline,d_cvae

def get_predictions(datasets, dataloaders, dataset_sizes, device
                    ,input_size, ncols,input_shape
                    ,filepath, nn_pred_path, cvae_pred_path
                   ,early_stop_patience, num_epochs,learning_rate):
    
    d_all = pd.read_pickle(filepath)
    d_baseline, d_cvae = defaultdict(dict), defaultdict(dict)
    d_base_temp, d_cvae_temp = {}, {}
    for ticker, dates in d_all.items():
        for date,df in dates.items():
            print(ticker,date)
            d_base_temp, d_cvae_temp = train_predict(ticker, date
                    ,datasets, dataloaders, dataset_sizes, device 
                            ,input_size, ncols, input_shape,early_stop_patience,num_epochs,learning_rate)
            d_base_temp['high'] = d_base_temp['high'] + df.iloc[-1]['close']
            d_cvae_temp['high'] = d_cvae_temp['high'] + df.iloc[-1]['close']
            d_cvae_temp['estimated_high'] = d_cvae_temp['estimated_high'] + df.iloc[-1]['close']
            d_base_temp['low'] = d_base_temp['low'] + df.iloc[-3]['close']
            d_cvae_temp['low'] = d_cvae_temp['low'] + df.iloc[-3]['close']
            d_baseline[ticker][date], d_cvae[ticker][date] = d_base_temp, d_cvae_temp
        dill.dump(d_baseline,open(nn_pred_path,'wb'))
        dill.dump(d_cvae,open(cvae_pred_path,'wb'))
    return d_baseline, d_cvae

def main(args):
    device = torch.device(
        "cuda:0" if torch.cuda.is_available() and args.cuda else "cpu")
    
    datasets, dataloaders, dataset_sizes = get_all_data(input_size = args.input_size
                                                       ,ncols = args.ncols
                                                        ,filepath = args.data_path)
    
    get_predictions(datasets, dataloaders, dataset_sizes, device = device
                    ,input_size = args.input_size,ncols = args.ncols,input_shape=args.input_shape
                    ,filepath = args.data_path,nn_pred_path=args.nn_predictions_path,cvae_pred_path=args.cvae_predictions_path
                   ,early_stop_patience=args.esp,num_epochs=args.n,learning_rate=args.lr)

if __name__ == "__main__":
    #assert pyro.__version__.startswith("1.8.6")
    # parse command line arguments
    parser = argparse.ArgumentParser(description="parse args")
    
    parser.add_argument(
        "--n",  default=101, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--esp", default=20, type=int, help="early stop patience"
    )
    parser.add_argument(
        "--lr", default=1.0e-3, type=float, help="learning rate"
    )
    parser.add_argument(
        "--cuda", action="store_true", default=False, help="whether to use cuda"
    )
    parser.add_argument(
        "--ncols", default=3, help="number of data columns"
    )
    parser.add_argument(
        "--input_shape", default=[10,3], help="[number of rows, number of colums] of input, including mask"
    )
    parser.add_argument(
        "--input_size", default=5, help="number of unmasked rows in input"
    )
    parser.add_argument(
        "--data_path", default='d_dfs.pkd', help="path to read the dataset"
    )
    parser.add_argument(
        "--nn_predictions_path", default='predictions_baseline_norm_input55.pkd', help="output path for baseline predictions"
    )
    parser.add_argument(
        "--cvae_predictions_path", default='predictions_cvae_norm_input55.pkd', help="output path for baseline predictions"
    )
    args = parser.parse_args()

    main(args)
