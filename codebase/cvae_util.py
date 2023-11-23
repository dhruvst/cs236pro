# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from codebase.baseline import MaskedMSELoss
# from codebase.goog import get_data
from codebase.stock import get_data
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import tqdm

from pyro.infer import Predictive, Trace_ELBO


def imshow(inp, image_path=None):
    # plot images
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    space = np.ones((inp.shape[0], 50, inp.shape[2]))
    inp = np.concatenate([space, inp], axis=1)

    ax = plt.axes(frameon=False, xticks=[], yticks=[])
    ax.text(0, 23, "Inputs:")
    ax.text(0, 23 + 28 + 3, "Truth:")
    ax.text(0, 23 + (28 + 3) * 2, "NN:")
    ax.text(0, 23 + (28 + 3) * 3, "CVAE:")
    ax.imshow(inp)

    if image_path is not None:
        Path(image_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(image_path, bbox_inches="tight", pad_inches=0.1)
    else:
        plt.show()

    plt.clf()


def visualize(
    device,
    num_quadrant_inputs,
    pre_trained_baseline,
    pre_trained_cvae,
    num_images,
    num_samples,
    image_path=None,
):
    # Load sample random data
    datasets, _, dataset_sizes = get_data(pos=num_quadrant_inputs, batch_size=num_images)
    dataloader = DataLoader(datasets["val"], batch_size=num_images, shuffle=False)

    batch = next(iter(dataloader))
    inputs = batch["input"].to(device)
    outputs = batch["output"].to(device)
    actual = batch["sixDays"].to(device)

    # Make predictions
    with torch.no_grad():
        baseline_preds = pre_trained_baseline(inputs).view(outputs.shape)

    predictive = Predictive(
        pre_trained_cvae.model, guide=pre_trained_cvae.guide, num_samples=num_samples
    )
    cvae_preds = predictive(inputs)["y"].view(num_samples, num_images, 6, 3)

    baseline_preds = torch.from_numpy(datasets['train'].reverseMinMax(baseline_preds.reshape(-1,3)))
    actual = torch.from_numpy(datasets['train'].reverseMinMax(actual.reshape(-1,3)))

    baseline_preds = baseline_preds.reshape(outputs.shape)
    actual = actual.reshape(outputs.shape)

    for i in range(cvae_preds.shape[0]):
        for j in range(cvae_preds.shape[1]):
            cvae_preds[i,j] = torch.from_numpy(datasets['train'].reverseMinMax(cvae_preds[i,j].reshape(-1,3))).squeeze()

    # Predictions are only made on the next day, so that is collected into
    # a single series.
    actual_final = []
    actual_final.append(actual[0,:5])
    actual_final.append(actual[:,5])
    actual_final = torch.cat(actual_final, dim=0)

    baseline_final = []
    baseline_final.append(actual[0,:5]) # The first five days are ignored.
    baseline_final.append(baseline_preds[:,5])
    baseline_final = torch.cat(baseline_final, dim=0)
    
    cvae_final = []
    cvae_final.append(actual[0,:5].unsqueeze(0))
    cvae_final.append(cvae_preds[:,:,5])
    cvae_final = torch.cat(cvae_final, dim=1)

    # Mitigates 'Tensor' object has no attribute 'ndim'
    torch.Tensor.ndim = property(lambda self: len(self.shape))
    
    t = range(baseline_final.shape[0])

    # breakpoint()

    fig, axs = plt.subplots(3)
    fig.suptitle('Don\'t know which one is which.')

    for i in range(actual_final.shape[-1]):
        axs[i].plot(t, actual_final[:,i])
        axs[i].plot(t, baseline_final[:,i])
        axs[i].plot(t, cvae_final[0,:,i]) # Assumes only one sample

    labels = ['actual', 'baseline', 'cvae']
    
    axs[2].legend(labels=labels)
    
    # Save plot
    fig.savefig(image_path)
    fig.clf()


def generate_table(
    device,
    num_quadrant_inputs,
    pre_trained_baseline,
    pre_trained_cvae,
    num_particles,
    col_name,
):
    # Load sample random data
    datasets, dataloaders, dataset_sizes = get_data(pos=num_quadrant_inputs, batch_size=32)

    # Load sample data
    criterion = MaskedMSELoss()
    loss_fn = Trace_ELBO(num_particles=num_particles).differentiable_loss

    baseline_cll = 0.0
    cvae_mc_cll = 0.0
    num_preds = 0

    df = pd.DataFrame(index=["NN (baseline)", "CVAE (Monte Carlo)"], columns=[col_name])

    # Iterate over data.
    bar = tqdm(dataloaders["val"], desc="Generating predictions".ljust(20))
    for batch in bar:
        inputs = batch["input"].to(device)
        outputs = batch["output"].to(device)
        num_preds += 1

        # Compute negative log likelihood for the baseline NN
        with torch.no_grad():
            preds = pre_trained_baseline(inputs)
        baseline_cll += criterion(preds, outputs).item() / inputs.size(0)

        # Compute the negative conditional log likelihood for the CVAE
        cvae_mc_cll += loss_fn(
            pre_trained_cvae.model, pre_trained_cvae.guide, inputs, outputs
        ).detach().item() / inputs.size(0)

    df.iloc[0, 0] = baseline_cll / num_preds
    df.iloc[1, 0] = cvae_mc_cll / num_preds
    return df
