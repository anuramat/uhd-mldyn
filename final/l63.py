# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import utils
from torchdyn.core import NeuralODE
import pytorch_lightning as pl
import torch
import torch.nn as nn
import psd

# %%
device = torch.device("cuda:0")
torch.set_default_dtype(torch.float64)
torch.set_float32_matmul_precision("high")


# %%
def get_model_l63():
    layers = [
        nn.Linear(3, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 3),
    ]
    f = nn.Sequential(*layers)
    model = NeuralODE(f, sensitivity="adjoint")
    return model


# %%
model = utils.Learner(
    model=get_model_l63(),
    t_span=torch.linspace(0, 1, 2),
    lr=1e-2,
    loader=utils.get_loader_l63(),
)
trainer = pl.Trainer(max_epochs=100, accelerator="gpu", devices="auto")
trainer.fit(model)


# %%
n_timesteps = 5000
preds = utils.make_trajectory(model, [1, 1, 1], n_timesteps=n_timesteps)
real = utils.get_data_l63()
utils.plot_l63(real, title="data", style="scatter")
utils.plot_l63(real, title="data", style="line")
utils.plot_l63(preds, title="generated trajectory", style="scatter")
utils.plot_l63(preds, title="generated trajectory", style="line")

# %%
print(psd.power_spectrum_error(preds, real[: len(preds)]))
