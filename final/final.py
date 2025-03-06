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
from torchdyn.core import NeuralODE
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.utils.data as data
import torch.nn as nn
import pytorch_lightning as pl

# %%
device = torch.device("cuda:0")
torch.set_default_dtype(torch.float64)
torch.set_float32_matmul_precision("high")


# %%
def plot_l63(data, n, style="scatter"):
    if n > 0:
        data = data[:n, :]
    else:
        n = data.shape[0]
    x, y, z = data.T
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    if style == "scatter":
        ax.scatter(x, y, z, s=1)
    elif style == "line":
        ax.plot(x, y, z, lw=0.3)
    else:
        raise ValueError
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"L63, {n} points")
    plt.show()


# %%
def get_loader(
    train_file: str,
    test_file: str,
    plot: bool = False,
    name: str = "",
    n_points: int = 1000,
    lag=1,
):
    traintensor = np.load(train_file)
    test = np.load(test_file)
    print(f"raw data shapes -- train: {traintensor.shape}, test: {test.shape}")
    X = torch.Tensor(traintensor[:-lag, :])
    Y = torch.Tensor(traintensor[lag:, :])
    print(f"train shapes -- x: {X.shape}, y: {Y.shape}")
    if plot and name == "l63":
        plot_l63(traintensor, n=n_points)
        plot_l63(traintensor, n=n_points, style="line")
    traintensor = data.TensorDataset(X, Y)
    trainloader = data.DataLoader(
        traintensor, batch_size=len(X), shuffle=True, num_workers=0
    )
    return trainloader


# %%
def get_loader_l63(plot=False, n_points=1000):
    return get_loader(
        train_file="lorenz63_on0.05_train.npy",
        test_file="lorenz63_test.npy",
        plot=plot,
        n_points=n_points,
        name="l63",
    )


# %%
def get_loader_l96():
    return get_loader(
        train_file="lorenz96_on0.05_train.npy",
        test_file="lorenz96_test.npy",
    )


# %%
class Learner_l63(pl.LightningModule):
    def __init__(self, t_span: torch.Tensor, model: nn.Module, lr: float):
        super().__init__()
        self.model, self.t_span = model, t_span
        self.trainloader = get_loader_l63()
        self.starting_lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        t_eval, y_hat = self.model(x, self.t_span)
        y_hat = y_hat[-1]  # select last point of solution trajectory
        loss = nn.MSELoss()(y_hat, y)
        print(f"loss: {loss}, lr: {self.optimizers().param_groups[0]['lr']}")
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.starting_lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "loss"}

    def train_dataloader(self):
        return self.trainloader


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
    model = NeuralODE(f, sensitivity="adjoint", solver="dopri5")
    # dopri5 works well for some reason
    # tsitouras45 is default but it sucks
    return model


# %%
model = Learner_l63(model=get_model_l63(), t_span=torch.linspace(0, 1, 2), lr=1e-2)
trainer = pl.Trainer(max_epochs=50, accelerator="gpu", devices="auto")
trainer.fit(model)


# %%
def make_trajectory(model, start, n):
    x = torch.Tensor(start)
    with torch.no_grad():
        _, traj = model(x, torch.linspace(0, n, 2))
        return traj.numpy()


# %%
n_points = 5000
preds = make_trajectory(model, [1, 1, 1], n_points)
get_loader_l63(True, n_points=n_points)
plot_l63(preds, -1, "scatter")
plot_l63(preds, -1, "line")

# %%
