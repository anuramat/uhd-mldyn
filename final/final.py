# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

from torchdyn.core import NeuralODE
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.utils.data as data
import torch.nn as nn
import pytorch_lightning as pl

device = torch.device("cuda:0")
torch.set_default_dtype(torch.float64)
torch.set_float32_matmul_precision("high")


def plot_l63(data, n, style="scatter"):
    x, y, z = data[:n, :].T
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


def get_loader(
    train_file: str,
    test_file: str,
    plot: bool = False,
    name: str = "",
):
    train = np.load(train_file)
    test = np.load(test_file)
    print(f"raw data shapes -- train: {train.shape}, test: {test.shape}")
    X = torch.Tensor(train[:-1, :])
    Y = torch.Tensor(train[1:, :])
    print(f"train shapes -- x: {X.shape}, y: {Y.shape}")
    if plot and name == "l63":
        plot_l63(train, n=1000)
        plot_l63(train, n=-1, style="line")
    train = data.TensorDataset(X, Y)
    trainloader = data.DataLoader(
        train, batch_size=len(X), shuffle=True, num_workers=-1
    )
    return trainloader


def get_loader_l63():
    return get_loader(
        train_file="lorenz63_on0.05_train.npy",
        test_file="lorenz63_test.npy",
    )


class Learner_l63(pl.LightningModule):
    def __init__(self, t_span: torch.Tensor, model: nn.Module):
        super().__init__()
        self.model, self.t_span = model, t_span
        self.trainloader = get_loader_l63()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        t_eval, y_hat = self.model(x, self.t_span)
        y_hat = y_hat[-1]  # select last point of solution trajectory
        loss = nn.MSELoss()(y_hat, y)
        print(loss)
        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=3e-4)

    def train_dataloader(self):
        return self.trainloader


def get_model_l63():
    layers = [
        nn.Linear(3, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 3),
    ]
    f = nn.Sequential(*layers)
    model = NeuralODE(f).to(device)
    t_span = torch.linspace(0, 1, 2)  # [0,1]
    return t_span, model


def train():
    learn = Learner_l63(*get_model_l63())
    # trainer = pl.Trainer(min_epochs=200, max_epochs=250)
    trainer = pl.Trainer(max_epochs=50)
    trainer.fit(learn)


def get_loader_l96():
    return get_loader(
        train_file="lorenz96_on0.05_train.npy",
        test_file="lorenz96_test.npy",
    )


def main():
    # train()

    path = ("lightning_logs/version_15/checkpoints/epoch=49-step=50.ckpt",)
    model = Learner_l63.load_from_checkpoint(path)
    model.eval().to("cpu")
    y_hat = model(torch.Tensor([1, 1, 1]))
    print(y_hat)


if __name__ == "__main__":
    main()
