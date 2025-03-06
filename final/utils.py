import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.utils.data as data


def get_loader(
    train_file: str,
    test_file: str,
    plot: bool = False,
    name: str = "",
    n_points=None,
    lag=1,
):
    traintensor = np.load(train_file)
    test = np.load(test_file)
    print(f"raw data shapes -- train: {traintensor.shape}, test: {test.shape}")
    X = torch.Tensor(traintensor[:-lag, :])
    Y = torch.Tensor(traintensor[lag:, :])
    print(f"train shapes -- x: {X.shape}, y: {Y.shape}")
    if plot and name == "l63":
        if n_points is None:
            raise ValueError
        plot_l63(data=traintensor, n=n_points, title="training data", style="scatter")
        plot_l63(data=traintensor, n=n_points, title="training data", style="line")
    traintensor = data.TensorDataset(X, Y)
    trainloader = data.DataLoader(
        traintensor, batch_size=len(X), shuffle=True, num_workers=0
    )
    return trainloader


def plot_l63(data, title=None, n=None, style=None):
    if n is None:
        n = data.shape[0]
    data = data[:n, :]

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

    if title is None:
        raise ValueError
    ax.set_title(f"{title}: Lorenz63, {data.shape[0]} points")

    plt.show()


def get_loader_l63(plot=False, n_points=None):
    return get_loader(
        train_file="lorenz63_on0.05_train.npy",
        test_file="lorenz63_test.npy",
        plot=plot,
        n_points=n_points,
        name="l63",
    )


def get_loader_l96():
    return get_loader(
        train_file="lorenz96_on0.05_train.npy",
        test_file="lorenz96_test.npy",
    )


def make_trajectory(model, start, n_timesteps):
    x = torch.Tensor(start)
    with torch.no_grad():
        _, traj = model.model(x, torch.range(0, n_timesteps - 1))
        return traj.numpy()


class Learner(pl.LightningModule):
    def __init__(self, t_span: torch.Tensor, model: nn.Module, lr: float, loader):
        super().__init__()
        self.model = model
        self.t_span = t_span
        self.trainloader = loader
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
