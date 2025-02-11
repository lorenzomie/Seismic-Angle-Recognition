import numpy as np
import os
import math
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import EarlyStopping
import hydra
from omegaconf import DictConfig
from plot_tools import plot_latent_space
from pathlib import Path
from sklearn.preprocessing import StandardScaler

class VAE(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(VAE, self).__init__()
        self.input_channels: int = cfg.input_channels
        self.encoder_layers: list = cfg.encoder_layers
        self.latent_dim: int = cfg.latent_dim
        self.decoder_layers: list = cfg.decoder_layers
        self.kernel_size: int = cfg.kernel_size
        self.stride: int = cfg.stride
        self.padding: int = cfg.padding
        self.output_padding: int = cfg.output_padding
        self.input_length: int = cfg.input_length

        self.final_feature_size = self.compute_feature_size(self.input_length)

        self.encoder = self.build_encoder(self.input_channels, self.encoder_layers, self.latent_dim * 2)
        self.decoder = self.build_decoder(self.latent_dim, self.decoder_layers, self.input_channels)

    def compute_feature_size(self, input_length: int) -> int:
        """Computes the final length of the feature map after Conv1d layers."""
        for _ in self.encoder_layers:
            input_length = math.floor((input_length + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
        return input_length
    
    def build_encoder(self, input_channels: int, layers: list, output_dim: int) -> nn.Sequential:
        network = []
        for layer_dim in layers:
            network.append(nn.Conv1d(input_channels, layer_dim, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
            network.append(nn.SiLU())
            input_channels = layer_dim
        network.append(nn.Flatten())
        network.append(nn.Linear(layers[-1] * self.final_feature_size, output_dim))  # Adjust the final feature map size as needed
        return nn.Sequential(*network)

    def build_decoder(self, latent_dim: int, layers: list, output_channels: int) -> nn.Sequential:
        network = []
        network.append(nn.Linear(latent_dim, layers[0] * self.final_feature_size))
        network.append(nn.Unflatten(1, (layers[0], self.final_feature_size)))
        for i in range(len(layers) - 1):
            network.append(nn.ConvTranspose1d(layers[i], layers[i + 1], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=self.output_padding))
            network.append(nn.SiLU())
        network.append(nn.ConvTranspose1d(layers[-1], output_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=self.output_padding))
        return nn.Sequential(*network)

    def encode(self, x: torch.Tensor) -> tuple:
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

class VAEModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(VAEModel, self).__init__()
        self.vae = VAE(cfg)
        self.learning_rate: float = cfg.learning_rate
        self.kl_loss_weight: float = cfg.kl_loss_weight

    def forward(self, x: torch.Tensor) -> tuple:
        return self.vae(x)

    def _step(self, batch: tuple, batch_idx: int, stage: str) -> torch.Tensor:
        x, _ = batch
        x_hat, mean, logvar = self.vae(x)
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        total_loss = recon_loss + self.kl_loss_weight * kl_loss

        values = {
            f"{stage}_loss": total_loss,
            f"{stage}_kl_loss": kl_loss,
            f"{stage}_recon_loss": recon_loss,
        }

        self.log_dict(values, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def training_step(self, train_batch: tuple, batch_idx: int) -> torch.Tensor:
        """Override for training_step"""
        return self._step(train_batch, batch_idx, "train")

    def validation_step(self, val_batch: tuple, batch_idx: int) -> torch.Tensor:
        """Override for validation_step"""
        return self._step(val_batch, batch_idx, "val")

    def test_step(self, test_batch: tuple, batch_idx: int) -> torch.Tensor:
        """Override for test_step"""
        return self._step(test_batch, batch_idx, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class SeismicDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.batch_size: int = cfg.batch_size
        self.train_split: float = cfg.train_split
        self.val_split: float = cfg.val_split
        self.data_path: str = cfg.data_path
        self.data_filename: str = cfg.data_filename
        self.scaler = StandardScaler()

    def preprocess_data(self):
        # Load data from file
        data = np.load(Path(self.data_path) / self.data_filename, allow_pickle=True)
        x = np.concatenate([d['signals'] for d in data])
        y = np.array([d['angle'] for d in data])
        
        # Scale the data
        x = self.scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        
        # Convert to tensor and permute (len, seq_len, channels) -> (len, channels, seq_len)
        x_tensor = torch.tensor(x, dtype=torch.float32).permute(0, 2, 1)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        self.dataset = TensorDataset(x_tensor, y_tensor)

    def setup(self, stage: str = None):
        # Split dataset into train, val, test
        self.preprocess_data()
        train_size = int(self.train_split * len(self.dataset))
        val_size = int(self.val_split * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    # Should get the hydra runtime directory
    mlflow_uri = f"file:{os.getcwd()}/mlruns/"
    os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri
    print(f"Path of the current run: {mlflow_uri}")

    # Data Module
    data_module = SeismicDataModule(cfg.vae_model)

    # Model
    model = VAEModel(cfg.vae_model)

    mlflow_logger = MLFlowLogger(experiment_name="my_experiment")

    # Training
    trainer = Trainer(
        max_epochs=cfg.vae_model.num_epochs,
        logger=mlflow_logger,
        callbacks=[EarlyStopping(monitor="val_loss", patience=cfg.vae_model.patience, min_delta=cfg.vae_model.early_stopping_delta)]
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)

    # Plot latent space
    plot_latent_space(model, data_module.val_dataloader())

if __name__ == "__main__":
    main()