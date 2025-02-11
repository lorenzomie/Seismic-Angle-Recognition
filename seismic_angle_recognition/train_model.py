import numpy as np
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
    def __init__(self, input_dim, encoder_layers, latent_dim, decoder_layers):
        super(VAE, self).__init__()
        self.encoder = self.build_network(input_dim, encoder_layers, latent_dim * 2)
        self.decoder = self.build_network(latent_dim, decoder_layers, input_dim)

    def build_network(self, input_dim, layers, output_dim):
        network = []
        for layer_dim in layers:
            network.append(nn.Linear(input_dim, layer_dim))
            network.append(nn.SiLU())
            input_dim = layer_dim
        network.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*network)

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

class VAEModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(VAEModel, self).__init__()
        self.vae = VAE(cfg.input_dim, cfg.encoder_layers, cfg.latent_dim, cfg.decoder_layers)
        self.learning_rate = cfg.learning_rate
        self.kl_loss_weight = cfg.kl_loss_weight

    def forward(self, x):
        return self.vae(x)

    def _step(self, batch, batch_idx, stage):
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

    def training_step(self, train_batch, batch_idx):
        """Override for training_step"""
        return self._step(train_batch, batch_idx, "train")

    def validation_step(self, val_batch, batch_idx):
        """Override for validation_step"""
        return self._step(val_batch, batch_idx, "val")

    def test_step(self, test_batch, batch_idx):
        """Override for step_step"""
        return self._step(test_batch, batch_idx, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class SeismicDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.seq_length = cfg.seq_length
        self.input_dim = cfg.input_dim
        self.batch_size = cfg.batch_size
        self.train_split = cfg.train_split
        self.val_split = cfg.val_split
        self.data_path = cfg.data_path
        self.data_filename = cfg.data_filename
        self.scaler = StandardScaler()

    def prepare_data(self):
        # Load data from file
        data = np.load(Path(self.data_path) / self.data_filename, allow_pickle=True)
        x = np.concatenate([d['signals'] for d in data])
        y = np.array([d['angle'] for d in data])
        
        # Scale the data
        x = self.scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        
        self.dataset = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))

    def setup(self, stage=None):
        # Split dataset into train, val, test
        self.prepare_data()
        train_size = int(self.train_split * len(self.dataset))
        val_size = int(self.val_split * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    # Data Module
    data_module = SeismicDataModule(cfg.vae_model)

    # Model
    model = VAEModel(cfg.vae_model)

    mlflow_logger = MLFlowLogger(experiment_name="my_experiment", tracking_uri="file:./mlruns")

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