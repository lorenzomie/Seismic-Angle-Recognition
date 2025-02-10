import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig
from plot_tools import plot_latent_space

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # mean and logvar
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

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
        self.vae = VAE(cfg.input_dim, cfg.hidden_dim, cfg.latent_dim)
        self.learning_rate = cfg.learning_rate

    def forward(self, x):
        return self.vae(x)

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, mean, logvar = self.vae(x)
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        loss = recon_loss + kl_loss

        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('recon_loss', recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class SeismicDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.num_samples = cfg.num_samples
        self.seq_length = cfg.seq_length
        self.input_dim = cfg.input_dim
        self.batch_size = cfg.batch_size

    def prepare_data(self):
        # Generate dummy data
        x = torch.randn(self.num_samples, self.seq_length, self.input_dim)
        y = torch.zeros(self.num_samples)  # Dummy labels
        self.dataset = TensorDataset(x, y)

    def setup(self, stage=None):
        # Split dataset into train, val, test
        train_size = int(0.7 * len(self.dataset))
        val_size = int(0.2 * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Data Module
    data_module = SeismicDataModule(cfg)

    # Model
    model = VAEModel(cfg)

    # TensorBoard Logger
    logger = TensorBoardLogger("tb_logs", name="vae_model")

    # Training
    trainer = Trainer(max_epochs=cfg.num_epochs, logger=logger)
    trainer.fit(model, data_module)

    # Plot latent space
    plot_latent_space(model, data_module.val_dataloader())

if __name__ == "__main__":
    main()