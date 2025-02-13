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
import mlflow.pytorch
import mlflow
from sklearn.preprocessing import StandardScaler
mlflow.set_tracking_uri("http://127.0.0.1:8080")

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
        self.triplet_margin: float = cfg.triplet_margin
        self.triplet_loss_weight: float = cfg.triplet_loss_weight
        self.similarity_distance: float = cfg.similarity_distance

    def forward(self, x: torch.Tensor) -> tuple:
        return self.vae(x)

    def triplet_loss(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float) -> torch.Tensor:
        pos_dist = torch.sum((anchor - positive).pow(2), dim=-1)
        neg_dist = torch.sum((anchor - negative).pow(2), dim=-1)
        loss = torch.relu(pos_dist - neg_dist + margin)
        return loss.mean()

    def get_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor, similarity_distance: float) -> tuple:
        # SIMPLE BOTTLENECK IMPLEMENTATION
        # anchor, positive, negative = [], [], []
        # for i in range(len(labels)):
        #     for j in range(len(labels)):
        #         if i != j and abs(labels[i] - labels[j]) < similarity_distance:
        #             anchor.append(embeddings[i])
        #             positive.append(embeddings[j])
        #         else:
        #             negative.append(embeddings[j])
        # min_len = min(len(anchor), len(positive), len(negative))
        # anchor = anchor[:min_len]
        # positive = positive[:min_len]
        # negative = negative[:min_len]
        # return torch.stack(anchor), torch.stack(positive), torch.stack(negative)

        # TORCH POWERED IMPLEMENTATION
        labels = labels.view(-1, 1)  # Reshape labels for broadcasting
        pairwise_diff = torch.abs(labels - labels.T)  # Compute pairwise absolute label differences

        mask_positive = (pairwise_diff < similarity_distance) & (pairwise_diff > 0)  # Exclude self-comparisons
        mask_negative = pairwise_diff >= similarity_distance

        indices = torch.arange(len(labels), device=embeddings.device)

        # Find indices where conditions are met
        anchor_idx, positive_idx = torch.where(mask_positive)
        _, negative_idx = torch.where(mask_negative)

        # If no valid triplets are found, return empty tensors
        if len(anchor_idx) == 0 or len(negative_idx) == 0:
            return (torch.empty(0, device=embeddings.device),
                    torch.empty(0, device=embeddings.device),
                    torch.empty(0, device=embeddings.device))

        # Ensure balanced triplets by truncating to the smallest set
        min_len = min(len(anchor_idx), len(negative_idx))
        anchor_idx, positive_idx, negative_idx = anchor_idx[:min_len], positive_idx[:min_len], negative_idx[:min_len]

        return embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
    
    def _step(self, batch: tuple, batch_idx: int, stage: str) -> torch.Tensor:
        x, y = batch
        x_hat, mean, logvar = self.vae(x)
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        total_loss = recon_loss + self.kl_loss_weight * kl_loss

        # Triplet loss
        anchor, positive, negative = self.get_triplets(mean, y, self.similarity_distance)
        triplet_loss = self.triplet_loss(anchor, positive, negative, self.triplet_margin)
        total_loss += self.triplet_loss_weight * triplet_loss

        values = {
            f"{stage}_loss": total_loss,
            f"{stage}_kl_loss": kl_loss,
            f"{stage}_recon_loss": recon_loss,
            f"{stage}_triplet_loss": triplet_loss,
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
    def __init__(self, cfg: DictConfig, after_vae: bool = False, vae_model: VAEModel = None):
        super().__init__()
        self.batch_size: int = cfg.batch_size
        self.train_split: float = cfg.train_split
        self.val_split: float = cfg.val_split
        self.data_path: str = cfg.data_path
        self.data_filename: str = cfg.data_filename
        self.after_vae: bool = after_vae
        if self.after_vae:
            assert vae_model is not None, "VAE model must be provided if after_vae is True"
            self.vae_model: VAEModel = vae_model
        self.scaler = StandardScaler()

    def preprocess_data(self):
        # Load data from file
        data = np.load(Path(self.data_path) / self.data_filename, allow_pickle=True)
        x = np.concatenate([d['signals'] for d in data])
        y = np.array([d['angle'] for d in data])
        
        # Scale the data
        x = self.scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        if self.after_vae:
            self.vae_model.eval()
            with torch.no_grad():
                x_tensor, _ = self.vae_model.vae.encode(torch.tensor(x, dtype=torch.float32).permute(0, 2, 1))
        else:
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

class EmbeddingToLabelModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super(EmbeddingToLabelModel, self).__init__()
        self.learning_rate: float = cfg.learning_rate
        self.layers: list = cfg.layers
        input_dim: int = cfg.input_dim

        layers = []
        for h_dim in self.layers:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.SiLU())
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def _step(self, batch: tuple, batch_idx: int, stage: str) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)
        loss = torch.sqrt(nn.functional.mse_loss(y_hat, y))
        values = {
            f"{stage}_mapping_loss": loss,
        }

        self.log_dict(values, on_epoch=True, prog_bar=True, logger=True)
        return loss

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

    def evaluate_performance(self, dataloader: DataLoader):
        self.eval()
        with torch.no_grad():
            for batch in dataloader:
                x, y = batch
                y_hat = self(x)
                for i in range(min(5, len(y))):  # Print first 5 examples
                    print(f"Input: {x[i]}")
                    print(f"Predicted: {y_hat[i].item()}, Actual: {y[i].item()}")

@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    mlflow.pytorch.autolog()

    # Should get the hydra runtime directory
    mlflow_uri = f"file:{Path(os.getcwd()).parent.parent.parent}/mlruns/"
    print(f"MLFlow URI: {mlflow_uri}")

    # Data Module for VAE
    data_module_vae = SeismicDataModule(cfg.vae_model)

    # Model
    vae_model = VAEModel(cfg.vae_model)

    mlflow_logger = MLFlowLogger(experiment_name="my_experiment", tracking_uri=mlflow_uri)

    # Training
    trainer_vae = Trainer(
        max_epochs=cfg.vae_model.num_epochs,
        logger=mlflow_logger,
        callbacks=[EarlyStopping(monitor="val_loss", patience=cfg.vae_model.patience, min_delta=cfg.vae_model.early_stopping_delta)]
    )
    trainer_vae.fit(vae_model, data_module_vae)
    trainer_vae.test(vae_model, data_module_vae)

    # Plot latent space
    plot_latent_space(vae_model, data_module_vae.val_dataloader(), cfg.vae_model.figures_dir, "latent_space.png")

    # Data Module for Embedding to Label Model
    data_module_embedding = SeismicDataModule(cfg.vae_model, after_vae=True, vae_model=vae_model)

    # Embedding to Label Model
    mapping_model = EmbeddingToLabelModel(cfg.mapping_model)
    trainer_mapping = Trainer(
        max_epochs=cfg.mapping_model.num_epochs,
        logger=mlflow_logger,
        callbacks=[EarlyStopping(monitor="val_mapping_loss", patience=cfg.mapping_model.patience, min_delta=cfg.mapping_model.early_stopping_delta)]
    )
    trainer_mapping.fit(mapping_model, data_module_embedding)
    trainer_mapping.test(mapping_model, data_module_embedding)

    # Evaluate performance
    mapping_model.evaluate_performance(data_module_embedding.test_dataloader())


if __name__ == "__main__":
    main()