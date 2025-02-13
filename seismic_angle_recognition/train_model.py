import math
import os
from pathlib import Path

import hydra
import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from plot_tools import plot_latent_space

mlflow.set_tracking_uri("http://127.0.0.1:8080")

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model for embedding 3D signal values into a compact latent space representation.
    This model is used to encode the input signals into a lower-dimensional latent space, which captures the essential features of the data.
    The latent space representation can then be used for various downstream tasks, such as classification or clustering.
    """
    def __init__(self, cfg: DictConfig):
        """
        Initialize the VAE model.

        Args:
            cfg (DictConfig): Configuration dictionary containing hyperparameters.
                - input_channels (int): Number of input channels.
                - encoder_layers (list): List of encoder layer dimensions.
                - latent_dim (int): Dimensionality of the latent space.
                - decoder_layers (list): List of decoder layer dimensions.
                - kernel_size (int): Size of the convolutional kernel.
                - stride (int): Stride of the convolution.
                - padding (int): Padding for the convolution.
                - output_padding (int): Output padding for the transposed convolution.
                - input_length (int): Length of the input sequence.
        """
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
        """
        Compute the final length of the feature map after Conv1d layers.
        """
        for _ in self.encoder_layers:
            input_length = math.floor((input_length + 2 * self.padding - (self.kernel_size - 1) - 1) / self.stride + 1)
        return input_length
    
    def build_encoder(self, input_channels: int, layers: list, output_dim: int) -> nn.Sequential:
        """
        Build the encoder network.
        """
        network = []
        for layer_dim in layers:
            network.append(nn.Conv1d(input_channels, layer_dim, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding))
            network.append(nn.SiLU())
            input_channels = layer_dim
        network.append(nn.Flatten())
        network.append(nn.Linear(layers[-1] * self.final_feature_size, output_dim))
        return nn.Sequential(*network)

    def build_decoder(self, latent_dim: int, layers: list, output_channels: int) -> nn.Sequential:
        """
        Build the decoder network.
        """
        network = []
        network.append(nn.Linear(latent_dim, layers[0] * self.final_feature_size))
        network.append(nn.Unflatten(1, (layers[0], self.final_feature_size)))
        for i in range(len(layers) - 1):
            network.append(nn.ConvTranspose1d(layers[i], layers[i + 1], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=self.output_padding))
            network.append(nn.SiLU())
        network.append(nn.ConvTranspose1d(layers[-1], output_channels, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, output_padding=self.output_padding))
        return nn.Sequential(*network)

    def encode(self, x: torch.Tensor) -> tuple:
        """
        Encode the input tensor into mean and log variance.
        """
        if next(self.parameters()).device != x.device:
            x = x.to(next(self.parameters()).device)
        h = self.encoder(x)
        mean, logvar = torch.chunk(h, 2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterize the latent variables using the reparameterization trick.

        Args:
            mean (torch.Tensor): Mean tensor.
            logvar (torch.Tensor): Log variance tensor.

        Returns:
            torch.Tensor: Reparameterized latent variables.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent variables into the output tensor.
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through the VAE.
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        return self.decode(z), mean, logvar

class VAEModel(pl.LightningModule):
    """
    PyTorch Lightning module for training the VAE.
    """
    def __init__(self, cfg: DictConfig):
        """
        Initialize the VAE model.

        Args:
            cfg (DictConfig): Configuration dictionary containing hyperparameters.
                - learning_rate (float): Learning rate for the optimizer.
                - kl_loss_weight (float): Weight for the KL divergence loss.
                - triplet_margin (float): Margin for the triplet loss.
                - triplet_loss_weight (float): Weight for the triplet loss.
                - similarity_distance (float): Distance threshold for similarity in triplet loss.
        """
        super(VAEModel, self).__init__()
        self.vae = VAE(cfg)
        self.learning_rate: float = cfg.learning_rate
        self.kl_loss_weight: float = cfg.kl_loss_weight
        self.triplet_margin: float = cfg.triplet_margin
        self.triplet_loss_weight: float = cfg.triplet_loss_weight
        self.similarity_distance: float = cfg.similarity_distance

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through the VAE model.
        """
        return self.vae(x)

    def triplet_loss(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor, margin: float) -> torch.Tensor:
        """
        Compute the triplet loss.

        Args:
            anchor (torch.Tensor): Anchor embeddings.
            positive (torch.Tensor): Positive embeddings.
            negative (torch.Tensor): Negative embeddings.
            margin (float): Margin for triplet loss.

        Returns:
            torch.Tensor: Triplet loss.
        """
        pos_dist = torch.sum((anchor - positive).pow(2), dim=-1)
        neg_dist = torch.sum((anchor - negative).pow(2), dim=-1)
        loss = torch.relu(pos_dist - neg_dist + margin)
        return loss.mean()

    def get_triplets(self, embeddings: torch.Tensor, labels: torch.Tensor, similarity_distance: float) -> tuple:
        """
        Generate triplets for triplet loss.

        Args:
            embeddings (torch.Tensor): Embeddings tensor.
            labels (torch.Tensor): Labels tensor.
            similarity_distance (float): Distance threshold for similarity.

        Returns:
            tuple: Anchor, positive, and negative embeddings.
        """
        labels = labels.view(-1, 1)
        pairwise_diff = torch.abs(labels - labels.T)

        mask_positive = (pairwise_diff < similarity_distance) & (pairwise_diff > 0)
        mask_negative = pairwise_diff >= similarity_distance

        indices = torch.arange(len(labels), device=embeddings.device)

        anchor_idx, positive_idx = torch.where(mask_positive)
        _, negative_idx = torch.where(mask_negative)

        if len(anchor_idx) == 0 or len(negative_idx) == 0:
            return (torch.empty(0, device=embeddings.device),
                    torch.empty(0, device=embeddings.device),
                    torch.empty(0, device=embeddings.device))

        min_len = min(len(anchor_idx), len(negative_idx))
        anchor_idx, positive_idx, negative_idx = anchor_idx[:min_len], positive_idx[:min_len], negative_idx[:min_len]

        return embeddings[anchor_idx], embeddings[positive_idx], embeddings[negative_idx]
    
    def _step(self, batch: tuple, batch_idx: int, stage: str) -> torch.Tensor:
        """
        Perform a single training/validation/test step.
        """
        x, y = batch
        x_hat, mean, logvar = self.vae(x)
        recon_loss = nn.functional.mse_loss(x_hat, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        total_loss = recon_loss + self.kl_loss_weight * kl_loss

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
        """Override for training_step."""
        return self._step(train_batch, batch_idx, "train")

    def validation_step(self, val_batch: tuple, batch_idx: int) -> torch.Tensor:
        """Override for validation_step."""
        return self._step(val_batch, batch_idx, "val")

    def test_step(self, test_batch: tuple, batch_idx: int) -> torch.Tensor:
        """Override for test_step."""
        return self._step(test_batch, batch_idx, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

class SeismicDataModule(pl.LightningDataModule):
    """
    Data module for seismic data.
    """
    def __init__(self, cfg: DictConfig, after_vae: bool = False, vae_model: VAEModel = None):
        """
        Initialize the data module.

        Args:
            cfg (DictConfig): Configuration dictionary containing hyperparameters.
                - batch_size (int): Batch size for data loaders.
                - train_split (float): Proportion of data for training.
                - val_split (float): Proportion of data for validation.
                - data_path (str): Path to the data directory.
                - data_filename (str): Name of the data file.
            after_vae (bool): Whether to use VAE for preprocessing.
            vae_model (VAEModel): VAE model for preprocessing.
        """
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
        """Preprocess the data."""
        data = np.load(Path(self.data_path) / self.data_filename, allow_pickle=True)
        x = np.concatenate([d['signals'] for d in data])
        y = np.array([d['angle'] for d in data])
        
        x = self.scaler.fit_transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        if self.after_vae:
            self.vae_model.eval()
            with torch.no_grad():
                x_tensor, _ = self.vae_model.vae.encode(torch.tensor(x, dtype=torch.float32).permute(0, 2, 1))
        else:
            x_tensor = torch.tensor(x, dtype=torch.float32).permute(0, 2, 1)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        self.dataset = TensorDataset(x_tensor, y_tensor)

    def setup(self, stage: str = None):
        """Setup the dataset splits."""
        self.preprocess_data()
        train_size = int(self.train_split * len(self.dataset))
        val_size = int(self.val_split * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size])

    def train_dataloader(self) -> DataLoader:
        """Return the train dataloader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Return the validation dataloader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self) -> DataLoader:
        """Return the test dataloader."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

class EmbeddingToLabelModel(pl.LightningModule):
    """
    Model to map embeddings to labels.
    """
    def __init__(self, cfg: DictConfig):
        """
        Initialize the model.

        Args:
            cfg (DictConfig): Configuration dictionary containing hyperparameters.
                - learning_rate (float): Learning rate for the optimizer.
                - layers (list): List of layer dimensions.
                - input_dim (int): Input dimension.
                - margin (float): Margin for contrastive loss.
                - threshold (float): Threshold for contrastive loss.
        """
        super(EmbeddingToLabelModel, self).__init__()
        self.learning_rate: float = cfg.learning_rate
        self.layers: list = cfg.layers
        input_dim: int = cfg.input_dim
        self.margin: float = cfg.margin
        self.threshold: float = cfg.threshold
        self.contrastive_weight: float = cfg.contrastive_weight

        layers = []
        layers.append(nn.BatchNorm1d(input_dim))
        for h_dim in self.layers:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(p=cfg.dropout_rate))
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        if next(self.parameters()).device != x.device:
            x = x.to(next(self.parameters()).device)
        return self.network(x)

    def contrastive_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Compute a contrastive loss that enforces dissimilar predictions for dissimilar targets.

        For each unique pair (i, j) in the batch:
        - If |y[i] - y[j]| > self.threshold, then we require that
            |y_hat[i] - y_hat[j]| is at least self.margin. If not, we add a penalty:
            (self.margin - |y_hat[i] - y_hat[j]|)^2.
        - If |y[i] - y[j]| <= self.threshold (i.e. targets are similar), we penalize large differences
            in predictions using:
            (|y_hat[i] - y_hat[j]|)^2.

        The final loss is the average of all these pair-wise losses.

        Args:
            y_hat (torch.Tensor): Model predictions, expected shape [batch_size] or [batch_size, 1].
            y (torch.Tensor): True target values, expected shape [batch_size] or [batch_size, 1].

        Returns:
            torch.Tensor: A scalar tensor representing the averaged contrastive loss.
        """
        # Ensure y and y_hat are flattened to shape [batch_size]
        y = y.view(-1)
        y_hat = y_hat.view(-1)
        
        batch_size = y.size(0)
        
        # Compute pairwise absolute differences for targets and predictions.
        # These produce matrices of shape [batch_size, batch_size]
        diff_y = torch.abs(y.unsqueeze(0) - y.unsqueeze(1))
        diff_y_hat = torch.abs(y_hat.unsqueeze(0) - y_hat.unsqueeze(1))
        
        # Get indices for the upper triangle (i < j) to consider each pair only once.
        indices = torch.triu_indices(batch_size, batch_size, offset=1)
        diff_y_pairs = diff_y[indices[0], indices[1]]
        diff_y_hat_pairs = diff_y_hat[indices[0], indices[1]]
        
        # For pairs where the targets are dissimilar, enforce a minimum difference (margin) in predictions.
        # Otherwise, for similar targets, penalize large differences in predictions.
        loss_per_pair = torch.where(
            diff_y_pairs > self.threshold,                         # Condition: dissimilar targets
            torch.clamp(self.margin - diff_y_hat_pairs, min=0.0) ** 2, # Loss: penalize if prediction diff is less than margin
            diff_y_hat_pairs ** 2                                    # Loss: penalize large prediction differences for similar targets
        )
        
        # Return the average loss over all pairs.
        return loss_per_pair.mean()

    def _step(self, batch: tuple, batch_idx: int, stage: str) -> torch.Tensor:
        """
        Perform a single training/validation/test step.
        """
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.squeeze()

        mapping_loss = nn.functional.mse_loss(y_hat, y)
        contrastive_loss = self.contrastive_loss(y_hat, y)
        total_loss = mapping_loss + self.contrastive_weight * contrastive_loss
        
        values = {
            f"{stage}_mapping_loss": mapping_loss,
            f"{stage}_contrastive_loss": contrastive_loss,
            f"{stage}_total_loss": total_loss,
        }

        self.log_dict(values, on_epoch=True, prog_bar=True, logger=True)
        return total_loss

    def training_step(self, train_batch: tuple, batch_idx: int) -> torch.Tensor:
        """Override for training_step."""
        return self._step(train_batch, batch_idx, "train")

    def validation_step(self, val_batch: tuple, batch_idx: int) -> torch.Tensor:
        """Override for validation_step."""
        return self._step(val_batch, batch_idx, "val")

    def test_step(self, test_batch: tuple, batch_idx: int) -> torch.Tensor:
        """Override for test_step."""
        return self._step(test_batch, batch_idx, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def evaluate_performance(self, dataloader: DataLoader):
        """Evaluate the model performance."""
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


    mlflow_logger = MLFlowLogger(experiment_name="my_experiment", tracking_uri=mlflow_uri)

    # Training VAE model if train_bool is True
    if cfg.vae_model.train_bool:
        # Model
        vae_model = VAEModel(cfg.vae_model)
        trainer_vae = Trainer(
            max_epochs=cfg.vae_model.num_epochs,
            logger=mlflow_logger,
            callbacks=[EarlyStopping(monitor="val_loss", patience=cfg.vae_model.patience, min_delta=cfg.vae_model.early_stopping_delta)]
        )
        trainer_vae.fit(vae_model, data_module_vae)
        trainer_vae.test(vae_model, data_module_vae)

        # Plot latent space
        plot_latent_space(vae_model, data_module_vae.val_dataloader(), cfg.vae_model.figures_dir, "latent_space.png")
    else:
        # Load the best model if not training
        # Please use the same config file of that date
        vae_model = VAEModel.load_from_checkpoint(cfg.vae_model.best_model_path, cfg=cfg.vae_model)

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