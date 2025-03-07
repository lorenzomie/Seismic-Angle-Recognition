import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pathlib import Path

def plot_latent_space(model: pl.LightningModule, dataloader: DataLoader, file_path: str, file_name: str, num_batches: int = 1000, additional_dataloader: DataLoader = None):
    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= num_batches:
                break
            mean, _ = model.vae.encode(x)
            z = mean  # Use only the mean, set logvar to zero
            latents.append(z)
            labels.append(y)

    latents = torch.cat(latents).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='viridis', alpha=0.5)

    
    if additional_dataloader is not None:
        for i, (x, y) in enumerate(additional_dataloader):
            with torch.no_grad():
                mean, _ = model.vae.encode(x)
                z = mean  # Use only the mean, set logvar to zero
                z = z.cpu().numpy()
                plt.scatter(z[:, 0], z[:, 1], c='red', marker='x')
                for j in range(len(z)):
                    plt.text(z[j, 0], z[j, 1], str(y[j].item()), color='red', fontsize=8)

    plt.colorbar(scatter)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Visualization')

    # Ensure the directory exists
    Path(file_path).mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(file_path) / file_name)
    plt.close()