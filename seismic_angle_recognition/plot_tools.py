import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

def plot_latent_space(model: pl.LightningModule, dataloader: DataLoader, num_batches: int = 100):
    model.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for i, (x, y) in enumerate(dataloader):
            if i >= num_batches:
                break
            mean, logvar = model.vae.encode(x)
            z = model.vae.reparameterize(mean, logvar)
            latents.append(z)
            labels.append(y)


    latents = torch.cat(latents).cpu().numpy()
    labels = torch.cat(labels).cpu().numpy()

    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(latents[:, 0], latents[:, 1], c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter)
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('Latent Space Visualization')
    plt.show()