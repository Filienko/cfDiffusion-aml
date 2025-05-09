import os
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('..')
import argparse
from sklearn.model_selection import train_test_split

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--subsample', action='store_true', help='Subsample to 10k rows with class balance')
args = parser.parse_args()
from VAE.VAE_model import VAE

# Set up paths and directories
#data_path = '/home/daniilf/scDiffusion/data/onek1k_annotated_train_release_cleaned.h5ad'
data_path = '/home/daniilf/scDiffusion/data/onek1k_annotated_train_release.h5ad'
vae_path = '/home/daniilf/cfDiffusion-aml/checkpoint/VAE/camda/model_seed=0_step=600000.pt'
output_dir = '/home/daniilf/cfDiffusion-aml/evaluation/vae_latent_space'
os.makedirs(output_dir, exist_ok=True)

# Load VAE with a patched approach to avoid the train method conflict
def load_VAE(vae_path, num_genes):
    autoencoder = VAE(
        num_genes=num_genes,
        device='cuda',
        seed=0,
        loss_ae='mse',
        hidden_dim=128,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(vae_path))
    autoencoder.to('cuda')
    # Skip calling eval() since it conflicts with the VAE's train method
    return autoencoder

# Load real data
print("Loading real data...")
adata = sc.read_h5ad(data_path)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
if args.subsample and adata.shape[0] > 10000:
    print("Subsampling to 10k with class balance...")
    stratify_labels = adata.obs['cell_type'].astype(str)
    indices, _ = train_test_split(
        np.arange(len(stratify_labels)),
        train_size=10000,
        stratify=stratify_labels,
        random_state=42,
    )
    adata = adata[indices].copy()

print(f"Dataset shape: {adata.shape}")
print(f"Cell types: {adata.obs['cell_type'].unique()}")

# Get gene expression data
cell_data = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X

# Load VAE model
vae = load_VAE(vae_path, adata.n_vars)

# Process data through VAE in batches to avoid OOM issues
print("Processing through VAE to get latent representations...")
latent_vectors = []
batch_size = 128
num_batches = (adata.shape[0] + batch_size - 1) // batch_size

with torch.no_grad():
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, adata.shape[0])
        batch = cell_data[start_idx:end_idx]
        
        batch_tensor = torch.tensor(batch, dtype=torch.float32).cuda()
        batch_latent = vae(batch_tensor, return_latent=True)
        latent_vectors.append(batch_latent.cpu().numpy())
        
        if (i + 1) % 10 == 0 or (i + 1) == num_batches:
            print(f"Processed batch {i+1}/{num_batches}")

# Concatenate all batches
latent_vectors = np.concatenate(latent_vectors, axis=0)
print(f"Latent vectors shape: {latent_vectors.shape}")

# Create AnnData object for latent space
latent_adata = ad.AnnData(latent_vectors)
latent_adata.obs = adata.obs.copy()  # Copy metadata

# Compute UMAP for visualization
print("Computing UMAP of latent space...")
sc.pp.neighbors(latent_adata, n_neighbors=15, use_rep='X')
sc.tl.umap(latent_adata)

# Plot UMAP colored by cell type
print("Creating plots...")
all_cell_types = latent_adata.obs['cell_type'].astype(str).unique()

# Plot all cell types in one figure
plt.figure(figsize=(12, 10))
sc.pl.umap(
    latent_adata, 
    color='cell_type',
    title='VAE Latent Space - All Cell Types',
    save="_all_cell_types_50k.png"
)

# Also save a version to our custom output directory
plt.figure(figsize=(12, 10))
sc.pl.umap(
    latent_adata, 
    color='cell_type',
    title='VAE Latent Space - All Cell Types'
)
plt.savefig(os.path.join(output_dir, 'vae_latent_space_all_celltypes_50k.png'), dpi=300, bbox_inches='tight')

'''
# For better visualization, also create plots with fewer cell types at a time
cell_type_groups = [all_cell_types[i:i+5] for i in range(0, len(all_cell_types), 5)]

for i, group in enumerate(cell_type_groups):
    plt.figure(figsize=(12, 10))
    sc.pl.umap(
        latent_adata,
        color='cell_type',
        groups=group,
        title=f'VAE Latent Space - Cell Types Group {i+1}',
        save=f"_cell_types_group_{i+1}.png"
    )

print(f"UMAP visualizations saved to the figures directory")

# Additionally, let's save the latent space data for later use
#latent_adata.write(os.path.join(output_dir, 'vae_latent_space.h5ad'))
#print(f"Latent space data saved to {os.path.join(output_dir, 'vae_latent_space.h5ad')}")
'''
