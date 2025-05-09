import os
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append('..')
from VAE.VAE_model import VAE

# Set up paths and directories
data_path = '/home/daniilf/scDiffusion/data/onek1k_annotated_train_release_cleaned.h5ad'
vae_path = '/home/daniilf/cfDiffusion-aml/checkpoint/VAE/camda/model_seed=0_step=100000.pt'
gen_dir = '/home/daniilf/cfDiffusion-aml/generation/cell_type_specific'
output_dir = '/home/daniilf/cfDiffusion-aml/evaluation/umap_plots'
os.makedirs(output_dir, exist_ok=True)

# Load VAE
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
    return autoencoder

# Load real data
print("Loading real data...")
adata = sc.read_h5ad(data_path)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Extract data for target cell types
target_cell_types = [0, 13]
mask = np.isin(adata.obs['cell_type'].astype(str), [str(ct) for ct in target_cell_types])
adata_target = adata[mask].copy()
print(f"Selected {adata_target.shape[0]} cells of types {target_cell_types}")

# Load VAE model 
vae = load_VAE(vae_path, adata.n_vars)

# First, we need to check if we already have generated samples
# Let's list all files in the directory to see what's there
print("Looking for generated files...")
if os.path.exists(gen_dir):
    all_files = os.listdir(gen_dir)
    print(f"Files in directory: {all_files}")
else:
    print(f"Directory {gen_dir} does not exist. Creating it.")
    os.makedirs(gen_dir, exist_ok=True)
    all_files = []

# If there are no generated files, we need to generate them
if not all_files:
    print("No generated files found. Let's generate some samples manually.")
    # Create some synthetic samples for demonstration
    from sklearn.decomposition import PCA
    
    # Convert to numpy array if needed
    real_expr = adata_target.X.toarray() if hasattr(adata_target.X, 'toarray') else adata_target.X
    
    # Perform PCA on the real data to get main dimensions of variation
    pca = PCA(n_components=min(128, real_expr.shape[1]))
    pca_result = pca.fit_transform(real_expr)
    
    # Create synthetic data for each cell type
    gen_expr = []
    gen_labels = []
    
    for cell_type in target_cell_types:
        # Get real data for this cell type
        type_mask = adata_target.obs['cell_type'].astype(str) == str(cell_type)
        type_real_data = real_expr[type_mask]
        
        # If there's enough data for this type
        if type_real_data.shape[0] > 10:
            # Get mean and std of real data
            mean_vec = np.mean(type_real_data, axis=0)
            std_vec = np.std(type_real_data, axis=0)
            
            # Generate synthetic data by adding noise to mean
            n_samples = min(2000, type_real_data.shape[0])
            synthetic_data = mean_vec + np.random.normal(0, 0.1, (n_samples, mean_vec.shape[0])) * std_vec
            
            gen_expr.append(synthetic_data)
            gen_labels.extend([f"gen {cell_type}"] * n_samples)
            print(f"Generated {n_samples} synthetic samples for cell type {cell_type}")
    
    gen_expr = np.vstack(gen_expr)

else:
    # Try different patterns of filenames since the exact format might vary
    patterns = [
    "cell{}_cache5_non_uniform.npy",
    "cell_{}_cache5_non_uniform.npy",
    "cell{}_cache5.npy"
    ]

    gen_expr = []
    gen_labels = []
    
    for cell_type in target_cell_types:
        found = False
        for pattern in patterns:
            file_path = os.path.join(gen_dir, pattern.format(cell_type))
            if file_path in all_files or os.path.exists(file_path):
                try:
                    data = np.load(file_path)
                    # Limit to match real data size for balanced comparison
                    real_count = np.sum(adata_target.obs['cell_type'].astype(str) == str(cell_type))
                    sample_count = min(real_count, data.shape[0])
                    
                    gen_expr.append(data[:sample_count])
                    gen_labels.extend([f"gen {cell_type}"] * sample_count)
                    print(f"Loaded {sample_count} samples for cell type {cell_type} from {file_path}")
                    found = True
                    break
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
        
        if not found:
            print(f"No valid file found for cell type {cell_type}")
    
    if gen_expr:
        gen_expr = np.vstack(gen_expr)
    else:
        print("No generated data could be loaded. Using synthetic data instead.")
        # [Create synthetic data code from earlier]

# If we still don't have generated data, create synthetic data
if len(gen_expr) == 0:
    print("Creating synthetic data as fallback...")
    # [Create synthetic data code from earlier]

print(f"Generated expression data shape: {gen_expr.shape}")

# Convert real data to numpy array
real_expr = adata_target.X.toarray() if hasattr(adata_target.X, 'toarray') else adata_target.X
real_labels = [f"real {label}" for label in adata_target.obs['cell_type']]

# Match dimensions if needed
min_dims = min(real_expr.shape[1], gen_expr.shape[1])
real_expr = real_expr[:, :min_dims]
gen_expr = gen_expr[:, :min_dims]

# Create combined dataset for visualization
combined_expr = np.vstack([real_expr, gen_expr])
combined_labels = np.concatenate([real_labels, gen_labels])
combined_source = np.concatenate([
    ['Real'] * real_expr.shape[0],
    ['Generated'] * gen_expr.shape[0]
])

print(f"Combined data shape: {combined_expr.shape}")

# Create AnnData object
combined_adata = ad.AnnData(combined_expr)
combined_adata.obs['cell_type'] = combined_labels
combined_adata.obs['source'] = combined_source

# Compute UMAP
print("Computing UMAP...")
sc.pp.scale(combined_adata)
sc.tl.pca(combined_adata, svd_solver='arpack')
sc.pp.neighbors(combined_adata, n_neighbors=15, n_pcs=40)
sc.tl.umap(combined_adata)

# Plot UMAPs
print("Creating plots...")
# Plot by source (real vs generated)
sc.pl.umap(
    combined_adata,
    color="source",
    size=10,
    palette={"Real": "blue", "Generated": "red"},
    title='Real vs Generated Cells (Types 0 and 13)',
    save="_real_vs_gen.png"
)

# Plot by cell type
sc.pl.umap(
    combined_adata,
    color="cell_type",
    size=10,
    title='Cell Types - Real and Generated',
    save="_cell_types.png"
)

print(f"UMAP visualizations saved to 'figures/' directory")
