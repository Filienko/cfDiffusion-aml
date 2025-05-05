"""
Evaluate per-cell-type diffusion models by plotting UMAPs and comparing real vs generated data
"""
import os
import argparse
import numpy as np
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import torch
import sys
sys.path.append('..')
from VAE.VAE_model import VAE

def load_VAE(vae_path, num_genes, hidden_dim=128):
    """
    Load the trained VAE model
    """
    print(f"Loading VAE from {vae_path}...")
    autoencoder = VAE(
        num_genes=num_genes, 
        device='cuda',
        seed=0,
        loss_ae='mse',
        hidden_dim=hidden_dim,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(vae_path))
    return autoencoder

def setup_output_dirs(output_dir):
    """
    Create necessary output directories for plots
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "per_cell_type"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "combined"), exist_ok=True)

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description="Evaluate per-cell-type diffusion models")
    parser.add_argument("--data_path", type=str, required=True, 
                        help="Path to the original dataset (h5ad)")
    parser.add_argument("--gen_dir", type=str, required=True, 
                        help="Directory containing generated samples")
    parser.add_argument("--vae_path", type=str, required=True,
                        help="Path to the trained VAE model")
    parser.add_argument("--output_dir", type=str, default="./eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--cache_interval", type=int, default=5,
                        help="Cache interval used for generation")
    parser.add_argument("--non_uniform", action="store_true",
                        help="Whether non-uniform sampling was used")
    
    return parser.parse_args()

def load_and_decode_generated_samples(gen_dir, cell_types, vae, args):
    """
    Load and decode the generated latent vectors back to gene expression
    """
    all_gen_data = []
    all_gen_labels = []
    
    suffix = "_non_uniform" if args.non_uniform else "_uniform"
    
    # Load generated data for each cell type
    for i, cell_type in enumerate(cell_types):
        file_path = f"{gen_dir}/cell{i}_cache{args.cache_interval}{suffix}.npy"
        
        if not os.path.exists(file_path):
            print(f"Warning: Generated data not found for cell type {i} at {file_path}")
            continue
        
        # Load latent vectors
        try:
            latent_vectors = np.load(file_path, allow_pickle=True)
            print(f"Loaded {latent_vectors.shape[0]} samples for cell type {i}")
            
            # Decode latent vectors to gene expression
            with torch.no_grad():
                gen_data_tensor = torch.tensor(latent_vectors, dtype=torch.float32).cuda()
                gen_expr = vae(gen_data_tensor, return_decoded=True).cpu().numpy()
            
            all_gen_data.append(gen_expr)
            all_gen_labels.extend([f"gen {cell_type}"] * gen_expr.shape[0])
            
        except Exception as e:
            print(f"Error loading/decoding cell type {i}: {e}")
    
    if not all_gen_data:
        raise ValueError("No generated data could be loaded")
    
    return np.vstack(all_gen_data), np.array(all_gen_labels)

def main():
    args = parse_args()
    setup_output_dirs(args.output_dir)
    
    # Load original data
    print("Loading original dataset...")
    adata = sc.read_h5ad(args.data_path)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Get cell types
    cell_types = adata.obs['cell_type'].astype(str).values  # Get as strings
    cell_type_categories = adata.obs['cell_type'].unique()
    
    # Load VAE and decode generated samples
    vae = load_VAE(args.vae_path, adata.n_vars)
    vae.to('cuda')
    vae.eval()
    
    gen_expr, gen_labels = load_and_decode_generated_samples(
        args.gen_dir, cell_type_categories, vae, args
    )
    
    # Prepare combined dataset for evaluation
    print("Preparing combined dataset...")
    
    # If working with sparse matrix, convert to dense
    if hasattr(adata.X, 'toarray'):
        real_expr = adata.X.toarray()
    else:
        real_expr = adata.X
    
    # Create true and generated labels
    true_labels = [f"true {label}" for label in cell_types]
    
    # Create combined AnnData object
    combined_expr = np.vstack([real_expr, gen_expr])
    combined_labels = np.concatenate([true_labels, gen_labels])
    combined_groups = np.concatenate([
        np.full(len(true_labels), "Real"),
        np.full(len(gen_labels), "Generated")
    ])
    
    combined_adata = ad.AnnData(combined_expr)
    combined_adata.obs['cell_type'] = combined_labels
    combined_adata.obs['group'] = combined_groups
    
    # Process for visualization
    print("Processing and visualizing...")
    sc.pp.scale(combined_adata)
    sc.tl.pca(combined_adata, svd_solver='arpack')
    sc.pp.neighbors(combined_adata, n_neighbors=15, n_pcs=40)
    sc.tl.umap(combined_adata)
    
    # Plot combined UMAP with real vs generated
    plt.figure(figsize=(10, 8))
    sc.pl.umap(
        combined_adata,
        color="group",
        size=6,
        title='Real vs Generated Cells',
        save=os.path.join(args.output_dir, "combined", "real_vs_gen.png")
    )
    
    # Plot combined UMAP colored by cell type
    plt.figure(figsize=(12, 10))
    sc.pl.umap(
        combined_adata,
        color="cell_type",
        size=6,
        title='Cell Types - Real and Generated',
        save=os.path.join(args.output_dir, "combined", "cell_types.png")
    )
    
    # Generate per-cell-type plots
    for i, cell_type in enumerate
