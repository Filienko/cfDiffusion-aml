import torch
import numpy as np
import scanpy as sc
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from VAE.VAE_model import VAE

def load_vae(vae_path, num_genes, hidden_dim=128):
    """Load the trained VAE model"""
    autoencoder = VAE(
        num_genes=num_genes,
        device='cuda',
        seed=0,
        loss_ae='mse',
        hidden_dim=hidden_dim,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(vae_path))
    autoencoder.to('cuda')
    
    # IMPORTANT: Don't call .eval() directly since it calls train(False)
    # which conflicts with the overridden train() method
    # Instead, we'll manually set the model's training mode
    for module in autoencoder.modules():
        if isinstance(module, torch.nn.Dropout) or isinstance(module, torch.nn.BatchNorm1d):
            module.training = False
    
    return autoencoder

def evaluate_reconstruction(vae, data, n_samples=1000):
    """
    Evaluate the reconstruction quality of the VAE.
    
    Args:
        vae: Trained VAE model
        data: AnnData object with gene expression data
        n_samples: Number of samples to use for evaluation
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Preprocess data
    if hasattr(data.X, 'toarray'):
        gene_expression = data.X.toarray()
    else:
        gene_expression = data.X
    
    # Select a random subset of cells
    if n_samples < gene_expression.shape[0]:
        indices = np.random.choice(gene_expression.shape[0], n_samples, replace=False)
        gene_expression = gene_expression[indices]
        cell_types = data.obs['cell_type'].iloc[indices] if 'cell_type' in data.obs else None
    else:
        cell_types = data.obs['cell_type'] if 'cell_type' in data.obs else None
    
    # Convert to torch tensor
    gene_expression_tensor = torch.tensor(gene_expression, dtype=torch.float).cuda()
    
    # Encode and decode
    with torch.no_grad():
        latent_vectors = vae(gene_expression_tensor, return_latent=True)
        reconstructed_expression = vae(latent_vectors, return_decoded=True)
    
    # Move back to CPU for evaluation
    original = gene_expression
    reconstructed = reconstructed_expression.cpu().numpy()
    latent = latent_vectors.cpu().numpy()
    
    # Calculate metrics
    mse = mean_squared_error(original, reconstructed)
    r2 = r2_score(original.flatten(), reconstructed.flatten())
    
    # Per-gene correlation
    gene_correlations = []
    for i in range(original.shape[1]):
        corr = np.corrcoef(original[:, i], reconstructed[:, i])[0, 1]
        # Handle NaN values (can occur if a gene has no variance)
        if not np.isnan(corr):
            gene_correlations.append(corr)
    
    # Per-cell correlation
    cell_correlations = []
    for i in range(original.shape[0]):
        corr = np.corrcoef(original[i, :], reconstructed[i, :])[0, 1]
        if not np.isnan(corr):
            cell_correlations.append(corr)
    
    # Return metrics
    return {
        'mse': mse,
        'r2': r2,
        'median_gene_correlation': np.median(gene_correlations),
        'median_cell_correlation': np.median(cell_correlations),
        'original': original,
        'reconstructed': reconstructed,
        'gene_correlations': gene_correlations,
        'cell_correlations': cell_correlations,
        'latent': latent,
        'cell_types': cell_types
    }

def plot_latent_umap(latent, cell_types, output_file=None):
    """
    Create a UMAP visualization of the latent space colored by cell type.
    
    Args:
        latent: Latent vectors (n_cells, latent_dim)
        cell_types: Array of cell type labels
        output_file: Path to save the plot
    """
    # Create AnnData object
    adata = sc.AnnData(latent)
    adata.obs['cell_type'] = cell_types.values if hasattr(cell_types, 'values') else cell_types
    
    # Compute UMAP
    sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
    sc.tl.umap(adata)
    
    # Plot UMAP
    plt.figure(figsize=(12, 10))
    sc.pl.umap(
        adata, 
        color=['cell_type'], 
        title='UMAP of Latent Space by Cell Type',
        frameon=False,
        return_fig=True
    )
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    return adata

def plot_results(results, output_file=None):
    """
    Plot the evaluation results.
    """
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Plot a scatterplot of original vs reconstructed values
    sample_idx = np.random.choice(results['original'].shape[0])
    axes[0, 0].scatter(
        results['original'][sample_idx], 
        results['reconstructed'][sample_idx],
        alpha=0.5, s=3
    )
    axes[0, 0].set_xlabel('Original gene expression')
    axes[0, 0].set_ylabel('Reconstructed gene expression')
    axes[0, 0].set_title(f'Gene Expression Reconstruction (Cell #{sample_idx})')
    
    # Add the identity line
    min_val = min(results['original'][sample_idx].min(), results['reconstructed'][sample_idx].min())
    max_val = max(results['original'][sample_idx].max(), results['reconstructed'][sample_idx].max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # 2. Plot distribution of gene correlations
    axes[0, 1].hist(results['gene_correlations'], bins=50)
    axes[0, 1].set_xlabel('Correlation')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Gene-level Correlations (Original vs. Reconstructed)')
    
    # 3. Plot distribution of cell correlations
    axes[1, 0].hist(results['cell_correlations'], bins=50)
    axes[1, 0].set_xlabel('Correlation')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Cell-level Correlations (Original vs. Reconstructed)')
    
    # 4. Plot a heatmap of original vs reconstructed for a few genes
    n_cells = 50
    n_genes = 50
    cells_idx = np.random.choice(results['original'].shape[0], n_cells, replace=False)
    genes_idx = np.random.choice(results['original'].shape[1], n_genes, replace=False)
    
    original_subset = results['original'][cells_idx][:, genes_idx]
    reconstructed_subset = results['reconstructed'][cells_idx][:, genes_idx]
    
    combined = np.vstack([original_subset, reconstructed_subset])
    im = axes[1, 1].imshow(combined, aspect='auto', cmap='viridis')
    plt.colorbar(im, ax=axes[1, 1])
    axes[1, 1].axhline(y=n_cells-0.5, color='red', linestyle='-')
    axes[1, 1].set_title('Original (top) vs Reconstructed (bottom)')
    
    # Add metrics as text
    metrics_text = f"MSE: {results['mse']:.4f}\n"
    metrics_text += f"R²: {results['r2']:.4f}\n"
    metrics_text += f"Median Gene Correlation: {results['median_gene_correlation']:.4f}\n"
    metrics_text += f"Median Cell Correlation: {results['median_cell_correlation']:.4f}"
    
    plt.figtext(0.5, 0.01, metrics_text, ha='center', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    plt.show()

# Main execution
def main():
    # Parameters
    vae_path = '/home/daniilf/cfDiffusion-aml/checkpoint/VAE/camda/model_seed=0_step=600000.pt'
    data_path = '/home/daniilf/scDiffusion/data/onek1k_annotated_train_release.h5ad'
    num_genes = 24622
    output_dir = './vae_evaluation'
    
    # Create output directory
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    print("Loading data...")
    adata = sc.read_h5ad(data_path)
    adata.var_names_make_unique()
    sc.pp.filter_cells(adata, min_genes=10)
    sc.pp.filter_genes(adata, min_cells=3)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    # Check if cell_type column exists
    if 'cell_type' not in adata.obs.columns:
        print("Warning: 'cell_type' column not found in data. Attempting to use alternative column...")
        # Try to find a column that might contain cell types
        potential_columns = ['celltype', 'cell_type1', 'CellType', 'cell.type', 'cluster']
        for col in potential_columns:
            if col in adata.obs.columns:
                print(f"Using '{col}' column as cell type information")
                adata.obs['cell_type'] = adata.obs[col]
                break
    
    # Preprocess data
    print("Preprocessing data...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Load the VAE
    print("Loading VAE model...")
    vae = load_vae(vae_path, num_genes)
    
    # Evaluate reconstruction
    print("Evaluating reconstruction...")
    results = evaluate_reconstruction(vae, adata, n_samples=2000)
    
    # Print metrics
    print(f"MSE: {results['mse']:.4f}")
    print(f"R²: {results['r2']:.4f}")
    print(f"Median Gene Correlation: {results['median_gene_correlation']:.4f}")
    print(f"Median Cell Correlation: {results['median_cell_correlation']:.4f}")
    
    # Plot reconstruction results
    print("Plotting reconstruction results...")
    plot_results(results, output_file=os.path.join(output_dir, 'vae_reconstruction_evaluation.png'))
    
    # Plot UMAP of latent space
    if results['cell_types'] is not None:
        print("Creating UMAP visualization of latent space...")
        adata_latent = plot_latent_umap(
            results['latent'], 
            results['cell_types'],
            output_file=os.path.join(output_dir, 'latent_space_umap.png')
        )
        
        # Save latent space AnnData for further analysis
        adata_latent.write(os.path.join(output_dir, 'latent_space.h5ad'))
    else:
        print("Cannot create UMAP visualization: cell type information not available")
    
    print(f"Evaluation complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main()
