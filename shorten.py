import scanpy as sc
import numpy as np
import argparse
import os

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create a smaller subset of an h5ad file')
    parser.add_argument('--input_file', type=str, required=True, help='Path to input h5ad file')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save output h5ad file')
    parser.add_argument('--num_cells', type=int, default=50000, help='Number of cells to keep')
    parser.add_argument('--num_genes', type=int, default=None, help='Number of genes to keep (most variable)')
    parser.add_argument('--random', action='store_true', help='Randomly sample cells instead of taking first N')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--filter', action='store_true', help="filter genes (same as is in sapients utils)")
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Input file '{args.input_file}' does not exist!")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Loading dataset from {args.input_file}...")
    adata = sc.read_h5ad(args.input_file)

    if args.filter:
        print(f"Filtering to cells with min_genes = 10...")
        sc.pp.filter_cells(adata, min_genes=10)
    print(f"Original dataset shape: {adata.shape}")
    
    # Filter cells
    if adata.shape[0] > args.num_cells:
        if args.random:
            print(f"Randomly sampling {args.num_cells} cells...")
            idx = np.random.choice(adata.shape[0], args.num_cells, replace=False)
            adata_subset = adata[idx].copy()
        else:
            print(f"Taking first {args.num_cells} cells...")
            adata_subset = adata[:args.num_cells].copy()
        print(f"Dataset shape after cell filtering: {adata_subset.shape}")
    else:
        print(f"Dataset already has fewer than {args.num_cells} cells. No cell filtering needed.")
        adata_subset = adata.copy()
    
    # Filter genes if requested
    if args.num_genes is not None and adata_subset.shape[1] > args.num_genes:
        print(f"Selecting {args.num_genes} most variable genes...")
        # Calculate gene variance (robust to potential sparsity)
        if isinstance(adata_subset.X, np.ndarray):
            gene_vars = np.var(adata_subset.X, axis=0)
        else:  # It's likely a sparse matrix
            gene_vars = np.var(adata_subset.X.toarray(), axis=0)
            
        # Get indices of top genes by variance
        top_genes_idx = np.argsort(gene_vars)[-args.num_genes:]
        
        # Subset AnnData object to keep only those genes
        adata_subset = adata_subset[:, top_genes_idx].copy()
        print(f"Dataset shape after gene filtering: {adata_subset.shape}")
    
    # Save the subset
    print(f"Saving subset to {args.output_file}...")
    adata_subset.write_h5ad(args.output_file)
    print("Done!")

if __name__ == "__main__":
    main()
