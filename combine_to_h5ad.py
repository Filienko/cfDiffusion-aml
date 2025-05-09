#!/usr/bin/env python
"""
Combine decoded expression data files by cell type and generate an h5ad file.

This script:
1. Reads all decoded expression files (with prefix 'expr_')
2. Extracts cell type information from filenames
3. Creates an AnnData object with cell type annotations
4. Saves it as an h5ad file

Usage:
    python combine_to_h5ad.py --input_dir /path/to/expr/files --output_file /path/to/output.h5ad
"""

import os
import argparse
import glob
import re
import numpy as np
import scanpy as sc
import anndata as ad
from tqdm import tqdm

def extract_cell_type(filename):
    """Extract cell type from filename"""
    # Expected format: expr_cell{TYPE}_cache{N}_{FLAG}.npy
    match = re.search(r'expr_cell(\d+)_', os.path.basename(filename))
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Could not extract cell type from filename: {filename}")

def combine_expression_files(input_dir, output_file, cell_type_map=None):
    """
    Combine decoded expression files by cell type and generate an h5ad file

    Args:
        input_dir: Directory containing decoded expression files
        output_file: Output h5ad file path
        cell_type_map: Optional dictionary mapping cell type indices to cell type names
    """
    # Find all expression files
    expr_files = glob.glob(os.path.join(input_dir, "expr_*.npy"))
    print(f"Found {len(expr_files)} expression files to combine.")
    
    if not expr_files:
        raise ValueError(f"No expression files found in directory: {input_dir}")
    
    # Group files by cell type
    files_by_cell_type = {}
    for file_path in expr_files:
        try:
            cell_type = extract_cell_type(file_path)
            if cell_type not in files_by_cell_type:
                files_by_cell_type[cell_type] = []
            files_by_cell_type[cell_type].append(file_path)
        except ValueError as e:
            print(f"Warning: {e}")
    
    print(f"Found {len(files_by_cell_type)} different cell types.")
    
    # Load data for each cell type
    all_data = []
    cell_types = []
    
    for cell_type, file_paths in sorted(files_by_cell_type.items()):
        print(f"Processing cell type {cell_type}...")
        
        # Load data from all files for this cell type
        cell_type_data = []
        for file_path in tqdm(file_paths, desc=f"Loading cell type {cell_type} files"):
            data = np.load(file_path)
            cell_type_data.append(data)
        
        # Concatenate data for this cell type
        if cell_type_data:
            combined_data = np.vstack(cell_type_data)[:500]
            print(f"  Loaded {combined_data.shape[0]} cells for cell type {cell_type}")
            
            all_data.append(combined_data)
            cell_types.extend([cell_type] * combined_data.shape[0])
    
    # Concatenate all data
    if all_data:
        combined_data = np.vstack(all_data)
        print(f"Combined data shape: {combined_data.shape}")
        
        # Create AnnData object
        adata = ad.AnnData(X=combined_data)
        
        # Add cell type annotation
        adata.obs['cell_type_idx'] = cell_types
        
        # Add cell type names if provided
        if cell_type_map:
            adata.obs['cell_type'] = [cell_type_map.get(ct, f"Type_{ct}") for ct in cell_types]
        
        # Add generation flag
        adata.obs['is_generated'] = "True"
        
        # Save as h5ad file
        print(f"Saving combined data to {output_file}...")
        adata.write(output_file)
        print("Done!")
    else:
        print("No data found to combine.")

def load_cell_type_map(file_path):
    """
    Load cell type mapping from a file
    
    Expected format: Each line contains "index,cell_type_name"
    """
    cell_type_map = {}
    if file_path and os.path.exists(file_path):
        print(f"Loading cell type mapping from {file_path}...")
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        idx = int(parts[0])
                        name = parts[1]
                        cell_type_map[idx] = name
        print(f"Loaded {len(cell_type_map)} cell type mappings.")
    return cell_type_map

def combine_with_real_data(generated_h5ad, real_h5ad, output_file):
    """
    Combine generated data with real data
    
    Args:
        generated_h5ad: Path to generated data h5ad file
        real_h5ad: Path to real data h5ad file
        output_file: Output h5ad file path
    """
    print(f"Loading generated data from {generated_h5ad}...")
    gen_adata = sc.read_h5ad(generated_h5ad)
    gen_adata.obs['source'] = 'generated'
    
    print(f"Loading real data from {real_h5ad}...")
    real_adata = sc.read_h5ad(real_h5ad)
    real_adata.obs['source'] = 'real'
    print("Applying consistent preprocessing to real data...")
    sc.pp.filter_genes(real_adata, min_cells=3)
    sc.pp.filter_cells(real_adata, min_genes=10)
    #real_adata.var_names_make_unique()
    print(real_adata.shape)
    real_adata.var_names.to_series().to_csv("var_names.txt", index=False, header=False)
    print(real_adata.var_names)
    # Ensure real data has cell_type_idx column if needed
    if 'cell_type_idx' not in real_adata.obs.columns and 'cell_type' in real_adata.obs.columns:
        # Create a mapping from cell type names to indices
        unique_types = real_adata.obs['cell_type'].unique()
        type_to_idx = {name: idx for idx, name in enumerate(unique_types)}
        
        # Apply mapping
        real_adata.obs['cell_type_idx'] = [type_to_idx.get(ct, -1) for ct in real_adata.obs['cell_type']]
    
    # Check if the gene sets match
    if gen_adata.shape[1] != real_adata.shape[1]:
        print(f"Warning: Gene counts don't match! Generated: {gen_adata.shape[1]}, Real: {real_adata.shape[1]}")
        # If you want to proceed anyway, you'd need a strategy to align genes
        # This is simplified and might need adjustment based on your data
        min_genes = min(gen_adata.shape[1], real_adata.shape[1])
        gen_adata = gen_adata[:, :min_genes]
        real_adata = real_adata[:, :min_genes]
        print(f"Truncated both datasets to {min_genes} genes")
    if gen_adata.shape[1] == real_adata.shape[1]:
        gen_adata.var_names = real_adata.var_names
        print(gen_adata.var_names)
    # Concatenate the datasets
    print("Concatenating datasets...")
    combined = ad.concat([real_adata, gen_adata], join='outer', label='source')
    
    # Save the combined data
    print(f"Saving combined data to {output_file}...")
    combined.write(output_file)
    print("Done!")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Combine expression files by cell type")
    
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing decoded expression files")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output h5ad file path")
    parser.add_argument("--cell_type_map", type=str, default=None,
                        help="Path to cell type mapping file (optional)")
    parser.add_argument("--real_data", type=str, default=None,
                        help="Path to real data h5ad file (optional)")
    parser.add_argument("--combined_output", type=str, default=None,
                        help="Output path for combined real and generated data (optional)")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    print("Starting combination process...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output file: {args.output_file}")
    
    # Load cell type mapping if provided
    cell_type_map = load_cell_type_map(args.cell_type_map)
    
    # Combine expression files
    combine_expression_files(args.input_dir, args.output_file, cell_type_map)
    
    # Combine with real data if provided
    if args.real_data and args.combined_output:
        print("\nCombining with real data...")
        combine_with_real_data(args.output_file, args.real_data, args.combined_output)
    
    print("Combination process completed!")

if __name__ == "__main__":
    main()
