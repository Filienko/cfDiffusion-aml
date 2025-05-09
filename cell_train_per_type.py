"""
Train a diffusion model for each cell type separately.
"""

import argparse
import os
import torch
import numpy as np
import random

from guided_diffusion import dist_util, logger
from guided_diffusion.cell_datasets_sapiens import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from guided_diffusion.train_util import TrainLoop
import scanpy as sc
from sklearn.preprocessing import LabelEncoder

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_for_cell_type(cell_type_idx, cell_type_name, args_dict):
    """
    Train a diffusion model for a specific cell type.
    
    Args:
        cell_type_idx: The index of the cell type
        cell_type_name: The name of the cell type (for logging)
        args_dict: Dictionary of arguments for training
    """
    print(f"Training model for cell type {cell_type_name} (index {cell_type_idx})")
    
    # Create cell-type specific output directory
    cell_type_model_name = f"{args_dict['model_name']}_{cell_type_idx}"
    cell_type_save_dir = os.path.join(args_dict['save_dir'], cell_type_model_name)
    os.makedirs(cell_type_save_dir, exist_ok=True)
    
    # Set up distributed training
    dist_util.setup_dist()
    
    # Create model and diffusion process
    print("Creating model and diffusion...")
    # IMPORTANT FIX: Ensure num_classes is a list with a single value
    args_dict['num_classes'] = [args_dict['num_classes']]
    model, diffusion = create_model_and_diffusion(**args_dict)
    model.to(dist_util.dev())
    
    # Create schedule sampler
    schedule_sampler = create_named_schedule_sampler(args_dict['schedule_sampler'], diffusion)
    
    # Load the data for filtering
    print("Loading data to filter by cell type...")
    adata = sc.read_h5ad(args_dict['data_dir'])
    
    # We need to get cell type labels consistent with your dataset
    classes = adata.obs['cell_type'].values
    label_encoder = LabelEncoder()
    label_encoder.fit(classes)
    encoded_classes = label_encoder.transform(classes)
    
    # Filter indices for the current cell type
    cell_type_indices = np.where(encoded_classes == cell_type_idx)[0]
    print(f"Found {len(cell_type_indices)} cells of type {cell_type_name}")
    
    if len(cell_type_indices) < args_dict['batch_size']:
        print(f"Warning: Cell type {cell_type_name} has fewer samples ({len(cell_type_indices)}) than batch size ({args_dict['batch_size']})")
        if len(cell_type_indices) < 10:  # Too few samples to train meaningfully
            print(f"Skipping cell type {cell_type_name} due to insufficient data")
            return
        # Adjust batch size for small cell types
        args_dict['batch_size'] = max(8, len(cell_type_indices) // 4)
        print(f"Adjusted batch size to {args_dict['batch_size']}")
    
    # Create a modified data loader that filters for this cell type
    print("Creating filtered data loader...")
    data = load_data(
        data_dir=args_dict['data_dir'],
        batch_size=args_dict['batch_size'],
        vae_path=args_dict['vae_path'],
        train_vae=False,
        cell_type_filter=cell_type_idx  # This parameter will be used in the modified load_data function
    )
    
    # Train the model
    print(f"Starting training for cell type {cell_type_name}...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args_dict['batch_size'],
        microbatch=args_dict['microbatch'],
        lr=args_dict['lr'],
        ema_rate=args_dict['ema_rate'],
        log_interval=args_dict['log_interval'],
        save_interval=args_dict['save_interval'],
        resume_checkpoint=args_dict.get('resume_checkpoint', ""),
        use_fp16=args_dict['use_fp16'],
        fp16_scale_growth=args_dict['fp16_scale_growth'],
        schedule_sampler=schedule_sampler,
        weight_decay=args_dict['weight_decay'],
        lr_anneal_steps=args_dict['lr_anneal_steps'],
        model_name=cell_type_model_name,
        save_dir=args_dict['save_dir']
    ).run_loop()
    
    print(f"Completed training for cell type {cell_type_name}")

def create_argparser():
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-5,
        weight_decay=0.0001,
        lr_anneal_steps=200000,  # Reduced from 800000 for per-cell-type training
        batch_size=64,  # Reduced batch size as we'll have fewer samples per cell type
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.99",
        log_interval=1000,
        save_interval=50000,  # Reduced from 400000
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--branch", type=int, default=0)
    parser.add_argument("--cache_interval", type=int, default=5)
    parser.add_argument("--non_uniform", action="store_true")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the h5ad dataset")
    parser.add_argument("--vae_path", type=str, required=True,
                        help="Path to the trained VAE model")
    parser.add_argument("--save_dir", type=str, default="./checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--model_name", type=str, default="per_celltype_diffusion",
                        help="Base name for model checkpoints")
    parser.add_argument("--cell_types", type=str, default="all",
                        help="Comma-separated list of cell type indices to train on, or 'all'")
    
    args, _ = parser.parse_known_args()
    args_dict = {arg: getattr(args, arg) for arg in vars(args)}
    
    # Update with default args
    defaults.update(model_and_diffusion_defaults())
    args_dict.update(defaults)
    
    return args_dict

def main():
    setup_seed(1234)
    args_dict = create_argparser()
    
    # Load the dataset to get cell type information
    print("Loading dataset to enumerate cell types...")
    adata = sc.read_h5ad(args_dict['data_dir'])
    cell_types = adata.obs['cell_type'].unique()
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(adata.obs['cell_type'])
    
    # Determine which cell types to train on
    if args_dict['cell_types'] == 'all':
        cell_type_indices = list(range(len(cell_types)))
    else:
        cell_type_indices = [int(idx) for idx in args_dict['cell_types'].split(',')]
    
    print(f"Will train on {len(cell_type_indices)} cell types:")
    for idx in cell_type_indices:
        if idx < len(cell_types):
            print(f"  {idx}: {cell_types[idx]}")
    
    # Train a separate model for each cell type
    for idx in cell_type_indices:
        if idx < len(cell_types):
            cell_type_name = cell_types[idx]
            train_for_cell_type(idx, cell_type_name, args_dict.copy())
        else:
            print(f"Warning: Cell type index {idx} out of range (max: {len(cell_types)-1})")

if __name__ == "__main__":
    main()
