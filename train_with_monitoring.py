"""
Modified training script with latent space monitoring integration.
This script demonstrates how to integrate the latent space monitor
with your cell-type specific diffusion model training.
"""

import argparse
import os
import torch
import numpy as np
import random
import scanpy as sc
from sklearn.preprocessing import LabelEncoder

from guided_diffusion import dist_util, logger
from guided_diffusion.cell_datasets_sapiens_modified import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from guided_diffusion.train_util import TrainLoop
from latent_space_monitor import LatentSpaceMonitor
from VAE.VAE_model import VAE

def setup_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_vae(vae_path, num_genes, hidden_dim=128):
    """Load the trained VAE model"""
    print(f"Loading VAE from {vae_path}")
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
    autoencoder.eval()  # Set to evaluation mode
    return autoencoder

def train_for_cell_type(cell_type_idx, cell_type_name, args_dict):
    """
    Train a diffusion model for a specific cell type with latent space monitoring.
    
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
    
    # Create latent space visualization directory
    latent_vis_dir = os.path.join(cell_type_save_dir, "latent_visualizations")
    os.makedirs(latent_vis_dir, exist_ok=True)
    
    # Set up distributed training
    dist_util.setup_dist()
    
    # Load the VAE model
    adata = sc.read_h5ad(args_dict['data_dir'])
    num_genes = adata.n_vars
    vae = load_vae(args_dict['vae_path'], num_genes)
    
    # Create model and diffusion process
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_dict)
    model.to(dist_util.dev())
    
    # Create schedule sampler
    schedule_sampler = create_named_schedule_sampler(args_dict['schedule_sampler'], diffusion)
    
    # Initialize latent space monitor
    latent_monitor = LatentSpaceMonitor(
        vae_model=vae,
        output_dir=latent_vis_dir,
        visualization_freq=args_dict.get('visualization_freq', 5000),
        device=dist_util.dev()
    )
    
    # Create a filtered data loader for this cell type
    print(f"Creating data loader for cell type {cell_type_idx}...")
    data_generator = load_data(
        data_dir=args_dict['data_dir'],
        batch_size=args_dict['batch_size'],
        vae_path=args_dict['vae_path'],
        train_vae=False,
        cell_type_filter=cell_type_idx
    )
    
    # Create a temporary data loader just for setting reference data
    # This collects a batch of data to use as reference for the latent space monitor
    reference_data = []
    reference_labels = []
    for i in range(min(10, args_dict.get('reference_batches', 5))):  # Collect 5-10 batches
        try:
            batch, label = next(data_generator)
            reference_data.append(batch)
            reference_labels.append(label)
        except StopIteration:
            break
    
    if not reference_data:
        print(f"Warning: Could not collect reference data for cell type {cell_type_idx}")
    else:
        # Set reference data for latent space monitor
        reference_loader = [(torch.tensor(batch), torch.tensor(label)) 
                           for batch, label in zip(reference_data, reference_labels)]
        latent_monitor.set_reference_data(reference_loader)
    
    # Create a custom training loop that incorporates the latent space monitor
    class MonitoredTrainLoop(TrainLoop):
        def __init__(self, *args, latent_monitor=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.latent_monitor = latent_monitor
        
        def run_loop(self):
            step_count = 0
            try:
                while (not self.lr_anneal_steps or 
                       step_count + self.resume_step < self.lr_anneal_steps):
                    batch, cond = next(self.data)
                    self.run_step(batch, cond)
                    
                    # Visualize latent space periodically
                    if self.latent_monitor is not None:
                        self.latent_monitor.step_counter = step_count + self.resume_step
                        self.latent_monitor.visualize_step(
                            diffusion_model=self.model,
                            batch_size=min(32, self.batch_size)
                        )
                    
                    if step_count % self.save_interval == 0:
                        self.save()
                        print(f"Saved checkpoint at step {step_count}")
                    
                    step_count += 1
                    self.step = step_count
                    
                    if step_count % 1000 == 0:
                        torch.cuda.empty_cache()
                        print("Step", step_count)
                    
                    if os.environ.get("DIFFUSION_TRAINING_TEST", "") and step_count > 0:
                        return
                    
            except Exception as e:
                print(f"Exception encountered during training: {e}")
                import traceback
                traceback.print_exc()
                self.save()  # Save before exiting
    
    # Train the model with latent space monitoring
    print(f"Starting training for cell type {cell_type_name}...")
    MonitoredTrainLoop(
        model=model,
        diffusion=diffusion,
        data=data_generator,
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
        save_dir=args_dict['save_dir'],
        latent_monitor=latent_monitor
    ).run_loop()
    
    print(f"Completed training for cell type {cell_type_name}")

def create_argparser():
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0001,
        lr_anneal_steps=200000,
        batch_size=64,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=1000,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        visualization_freq=5000,  # How often to visualize latent space
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
