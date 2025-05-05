"""
Train a diffusion model for each cell type separately, with UMAP monitoring
of the latent space during training using scanpy's UMAP implementation.

Before running, make sure to have scanpy and anndata installed:
pip install scanpy anndata
"""

import argparse
import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime

from guided_diffusion import dist_util, logger
from guided_diffusion.cell_datasets_sapiens import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from guided_diffusion.train_util import TrainLoop
import scanpy as sc
import anndata as ad
from sklearn.preprocessing import LabelEncoder
from VAE.VAE_model import VAE

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_vae(vae_path, num_gene, hidden_dim=128):
    """Load the trained VAE model"""
    print(f"Loading VAE from {vae_path}")
    autoencoder = VAE(
        num_genes=num_gene,
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

class LatentSpaceMonitor:
    """Monitor the latent space during diffusion model training using scanpy's UMAP"""
    def __init__(
        self,
        vae_model,
        output_dir,
        visualization_freq=5000,
        sample_size=1000,
        device="cuda",
    ):
        self.vae_model = vae_model
        self.output_dir = output_dir
        self.visualization_freq = visualization_freq
        self.sample_size = sample_size
        self.device = device
        self.step_counter = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Store reference data (from the real dataset)
        self.reference_latents = None
        self.reference_labels = None
    
    def set_reference_data(self, latents, labels=None):
        """Set reference data to compare against generated samples"""
        self.reference_latents = latents
        self.reference_labels = labels
        print(f"Set {len(latents)} reference latent vectors")
    
    def visualize_step(self, diffusion_model, batch_size=64, num_samples=None):
        """Generate and visualize samples at the current step"""
        if self.step_counter % self.visualization_freq != 0:
            self.step_counter += 1
            return
        
        print(f"Creating latent space visualization at step {self.step_counter}...")
        
        # Set number of samples
        if num_samples is None:
            num_samples = min(self.sample_size, 500)  # Limit to 500 samples for speed
        
        # Generate samples
        generated_latents = []
        
        # For training 1 cell type, the label is just the cell type index
        current_labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        
        # Generate samples in batches using the diffusion model
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                # Determine batch size for this iteration
                current_batch_size = min(batch_size, num_samples - i)
                
                # Generate random noise
                noise = torch.randn(current_batch_size, diffusion_model.input_dim).to(self.device)
                
                # Generate samples - handle either p_sample_loop or ddim_sample_loop
                try:
                    sample, _ = diffusion_model.diffusion.p_sample_loop(
                        model=diffusion_model,
                        shape=(current_batch_size, diffusion_model.input_dim),
                        noise=noise,
                        y=current_labels[:current_batch_size],
                    )
                except:
                    # If p_sample_loop doesn't work, try direct model forward pass
                    t = torch.zeros(current_batch_size, dtype=torch.long).to(self.device)  # Starting at t=0
                    sample = diffusion_model(noise, t, current_labels[:current_batch_size])
                
                # Add to list
                generated_latents.append(sample.cpu().numpy())
        
        # Concatenate
        generated_latents = np.concatenate(generated_latents, axis=0)
        
        # Combine with reference data if available
        if self.reference_latents is not None:
            # Subsample reference data to match generated data for balanced visualization
            if len(self.reference_latents) > len(generated_latents):
                indices = np.random.choice(len(self.reference_latents), len(generated_latents), replace=False)
                ref_latents_subset = self.reference_latents[indices]
            else:
                ref_latents_subset = self.reference_latents
            
            combined_latents = np.concatenate([ref_latents_subset, generated_latents], axis=0)
            
            # Create labels for real vs generated
            real_vs_gen = np.concatenate([
                np.ones(len(ref_latents_subset)),  # 1 for real
                np.zeros(len(generated_latents))   # 0 for generated
            ])
        else:
            combined_latents = generated_latents
            real_vs_gen = np.zeros(len(generated_latents))  # All generated
        
        # Create an AnnData object for scanpy
        adata = ad.AnnData(X=combined_latents)
        adata.obs['source'] = ['Real' if x == 1 else 'Generated' for x in real_vs_gen]
        
        # Apply UMAP using scanpy
        print("Applying scanpy UMAP to latent vectors...")
        sc.pp.neighbors(adata, n_neighbors=15, use_rep='X')
        sc.tl.umap(adata)
        
        # Create visualizations
        self._create_plots(adata)
        
        # Increment step counter
        self.step_counter += 1
    
    def _create_plots(self, adata):
        """Create and save UMAP plots using scanpy"""
        # Plot and save
        plt.figure(figsize=(12, 10))
        
        # Plot using scanpy
        sc.pl.umap(
            adata, 
            color='source', 
            title=f'Latent Space UMAP - Step {self.step_counter}',
            show=False,
            legend_loc='on data',
            palette={'Real': 'tab:orange', 'Generated': 'tab:blue'},
            size=50
        )
        
        # Save the figure
        plt.savefig(os.path.join(self.output_dir, f'latent_umap_step_{self.step_counter}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save raw data for future analysis
        adata.write(os.path.join(self.output_dir, f'latent_data_step_{self.step_counter}.h5ad'))
        
        print(f"Saved latent space visualizations for step {self.step_counter}")


class MonitoredTrainLoop(TrainLoop):
    """Extension of TrainLoop with latent space monitoring"""
    def __init__(self, *args, latent_monitor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.latent_monitor = latent_monitor
    
    def run_loop(self):
        """Run training loop with monitoring"""
        step_count = 0
        try:
            while (not self.lr_anneal_steps 
                    or step_count + self.resume_step < self.lr_anneal_steps):
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
    
    # Create directory for UMAP visualizations
    umap_dir = os.path.join(cell_type_save_dir, "umap_visualizations")
    os.makedirs(umap_dir, exist_ok=True)
    
    # Set up distributed training
    dist_util.setup_dist()
    
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
    
    # Create model and diffusion process
    print("Creating model and diffusion...")
    # Ensure num_classes is a list with a single value
    args_dict['num_classes'] = [args_dict['num_classes']]
    model, diffusion = create_model_and_diffusion(**args_dict)
    model.to(dist_util.dev())
    
    # Create schedule sampler
    schedule_sampler = create_named_schedule_sampler(args_dict['schedule_sampler'], diffusion)
    
    # Create a modified data loader that filters for this cell type
    print("Creating filtered data loader...")
    data = load_data(
        data_dir=args_dict['data_dir'],
        batch_size=args_dict['batch_size'],
        vae_path=args_dict['vae_path'],
        train_vae=False,
        cell_type_filter=cell_type_idx  # This parameter filters by cell type
    )
    
    # Setup latent space monitoring
    
    # 1. Load the VAE model
    vae_gene_count = 24622  # Use the expected gene count based on error message
    vae = load_vae(args_dict['vae_path'], vae_gene_count)
    
    # 2. Initialize latent space monitor
    latent_monitor = LatentSpaceMonitor(
        vae_model=vae,
        output_dir=umap_dir,
        visualization_freq=args_dict.get('visualization_freq', 5000),
        device=dist_util.dev()
    )
    
    # 3. Collect some reference data for the monitor
    # We need to get some real latent vectors to compare against
    print("Collecting reference latent vectors...")
    
    # Directly filter dataset for this cell type
    cell_type_adata = adata[cell_type_indices].copy()
    
    # Preprocess 
    sc.pp.normalize_total(cell_type_adata, target_sum=1e4)
    sc.pp.log1p(cell_type_adata)
    
    # Convert to dense and get a subset for reference
    max_ref_samples = min(1000, len(cell_type_adata))
    if len(cell_type_adata) > max_ref_samples:
        # Subsample to avoid using too much memory
        indices = np.random.choice(len(cell_type_adata), max_ref_samples, replace=False)
        cell_type_adata = cell_type_adata[indices].copy()
    
    # Get gene expression data
    cell_data = cell_type_adata.X.toarray() if hasattr(cell_type_adata.X, 'toarray') else cell_type_adata.X
    
    # Filter genes to match VAE input if needed
    if cell_data.shape[1] > vae_gene_count:
        # Calculate variance and keep top genes
        gene_vars = np.var(cell_data, axis=0)
        top_indices = np.argsort(gene_vars)[-vae_gene_count:]
        cell_data = cell_data[:, top_indices]
    
    # Get latent vectors using VAE
    with torch.no_grad():
        # Process in batches to avoid OOM
        latent_vecs = []
        batch_size = 100
        for i in range(0, len(cell_data), batch_size):
            batch = cell_data[i:min(i + batch_size, len(cell_data))]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).cuda()
            latent_batch = vae(batch_tensor, return_latent=True)
            latent_vecs.append(latent_batch.cpu().numpy())
        
        # Concatenate batches
        reference_latents = np.concatenate(latent_vecs, axis=0)
    
    # Set reference data for monitor
    latent_monitor.set_reference_data(reference_latents)
    
    # Train the model with monitoring
    print(f"Starting training for cell type {cell_type_name}...")
    MonitoredTrainLoop(
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
        save_dir=args_dict['save_dir'],
        latent_monitor=latent_monitor
    ).run_loop()
    
    print(f"Completed training for cell type {cell_type_name}")


def create_argparser():
    defaults = dict(
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0001,
        lr_anneal_steps=200000,  # Reduced from 800000 for per-cell-type training
        batch_size=64,  # Reduced batch size as we'll have fewer samples per cell type
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",
        log_interval=1000,
        save_interval=50000,  # Reduced from 400000
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        visualization_freq=5000,  # How often to create UMAP visualization
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
    parser.add_argument("--visualization_freq", type=int, default=5000,
                        help="How often to create UMAP visualizations (in steps)")
    
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
