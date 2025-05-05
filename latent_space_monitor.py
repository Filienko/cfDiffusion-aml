"""
Latent space monitoring utility for visualizing embeddings during diffusion model training.
This script can be used to periodically generate UMAP visualizations of the latent space
to help diagnose problems in either the VAE or diffusion model.
"""

import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import UMAP
from datetime import datetime
import scanpy as sc
import anndata as ad
from torch.utils.data import DataLoader, Dataset

class LatentSpaceMonitor:
    def __init__(
        self,
        vae_model,
        output_dir="./latent_visualizations",
        sample_size=1000,
        visualization_freq=5000,  # How often to create visualizations (in steps)
        device="cuda"
    ):
        """
        Initialize the latent space monitor.
        
        Args:
            vae_model: The trained VAE model
            output_dir: Directory to save visualizations
            sample_size: Number of samples to use for visualizations
            visualization_freq: How often to create visualizations (in steps)
            device: Device to use for computations
        """
        self.vae_model = vae_model
        self.output_dir = output_dir
        self.sample_size = sample_size
        self.visualization_freq = visualization_freq
        self.device = device
        self.step_counter = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize UMAP reducer
        self.umap_reducer = UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=42
        )
        
        # Store reference data (from the real dataset)
        self.reference_latents = None
        self.reference_labels = None
        
    def set_reference_data(self, data_loader, max_samples=None):
        """
        Set reference data from the training dataset to compare against generated samples.
        
        Args:
            data_loader: DataLoader for the training dataset
            max_samples: Maximum number of samples to use (if None, use self.sample_size)
        """
        if max_samples is None:
            max_samples = self.sample_size
            
        print(f"Collecting reference latent vectors from dataset...")
        
        # Collect latent vectors and labels
        latents = []
        labels = []
        count = 0
        
        with torch.no_grad():
            for batch, y in data_loader:
                # Get batch size
                batch_size = batch.size(0)
                
                # Check if we have enough samples
                if count + batch_size > max_samples:
                    batch_size = max_samples - count
                    batch = batch[:batch_size]
                    y = y[:batch_size]
                
                # Move to device
                batch = batch.to(self.device)
                
                # If the data is already in latent space, use it directly
                # Otherwise, encode it using the VAE
                if batch.size(1) == self.vae_model.latent_dim:
                    latent_batch = batch
                else:
                    latent_batch = self.vae_model(batch, return_latent=True)
                
                # Add to lists
                latents.append(latent_batch.cpu().numpy())
                labels.append(y.numpy())
                
                # Update count
                count += batch_size
                
                # Check if we have enough samples
                if count >= max_samples:
                    break
        
        # Concatenate
        self.reference_latents = np.concatenate(latents, axis=0)
        self.reference_labels = np.concatenate(labels, axis=0)
        
        print(f"Collected {self.reference_latents.shape[0]} reference latent vectors")
    
    def visualize_step(self, diffusion_model, batch_size=64, num_samples=None, labels=None):
        """
        Generate samples and visualize the latent space at the current step.
        
        Args:
            diffusion_model: The diffusion model being trained
            batch_size: Batch size for generation
            num_samples: Number of samples to generate (if None, use self.sample_size)
            labels: Labels to use for generation (if None, generate diverse samples)
        """
        # Check if it's time to create a visualization
        if self.step_counter % self.visualization_freq != 0:
            self.step_counter += 1
            return
        
        print(f"Creating latent space visualization at step {self.step_counter}...")
        
        # Set number of samples
        if num_samples is None:
            num_samples = self.sample_size
        
        # Generate samples
        generated_latents = []
        generated_labels = []
        
        # Generate samples in batches
        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                # Determine batch size for this iteration
                current_batch_size = min(batch_size, num_samples - i)
                
                # Generate random noise
                noise = torch.randn(current_batch_size, diffusion_model.input_dim).to(self.device)
                
                # Generate labels if needed
                if labels is None:
                    # Generate diverse labels
                    current_labels = torch.randint(0, diffusion_model.num_classes, (current_batch_size,)).to(self.device)
                else:
                    # Use provided labels
                    current_labels = labels[:current_batch_size].to(self.device)
                
                # Generate samples
                sample = diffusion_model.p_sample_loop(
                    noise=noise,
                    y=current_labels
                )[0]
                
                # Add to lists
                generated_latents.append(sample.cpu().numpy())
                generated_labels.append(current_labels.cpu().numpy())
        
        # Concatenate
        generated_latents = np.concatenate(generated_latents, axis=0)
        generated_labels = np.concatenate(generated_labels, axis=0)
        
        # Combine with reference data if available
        if self.reference_latents is not None:
            combined_latents = np.concatenate([self.reference_latents, generated_latents], axis=0)
            combined_labels = np.concatenate([self.reference_labels, generated_labels], axis=0)
            
            # Create labels for real vs generated
            real_vs_gen = np.concatenate([
                np.ones(self.reference_latents.shape[0]),  # 1 for real
                np.zeros(generated_latents.shape[0])       # 0 for generated
            ])
        else:
            combined_latents = generated_latents
            combined_labels = generated_labels
            real_vs_gen = None
        
        # Apply UMAP
        print("Applying UMAP to latent vectors...")
        embedding = self.umap_reducer.fit_transform(combined_latents)
        
        # Create visualizations
        self._create_plots(embedding, combined_labels, real_vs_gen)
        
        # Increment step counter
        self.step_counter += 1
    
    def _create_plots(self, embedding, labels, real_vs_gen=None):
        """
        Create and save UMAP plots.
        
        Args:
            embedding: UMAP embedding
            labels: Labels for each point (cell types)
            real_vs_gen: Binary labels for real vs generated points (if available)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create AnnData object for easy plotting with scanpy
        adata = ad.AnnData(X=embedding)
        adata.obs['cell_type'] = labels
        
        if real_vs_gen is not None:
            adata.obs['real_vs_gen'] = ['Real' if x == 1 else 'Generated' for x in real_vs_gen]
        
        # Set UMAP coordinates
        adata.obsm['X_umap'] = embedding
        
        # Plot and save - cell types
        plt.figure(figsize=(12, 10))
        sc.pl.embedding(
            adata, 
            basis='umap', 
            color='cell_type', 
            title=f'Latent Space UMAP - Step {self.step_counter}',
            show=False
        )
        plt.savefig(os.path.join(self.output_dir, f'latent_umap_celltypes_step_{self.step_counter}.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot and save - real vs generated (if available)
        if real_vs_gen is not None:
            plt.figure(figsize=(12, 10))
            sc.pl.embedding(
                adata, 
                basis='umap', 
                color='real_vs_gen', 
                title=f'Latent Space UMAP (Real vs Generated) - Step {self.step_counter}',
                show=False
            )
            plt.savefig(os.path.join(self.output_dir, f'latent_umap_realvsgen_step_{self.step_counter}.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Save raw data for future analysis
        np.savez(
            os.path.join(self.output_dir, f'latent_data_step_{self.step_counter}.npz'),
            embedding=embedding,
            labels=labels,
            real_vs_gen=real_vs_gen
        )
        
        print(f"Saved latent space visualizations for step {self.step_counter}")
