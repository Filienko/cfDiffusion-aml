"""
Generate scRNA-seq gene expression samples from cell type-specific diffusion models
and save them as numpy arrays.
"""
import argparse
import time
import os
import numpy as np
import torch as th
import torch.distributed as dist
import random
import scanpy as sc
from sklearn.preprocessing import LabelEncoder

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (   
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
)

def save_data(all_cells, data_dir):
    """
    Save generated cells to a numpy file
    """
    np.save(data_dir, all_cells)
    return

def setup_seed(seed):
    """
    Set random seeds for reproducibility
    """
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    th.backends.cudnn.deterministic = True

def generate_cells(cell_type, args_dict):
    """
    Generate cells for a specific cell type using the corresponding trained model
    
    Args:
        cell_type: The cell type index
        args_dict: Dictionary of arguments for generation
    
    Returns:
        Elapsed time for generation
    """
    # Create model directory path for this cell type
    model_dir = args_dict['model_path']
    model_subdir = f"{args_dict['model_base_name']}_{cell_type}"
    model_file = os.path.join(model_dir, model_subdir, f"model{args_dict['model_step']:06d}.pt")
    
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        print(f"Skipping cell type {cell_type}")
        return 0
    
    # Setup distributed training
    dist_util.setup_dist()
    
    print(f"Creating model and diffusion for cell type {cell_type}...")
    # IMPORTANT FIX: Ensure num_classes is a list with a single value
    args_dict['num_classes'] = [args_dict['num_classes']]
    model, diffusion = create_model_and_diffusion(**args_dict)
    
    print(f"Loading model from {model_file}...")
    model.load_state_dict(
        dist_util.load_state_dict(model_file, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()

    print(f"Sampling cells for type {cell_type}...")
    all_cells = []
    
    # The label for generation is just the cell type
    # For the per-cell-type model, we still use the same encoding
    y = th.tensor([cell_type] * args_dict['batch_size'])
    
    elapse_all = 0.
    while len(all_cells) * args_dict['batch_size'] < args_dict['num_samples']:
        sample_fn = (
            diffusion.p_sample_loop if not args_dict['use_ddim'] else diffusion.ddim_sample_loop
        )
        
        start = time.process_time() 
        sample, _ = sample_fn(
            model,
            (args_dict['batch_size'], args_dict['input_dim']), 
            clip_denoised=args_dict['clip_denoised'],
            y=y,
            start_time=diffusion.betas.shape[0],
        )
        elapse_all = elapse_all + time.process_time() - start

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_cells.extend([sample.cpu().numpy() for sample in gathered_samples])
        print(f"Created {len(all_cells) * args_dict['batch_size']} samples for cell type {cell_type}")

    # Take only the requested number of samples (we might have generated more)
    arr = np.concatenate(all_cells, axis=0)[:args_dict['num_samples']]
    
    # Create output directory if needed
    os.makedirs(args_dict['sample_dir'], exist_ok=True)
    
    # Save the generated cells
    output_file = f"{args_dict['sample_dir']}/cell{cell_type}_cache{args_dict['cache_interval']}_{'non_uniform' if args_dict['non_uniform'] else 'uniform'}"
    save_data(arr, output_file)

    dist.barrier()
    print(f"Sampling complete for cell type {cell_type}")
    return elapse_all

def create_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./checkpoints",
                        help="Directory containing the cell type-specific models")
    parser.add_argument("--model_base_name", type=str, default="per_celltype_diffusion",
                        help="Base name for model directories")
    parser.add_argument("--model_step", type=int, default=200000,
                        help="Step number of the model checkpoint to use")
    parser.add_argument("--sample_dir", type=str, default="./generated_samples",
                        help="Directory to save generated samples")
    parser.add_argument("--cell_types", type=str, default="all",
                        help="Comma-separated list of cell type indices to generate, or 'all'")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the h5ad dataset (needed to enumerate cell types)")
    
    parser.add_argument("--num_classes", type=int, default=14)
    parser.add_argument("--branch", type=int, default=0)
    parser.add_argument("--cache_interval", type=int, default=5)
    parser.add_argument("--non_uniform", action="store_true")
    parser.add_argument("--clip_denoised", action="store_true")
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=100)
    
    args, _ = parser.parse_known_args()
    args_dict = {arg: getattr(args, arg) for arg in vars(args)}
    args_dict.update(model_and_diffusion_defaults())
    return args_dict

def main():
    setup_seed(1234)
    args_dict = create_argparser()
    
    # Load dataset to get cell type information
    print("Loading dataset to enumerate cell types...")
    adata = sc.read_h5ad(args_dict['data_dir'])
    cell_types = adata.obs['cell_type'].unique()
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(adata.obs['cell_type'])
    
    # Determine which cell types to generate
    if args_dict['cell_types'] == 'all':
        cell_type_indices = list(range(len(cell_types)))
    else:
        cell_type_indices = [int(idx) for idx in args_dict['cell_types'].split(',')]
    
    print(f"Will generate samples for {len(cell_type_indices)} cell types:")
    for idx in cell_type_indices:
        if idx < len(cell_types):
            print(f"  {idx}: {cell_types[idx]}")
    
    # Generate samples for each cell type
    total_time = 0
    generation_times = []
    
    for i in cell_type_indices:
        if i < len(cell_types):
            print(f"Generating samples for cell type {i}: {cell_types[i]}")
            elapsed_time = generate_cells(i, args_dict.copy())
            generation_times.append(elapsed_time)
            total_time += elapsed_time
            print(f"Time elapsed for cell type {i}: {elapsed_time:.2f} seconds")
    
    print(f"Generation times per cell type: {generation_times}")
    print(f"Total generation time: {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
