import numpy as np
from torch.utils.data import DataLoader, Dataset

import scanpy as sc
import torch
import sys
sys.path.append('..')
from VAE.VAE_model import VAE
from sklearn.preprocessing import LabelEncoder


def load_VAE(vae_path, num_gene, hidden_dim):
    autoencoder = VAE(
        num_genes=num_gene,
        device='cuda',
        seed=0,
        loss_ae='mse',
        hidden_dim=hidden_dim,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load(vae_path))
    return autoencoder

def load_data(
    *,
    data_dir,
    batch_size,
    vae_path=None,
    deterministic=False,
    train_vae=False,
    hidden_dim=128,
    cell_type_filter=None  # New parameter to filter by cell type
):
    """
    Load data with memory optimization and gene count handling,
    with optional filtering by cell type.
    
    Parameters:
        cell_type_filter: If provided, only return cells of this type (index)
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    # Load and preprocess data
    print("Loading h5ad file...")
    adata = sc.read_h5ad(data_dir)
    adata.var_names_make_unique()
    
    # Get cell type labels
    classes = adata.obs['cell_type'].values
    print('#####################\n',np.unique(classes).shape[0],'\n#####################')
    
    label_encoder = LabelEncoder()
    labels = classes
    label_encoder.fit(labels)
    classes = label_encoder.transform(labels)
    print(label_encoder.classes_)
    
    # Apply cell type filtering if specified
    if cell_type_filter is not None:
        filter_mask = classes == cell_type_filter
        if np.sum(filter_mask) == 0:
            raise ValueError(f"No cells found for cell type index {cell_type_filter}")
        
        print(f"Filtering to {np.sum(filter_mask)} cells of type {cell_type_filter}")
        adata = adata[filter_mask].copy()
        classes = classes[filter_mask]
    
    print("BEFORE filtering genes and cells:", adata.shape)
    sc.pp.filter_cells(adata, min_genes=10)
    adata.var_names_make_unique()
    print("AFTER filtering:", adata.shape)
    
    # Standard preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    print("Converting to dense array...")
    cell_data = adata.X.toarray()
    print('#####################\n',cell_data.shape,'\n#####################')
    
    # Process through VAE if needed
    if not train_vae:
        print("Processing through VAE...")
        num_gene = cell_data.shape[1]
        autoencoder = load_VAE(vae_path, num_gene, hidden_dim)
        
        # Move the autoencoder to GPU once
        autoencoder.to('cuda')
        
        # Use smaller chunks - 100 cells at a time
        chunk_size = 100
        total_chunks = (len(cell_data) + chunk_size - 1) // chunk_size
        processed_data = []
        
        for i in range(0, len(cell_data), chunk_size):
            chunk_num = i // chunk_size + 1
            end_idx = min(i + chunk_size, len(cell_data))
            
            chunk = cell_data[i:end_idx]
            
            # Move to GPU, process, and immediately move back to CPU
            with torch.no_grad():
                chunk_tensor = torch.tensor(chunk, dtype=torch.float32).cuda()
                # Use the forward method directly with return_latent=True
                chunk_processed = autoencoder(chunk_tensor, return_latent=True)
                processed_chunk = chunk_processed.cpu().numpy()
                
                # Add to results and delete variables to free memory
                processed_data.append(processed_chunk)
                del chunk_tensor, chunk_processed
            
            # Explicitly clear GPU cache
            torch.cuda.empty_cache()
    
        print("Concatenating processed chunks...")
        cell_data = np.concatenate(processed_data, axis=0)
        print(f"Final processed data shape: {cell_data.shape}")
    
    # Create dataset
    print("Creating dataset...")
    dataset = CellDataset(
        cell_data,
        classes
    )
    
    print(f"Creating DataLoader with batch size {batch_size}...")
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
        )
    
    # Use a generator approach to yield batches continuously
    def data_generator():
        while True:
            for batch in loader:
                yield batch
    
    return data_generator()


class CellDataset(Dataset):
    def __init__(
        self,
        cell_data,
        class_name
    ):
        super().__init__()
        self.data = cell_data
        self.class_name = class_name

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        arr = self.data[idx]
        y = np.array(self.class_name[idx], dtype=np.int64)
        return arr, y
