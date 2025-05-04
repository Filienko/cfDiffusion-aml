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
):
    """
    Load data with memory optimization and gene count handling
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    
    # Load and preprocess data
    print("Loading h5ad file...")
    adata = sc.read_h5ad(data_dir)
    adata.var_names_make_unique()
    classes = adata.obs['cell_type'].values
    print('#####################\n',np.unique(classes).shape[0],'\n#####################')
    
    label_encoder = LabelEncoder()
    labels = classes
    label_encoder.fit(labels)
    classes = label_encoder.transform(labels)
    print(label_encoder.classes_)
    
    print("BEFORE", adata.shape)
    #sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=10)
    adata.var_names_make_unique()
    print("AFTER", adata.shape)
    
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
            #print(f"Processing chunk {chunk_num}/{total_chunks} (cells {i} to {end_idx-1})...")
            
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
            
            # Print memory usage periodically
            if chunk_num % 10 == 0 and False:
                if torch.cuda.is_available():
                    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                    print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    print("Concatenating processed chunks...")
    cell_data = np.concatenate(processed_data, axis=0)
    print(f"Final processed data shape: {cell_data.shape}")
    
    # Consider sampling a subset of the data if it's too large
    # This is optional but can help if you're hitting memory limits
    max_cells = 100000  # Adjust based on your memory constraints
    if len(cell_data) > max_cells:
        print(f"Sampling {max_cells} cells from {len(cell_data)} total cells...")
        indices = np.random.choice(len(cell_data), max_cells, replace=False)
        cell_data = cell_data[indices]
        classes = classes[indices]
        print(f"Data shape after sampling: {cell_data.shape}")
    
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
    
    # Use a more robust generator approach
    def data_generator():
        while True:
            for batch in loader:
                yield batch
    
    return data_generator()

'''
def load_data(
    *,
    data_dir,
    batch_size,
    vae_path=None,
    deterministic=False,
    train_vae=False,
    hidden_dim=128,
):
    """
    For a dataset, create a generator over (cells, kwargs) pairs.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param vae_path: the path to save autoencoder / read autoencoder checkpoint.
    :param deterministic: if True, yield results in a deterministic order.
    :param train_vae: train the autoencoder or use the autoencoder.
    :param hidden_dim: the dimensions of latent space. If use pretrained weight, set 128
    """
    if not data_dir:
        raise ValueError("unspecified data directory")

    adata = sc.read_h5ad(data_dir)
    adata.var_names_make_unique()  # has been process by the SCimilarity code base. No need to filter cells and genes
    print("BEFORE", adata.shape)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.filter_cells(adata, min_genes=10)
    adata.var_names_make_unique()
    print("AFTER", adata.shape)

    # filter spleen macrophage cell
    #selected_cells = (adata.obs['organ_tissue'] != 'Spleen') | (adata.obs['free_annotation'] != 'macrophage')  
    #adata = adata[selected_cells, :]  
    
    # filter Thymus memory b cell
    #selected_cells = (adata.obs['organ_tissue'] != 'Thymus') | (adata.obs['free_annotation'] != 'memory b cell')  
    #adata = adata[selected_cells, :]  

    # classes = adata.obs['organ_tissue'].values
    classes = adata.obs['cell_type'].values
    print('#####################\n',np.unique(classes).shape[0],'\n#####################')
    
    label_encoder = LabelEncoder()
    labels = classes
    label_encoder.fit(labels)
    classes = label_encoder.transform(labels)
    print(label_encoder.classes_)
    

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    cell_data = adata.X.toarray()
    print('#####################\n',cell_data.shape,'\n#####################')

    # if not train autoencoder
    if not train_vae:
        num_gene = cell_data.shape[1]
        autoencoder = load_VAE(vae_path,num_gene,hidden_dim)
        cell_data = autoencoder(torch.tensor(cell_data).cuda(),return_latent=True)
        cell_data = cell_data.cpu().detach().numpy()
    
    dataset = CellDataset(
        cell_data,
        classes
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader
'''
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
        # out_dict = {}
        # if self.class_name is not None:
        #     out_dict["y"] = np.array(self.class_name[idx], dtype=np.int64)
        y = np.array(self.class_name[idx], dtype=np.int64)
        return arr, y
