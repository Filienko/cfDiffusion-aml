import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from scipy import stats
import torch
import sys
sys.path.append('..')
from VAE.VAE_model import VAE
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def load_VAE():
    autoencoder = VAE(
        num_genes=19423, 
        device='cuda',
        seed=0,
        loss_ae='mse',
        hidden_dim=128,
        decoder_activation='ReLU',
    )
    autoencoder.load_state_dict(torch.load('/home/daniilf/cfDiffusion-aml/checkpoint/VAE/camda/model_seed=0_step=100000.pt'))
    return autoencoder


adata = sc.read_h5ad('/home/daniilf/scDiffusion/data/onek1k_annotated_train_release_filtered_50k.h5ad')
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

gene_names = adata.var_names

celltype = adata.obs['cell_type']
cell_data = adata.X
cell_data.shape

print("unique types",np.unique(adata.obs['cell_type'].values))
'''
cell_gen_all = []
gen_class = []
cato = ['D0', 'D0.5', 'D1', 'D1.5', 'D2', 'D2.5', 'D3', 'D4.5', 'D5', 'D5.5', 'D6', 'D6.5',
        'D7', 'D7.5', 'D8']

index2 = [i for i in range(len(cato))]
for i in index2:
    npyfile=np.load(f'/home/workplace/cfDiffusion/generation/wot/cell{i}_cache5_non_uniform.npy',allow_pickle=True)
    length = min(adata[adata.obs['period']==cato[i]].X.shape[0],9000)
    print(length)
    cell_gen_all.append(npyfile[:int(length)])#.squeeze(1)
    
    gen_class+=['generation '+cato[i]]*int(length)

cell_gen_all = np.concatenate(cell_gen_all,axis=0)

autoencoder = load_VAE()
cell_gen_all = autoencoder(torch.tensor(cell_gen_all).cuda(),return_decoded=True).cpu().detach().numpy()

cell_gen = cell_gen_all
print("AFTER VAE", cell_gen.shape)
'''

#####################################################################################
cell_gen_all = []
gen_class = []
#cato = ['0', '1', '2', '3', '4', '5', '6', '8', '9', '10', '12', '13', '14', '15']
cato = [0,1,2,3,4,5,6,8,9,10,12,13,14,15]
index2 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13]
#index2 = [0]
'''
for i in index2:
    npzfile=np.load(f'/home/daniilf/scDiffusion/output/camda_hvg/cell{i}_cache5_non_uniform.npy',allow_pickle=True)
    length = min(adata[adata.obs['cell_type']==cato[i]].X.shape[0],6000)
    #print(adata[adata.obs['cell_type']==cato[i]])
    cell_gen_all.append(npzfile['cell_gen'][:int(length)])#.squeeze(1)

    gen_class+=['gen '+str(cato[i])]*int(length)
'''

for i in index2:
    npyfile = np.load(f'/home/daniilf/cfDiffusion/generation/camda/cell{i}_cache5_non_uniform.npy', allow_pickle=True)
    gen_data = npyfile['cell_gen']
    # Determine how many samples to take
    length = min(adata[adata.obs['cell_type']==cato[i]].X.shape[0], 9000, gen_data.shape[0])
    print(f"Using {length} samples for cell type {cato[i]}")
    # Add exactly length samples
    cell_gen_all.append(gen_data[:int(length)])
    # Add exactly the same number of labels
    gen_class += ['gen '+str(cato[i])] * int(length)
    # Verify counts match for this category
    print(f"Category {cato[i]}: Added {length} samples with matching labels")

cell_gen_all = np.concatenate(cell_gen_all,axis=0)

autoencoder = load_VAE()
cell_gen_all = autoencoder(torch.tensor(cell_gen_all).cuda(),return_decoded=True).cpu().detach().numpy()

cell_gen = cell_gen_all
print("cell_data.shape:", cell_data.shape)
print("cell_gen.shape:", cell_gen.shape)
print("celltype.shape:", len(celltype))
print("gen_class length:", len(gen_class))


adata = np.concatenate((cell_data, cell_gen),axis=0)
adata = ad.AnnData(adata, dtype=np.float32)

# if conditional generate. other wise commented this.
adata.obs['cell_type'] = np.concatenate((celltype, gen_class))

adata.obs['cell_name'] = [f"true_Cell" for i in range(cell_data.shape[0])]+[f"gen_Cell" for i in range(cell_gen.shape[0])]


# the data is already log norm
#sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
#adata.raw = adata
#adata = adata[:, adata.var.highly_variable]
print("adata.obs['cell_name']", adata)
print("adata.obs['cell_name']", adata.obs['cell_name'].value_counts())
#exit()
sc.pp.scale(adata)
sc.tl.pca(adata, svd_solver='arpack')


sc.pp.neighbors(adata, n_neighbors=10, n_pcs=20)
sc.tl.umap(adata)
sc.pl.umap(
    adata,
    color="cell_name",
    size=6,
    title='tabular camda',
    save="_camda_hvg.png"  # this will save as 'figures/umap_tabular_muris.png' by default
)



for category in cato:
    color_dict = {}
    for cat in adata.obs['cell_type'].cat.categories:
        if cat == category:
            color_dict[cat] = 'tab:orange'
        elif cat == 'gen ' + str(category):
            color_dict[cat] = 'tab:blue'
        else:
            color_dict[cat] = 'black'

    # Generate UMAP without showing
    sc.pl.umap(
        adata=adata,
        color="cell_type",
        groups=[str(category), 'gen ' + str(category)],
        size=8,
        palette=color_dict,
        show=False
    )

    plt.legend(loc='upper right')
    plt.title("Cell Type "+str(category))

    # Save the figure
    plt.savefig(f"/home/daniilf/scDiffusion/output/figure/umap_{str(category)}.png", bbox_inches='tight')
    plt.close()
