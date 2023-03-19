import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

# 1. load processed, enumerated sequence alignment
# 2. convert NP array to tensor
# 3. pass tensor to F.one_hot()

class MSA_Dataset(Dataset):
    '''
    Helps manage processed multiple sequence alignment and related data.
    '''

    def __init__(self, enumd_msa_binary: np.ndarray, seq_weight: np.ndarray, seq_keys: np.ndarray):
        '''
        enumd_msa_binary: a two dimensional np.array, the original sequences with letters replaced by number.
                          size: [num_of_sequences, length_of_msa]
        seq_weight: one dimensional array.
                    size: [num_sequences].
                    Weights for sequences in a MSA.
                    The sum of seq_weight has to be equal to 1 when training latent space models using VAE
        seq_keys: name of sequences in MSA
        '''
        super(MSA_Dataset).__init__()
        self.enumd_msa_binary = enumd_msa_binary
        self.seq_weight = seq_weight
        self.seq_keys = seq_keys

    def __len__(self):
        # number of sequences
        assert(self.enumd_msa_binary.shape[0] == len(self.seq_weight))
        assert(self.enumd_msa_binary.shape[0] == len(self.seq_keys))
        return self.enumd_msa_binary.shape[0]

    def __getitem__(self, ind):
        return (self.enumd_msa_binary[ind, :], self.seq_weight[ind], self.seq_keys[ind])
    
    class VAE(nn.Module):
