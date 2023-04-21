import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset

class MSA_to_OneHot(object):
    """
    Convert a protein multiple sequence alignment to a one-hot encoding.
    """

    def __init__(self, n_aa: int = 20):
        self.n_aa = n_aa

    def __call__(self, msa: np.ndarray) -> torch.Tensor:
        """
        Convert a protein multiple sequence alignment to a one-hot encoding.

        Parameters
        ----------
        msa : np.ndarray
            A multiple sequence alignment of protein sequences.

        Returns
        -------
        torch.Tensor
            A one-hot encoding of the multiple sequence alignment.
        """
        # should still be a 2D matrix, with each row being a sequence, each aa represented by a number 0-20
        assert(msa.ndim == 2)
        return F.one_hot(torch.from_numpy(msa), num_classes=self.n_aa)

# 1. load processed, enumerated sequence alignment
# 2. convert NP array to tensor
# 3. pass tensor to F.one_hot()

class MSA_Dataset(Dataset):
    '''
    Helps manage processed multiple sequence alignment and related data.
    '''

    def __init__(self, enumd_msa_binary: np.ndarray, seq_weight: np.ndarray, seq_keys: np.ndarray, transform=MSA_to_OneHot):
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
        self.transform = transform

    def __len__(self):
        # number of sequences
        assert(self.enumd_msa_binary.shape[0] == self.seq_weight.shape[0])
        return self.enumd_msa_binary.shape[0]

    def __getitem__(self, ind):
        """
        Returns a tuple of (sequence, weight, key)
        sequence is a (# of amino acids) x (# of amino acid types) tensor,
        """
        return (self.enumd_msa_binary[ind, :], self.seq_weight[ind], self.seq_keys[ind])
    
class VAE(nn.Module):
    def __init__(self, n_aa_type: int, dim_latent: int, dim_msa: int, layers_n_hiddens: list[int]):
        super().__init__()
        # # of amino acid types
        self.n_aa_type = n_aa_type
        # dimension of latent space
        self.dim_latent = dim_latent
        # dimension of processed, enumerated representation of multiple sequence alignment
        # will flatten into a long vector
        if isinstance(dim_msa, int):
            self.dim_msa = dim_msa
        elif isinstance(dim_msa, np.ndarray):
            self.dim_msa = dim_msa.prod()
        elif isinstance(dim_msa, tuple):
            self.dim_msa = np.prod(dim_msa)
        else:
            print("dim_msa must be an int, np.ndarray, or tuple")
        # a list of ints, #s of hidden neurons in each layer of encoder and decoder networks
        self.layers_n_hiddens = layers_n_hiddens

        hidden_layers = []
        for i in range(1, len(self.layers_n_hiddens)):
            # repeating layers of (linear ==> tanh)
            hidden_layers.append(nn.Sequential(
                nn.Linear(self.layers_n_hiddens[i-1], self.layers_n_hiddens[i]),
                nn.Tanh()
            ))

        self.encoder_comm_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(self.dim_latent, self.layers_n_hiddens[0]),
            nn.Tanh()
        )])
        self.encoder_comm_layers.extend(hidden_layers)
        self.encoder_comm_layers = nn.Sequential(*self.encoder_comm_layers)
        # leave last layer because need to split up for mean, vars respectively
        self.enc_mu = nn.Linear(self.layers_n_hiddens[-1], self.dim_latent)
        self.enc_logvars = nn.Linear(self.layers_n_hiddens[-1], self.dim_latent)

        decoder = nn.ModuleList([nn.Sequential(
            nn.Linear(self.dim_latent, self.layers_n_hiddens[0]),
            nn.Tanh()
        )])
        decoder.extend(hidden_layers)
        decoder.append(nn.Sequential(
            nn.Linear(self.layers_n_hiddens[-1], self.dim_msa)
        ))
        decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor):
        '''
        x is flattened one-hot tensor encoding of a multiple sequence alignment, projects into latent space
        returns parameters of the latent distribution, set as a normal distribution
        '''
        res = self.encoder_comm_layers(x)
        mu = self.enc_mu(res)
        vars = torch.exp(self.enc_logvars(res))
        return (mu, vars)

    def decode(self, z: torch.Tensor):
        '''
        z is a sample from latent space, projects back into multiple sequence alignment space
        '''
        # output of last decoder linear layer
        out = self.decoder(z)

        # "unflatten" back into 3D
        # for each sequence,
        #   for each position,
        #     a vector, the prob distrib over all 20 a.a. types,
        #     of length # of amino acid types
        n_seqs = out.shape[0]
        # don't need to explicitly specify alignment length, will infer, so -1
        out = out.view(n_seqs-1, -1, self.n_aa_type)
        log_ps = F.log_softmax(out, dim=-1)
        return log_ps
        
