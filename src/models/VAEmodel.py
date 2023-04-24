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

    def __init__(self, n_aa: int = 21):
        self.n_aa = n_aa

    def __call__(self, enumd_seq: np.ndarray) -> torch.Tensor:
        """
        Convert a protein multiple sequence alignment to a one-hot encoding.

        Parameters
        ----------
        enumd_seq : np.ndarray
            One sequence in an enumd_seq of protein sequences.

        Returns
        -------
        torch.Tensor
            A one-hot encoding of the input.
        """
        # print("before transform shape:", enumd_seq.shape)
        print("before transform:", enumd_seq)
        assert(enumd_seq.ndim == 1)
        # +1 to include '0', which represents alignment gaps
        return F.one_hot(torch.from_numpy(enumd_seq), num_classes=(self.n_aa))

# 1. load processed, enumerated sequence alignment
# 2. convert NP array to tensor
# 3. pass tensor to F.one_hot()

class MSA_Dataset(Dataset):
    '''
    Helps manage processed multiple sequence alignment and related data.
    '''

    def __init__(self, enumd_msa: np.ndarray, seq_weight: np.ndarray, seqIDs: np.ndarray, transform=MSA_to_OneHot):
        '''
        enumd_msa: a two dimensional np.array, the original sequences with letters replaced by number.
                          size: [num_of_sequences, length_of_msa]
        seq_weight: one dimensional array.
                    size: [num_sequences].
                    Weights for sequences in a MSA.
                    The sum of seq_weight has to be equal to 1 when training latent space models using VAE
        seqIDs: name of sequences in MSA
        '''
        super(MSA_Dataset).__init__()
        self.enumd_msa = enumd_msa
        self.seq_weight = seq_weight
        self.seqIDs = seqIDs
        self.transform = transform()

    def __len__(self):
        # number of sequences
        assert(self.enumd_msa.shape[0] == self.seq_weight.shape[0])
        return self.enumd_msa.shape[0]

    def __getitem__(self, ind):
        """
        Returns a tuple of (sequence, weight, seq_name)
        sequence is a (# of amino acids) x (# of amino acid types) tensor,
        """
        seq = self.enumd_msa[ind, :]
        weight = self.seq_weight[ind]

        if self.transform:
            print("applying transform: ", self.transform)
            seq = self.transform(seq)

        seq = seq.to(torch.float32).flatten()
        # return (seq, weight, self.seqIDs[ ])
        return (seq, weight)
    
class VAE(nn.Module):
    def __init__(self, n_aa_type: int, dim_latent: int, dim_seq_in: int, layers_n_hiddens: "list[int]"):
        super().__init__()
        # # of amino acid types
        self.n_aa_type = n_aa_type
        # dimension of latent space
        self.dim_latent = dim_latent
        # dimension of processed, enumerated representation of multiple sequence alignment
        # will flatten into a long vector
        if isinstance(dim_seq_in, int):
            self.dim_seq_in = dim_seq_in
        elif isinstance(dim_seq_in, np.ndarray):
            self.dim_seq_in = dim_seq_in.prod()
        elif isinstance(dim_seq_in, tuple):
            self.dim_seq_in = np.prod(dim_seq_in)
        else:
            print("dim_seq_in must be an int, np.ndarray, or tuple")
        # a list of ints, #s of hidden neurons in each layer of encoder and decoder networks
        self.layers_n_hiddens = layers_n_hiddens

        """
        Set up layers
        """
        hidden_layers = []
        for i in range(1, len(self.layers_n_hiddens)):
            # repeating layers of (linear ==> tanh)
            hidden_layers.append(nn.Sequential(
                nn.Linear(self.layers_n_hiddens[i-1], self.layers_n_hiddens[i]),
                nn.Tanh()
            ))

        self.encoder_comm_layers = nn.ModuleList([nn.Sequential(
            nn.Linear(self.dim_seq_in, self.layers_n_hiddens[0]),
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
            nn.Linear(self.layers_n_hiddens[-1], self.dim_seq_in)
        ))
        decoder = nn.Sequential(*decoder)

        """
        For convenient epsilon (noise) sampling
        """
        self.noiseN = torch.distributions.Normal(0,1)
        if torch.cuda.is_available():
            self.noiseN.loc = self.noiseN.loc.cuda()
            self.noiseN.scale = self.noiseN.scale.cuda()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        '''
        x is flattened one-hot tensor encoding of a sequence, projects into latent space
        returns parameters of the latent distribution, set as a normal distribution
        '''
        res = self.encoder_comm_layers(x)
        mu = self.enc_mu(res)
        vars = torch.exp(self.enc_logvars(res))
        return (mu, vars)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        '''
        z is a sample from latent space, projects back into multiple sequence alignment space
        '''
        # output of last decoder linear layer
        out = self.decoder(z)

        # "unflatten" back into 3D
        # for each sequence,
        #   for each position,
        #     a vector, the prob distrib over all (20 a.a. types + 1 gap "type"),
        #     of length # of amino acid types
        n_seqs = out.shape[0]
        # don't need to explicitly specify alignment length, will infer, so -1
        out = out.view(n_seqs-1, -1, (self.n_aa_type))
        log_ps = F.log_softmax(out, dim=-1)
        return log_ps
        
    def sample_latent(self, mu: torch.Tensor, vars: torch.Tensor) -> torch.Tensor:
        '''
        returns a sample from the diagonal normal prior p(z) with mean mu and covariance matrix (vars)I
        '''
        eps = self.noiseN.sample(mu.shape)
        z = mu + vars*eps
        return z

    def calc_weighted_elbo(self, x: torch.Tensor, seq_weights: torch.Tensor) -> torch.float32:
        """
        returns the evidence lower bounded for a single example, MSA sequence x,
        weighted by the seq weight for x in the MSA
        """
        # project into latent space
        (mu, vars) = self.encode(x)
        # _variational_ autoencoder => _sample_ a latent from the projection _distribution_
        z = self.sample_latent(mu, vars)
        # project "back" into seqs space, except probs over possib a.a.'s for all pos's
        log_ps = self.decoder(z)

        # calculate log p(x|z)
        # element-wise prod on one-hot encoding with
        # log p(each aa type in one-hot encoding)
        # = p(the real aa type, the single 1 in x_i, for each pos)
        log_PxIz = torch.sum(x*log_ps, -1)

        # for two diagonal normals
        # keep each seq pos separate before weighing each with seq weight
        kl_div = torch.sum(0.5*(vars**2 + mu**2 - 2*torch.log(vars) - 1), -1)

        seq_weights /= torch.sum(seq_weights)
        return torch.sum(seq_weights * (log_PxIz - kl_div))
