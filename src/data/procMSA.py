'''
Encode protein sequences into one-hot vectors, pickle for later use
'''

import json, pickle
from pathlib import Path
import numpy as np
from Bio import AlignIO

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
AAs = ['R', 'H', 'K',
        'D', 'E',
        'S', 'T', 'N', 'Q',
        'C', 'G', 'P',
        'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']

def prune_seqs(msa: AlignIO.MultipleSeqAlignment, query_seq_id: str) -> np.array:
    '''
    For meaningful comparison, ignore gaps
    '''

    # Since query seq is reference for alignment,
    # omit gaps and corresponding positions in other seqs

    # Find query seq
    query_seq = ""
    query_ind = -1
    for i in range(len(msa)):
        if msa[i].id == query_seq_id:
            print("a match")
            query_seq = msa[i].seq
            query_ind = i
            break
    pruneds = np.array(msa)
    pruneds = np.char.upper(pruneds)
    # print(
    #     ~(np.logical_or(
    #             pruneds[query_ind,:] == '-',
    #             pruneds[query_ind,:] == '.'
    #         ))
    # )
    # Omit query gaps
    pruneds = pruneds[:, 
        ~(np.logical_or(
            pruneds[query_ind,:] == '-',
            pruneds[query_ind,:] == '.'
        ))
    ]
    # print("{} columns".format(pruneds.shape[1]))
    # print("query seq: {}".format(pruneds[query_ind,:]))
    # print(pruneds)

    # Remove sequences with too many gaps
    pruneds = pruneds[
        np.count_nonzero((pruneds == '-') | (pruneds == '.'), axis=1)
        <= 10,
    :]

    # Remove positions with too many gaps
    pruneds = pruneds[:,
        np.count_nonzero((pruneds == '-') | (pruneds == '.'), axis=0)
        <= 0.2 * pruneds.shape[0],
    ]

    return pruneds

def enum_seqs(lettMSA: np.array, query_seq_id: str) -> np.array:
    '''
    Convert letter representation of protein sequences to
    numbering of amino acids in given MSA, save as numpy array.
    Takes Stockholm format MSAs, pfam sequence ID.
    '''
    # amino acids from 1 - 20, 0 for gaps
    aas = np.array(AAs)
    np.append(aas, ('-','.'))
    print(aas)
    enums = np.arange(1, len(AAs) + 1)
    np.append(aas, (0,0))

    # Replace aa letters in MSA with numbers,
    # credits to https://stackoverflow.com/q/55949809
    sorted_inds = aas.argsort()
    aas = aas[sorted_inds]
    enums = enums[sorted_inds]

    inds = np.searchsorted(aas, lettMSA.ravel()).reshape(lettMSA.shape)
    '''
    # ignore any vals(letters) beyond a.a.'s
    inds[inds == len(aas)] = 0
    mask = aas[inds] == lettMSA
    return np.where(mask, enums[inds], 0)
    '''
    return enums[inds]

def proc_MSA(MSA_file_name: str, query_seq_id: str) -> np.array:
    '''
    Encode an MSA into an enumeration of the amino acids,
    return as a NumPy array of dimensions
    (# seqs) x (aligned length of each seq).
    Takes Stockholm format MSAs, pfam sequence ID.
    '''
    # Read MSA
    msa = AlignIO.read(MSA_file_name, "stockholm")
    procd_seqs = prune_seqs(msa, query_seq_id)
    procd_seqs = enum_seqs(procd_seqs, query_seq_id)
    return procd_seqs

def calc_seq_weights(msa: np.array) -> np.array:
    '''
    Calculate weights to weigh each
    sequence in MSA by, based on number of
    matches in each column.
    One weight per sequence =>
    returns a column vector of length
    equal to number of sequences.
    '''
    
    weights = np.empty(msa.shape)
    # for each column
    for j in range(msa.shape[1]):
        aa_type, aa_counts = np.unique(msa, return_counts=True)
        # For quick lookup of corresponding aa
        counts_inds = {}
        for i in range(msa.shape[0]):
            counts_inds[i] = aa_counts[i]

if __name__ == "__main__":
    fam_id = "PF00041"
    msas_dir = DATA_DIR / "external" / "MSA"
    enumd_mtx = proc_MSA(msas_dir / "{}.sth".format(fam_id), "A0A6P7LW62_BETSP/432-517")
    print(enumd_mtx)