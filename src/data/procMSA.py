'''
Encode protein sequences into one-hot vectors, pickle for later use
'''

import argparse
import json, pickle
from pathlib import Path
import numpy as np
import pandas as pd
from Bio import AlignIO

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
AAs = ['R', 'H', 'K',
        'D', 'E',
        'S', 'T', 'N', 'Q',
        'C', 'G', 'P',
        'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W']

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [PFAM ID] [FILE] [QUERY SEQ ID]",
        description="PFAM multiple sequence alignment processor: converts protein alignments from Stockholm file format to Numpy int arrays, with the amino acid types enumerated from 1 - 20, and 0 for gaps.",
    )
    parser.add_argument("PFAM_ID", type=str, help="PFAM accession ID of the family.")
    parser.add_argument('MSA_file_path', type=str, help='Path to the input Stockholm sequence alignment file')
    parser.add_argument('query_seq_ID', type=str, help='ID of query sequence used as the reference for the alignment.')
    return parser

def prune_seqs(msa: AlignIO.MultipleSeqAlignment, query_seq_id: str) -> tuple[np.array, np.array]:
    '''
    For meaningful comparison, ignore gaps
    When remove seqs from MSA, also remove corresponding rows in seqs_infos
    '''
    # Since query seq is reference for alignment,
    # omit gaps and corresponding positions in other seqs

    # Find query seq
    query_seq = ""
    query_ind = -1
    for i in range(len(msa)):
        if msa[i].id == query_seq_id:
            print("query seq found in MSA")
            query_seq = msa[i].seq
            query_ind = i
            break

    pruneds = np.array(msa)
    pruneds = np.char.upper(pruneds)
    # keep track of "surviving" seqs by indices
    idx = np.arange(pruneds.shape[0])

    # Omit query gaps
    gap_idx = ~(np.logical_or(
        pruneds[query_ind,:] == '-',
        pruneds[query_ind,:] == '.'
    ))
    pruneds = pruneds[:, gap_idx]
    # print("{} columns".format(pruneds.shape[1]))
    # print("query seq: {}".format(pruneds[query_ind,:]))
    # print(pruneds)

    # Remove sequences with too many gaps
    idx_gapy_seqs = np.count_nonzero((pruneds == '-') | (pruneds == '.'), axis=1) <= 10
    pruneds = pruneds[idx_gapy_seqs, :]
    idx = idx[idx_gapy_seqs]

    # Remove positions with too many gaps
    idx_gapy_poss = np.count_nonzero((pruneds == '-') | (pruneds == '.'), axis=0) <= 0.2 * pruneds.shape[0]
    pruneds = pruneds[:, idx_gapy_poss]

    return pruneds, idx

def enum_seqs(lettMSA: np.array, query_seq_id: str) -> np.array:
    '''
    Convert letter representation of protein sequences to
    numbering of amino acids in given MSA, save as numpy array.
    Takes Stockholm format MSAs, pfam sequence ID.
    '''
    # amino acids from 1 - 20, 0 for gaps
    aas = np.array(AAs)
    np.append(aas, ('-','.'))
    # print(aas)
    enums = np.arange(1, len(AAs) + 1)
    np.append(aas, (0,0))

    # Replace aa letters in MSA with numbers,
    # credits to https://stackoverflow.com/q/55949809
    sorted_inds = aas.argsort()
    # print("sorted inds: ", sorted_inds)
    # print("original aas: ", aas)
    aas = aas[sorted_inds]
    # print("aas[sorted_inds]: ", aas)
    enums = enums[sorted_inds]

    inds = np.searchsorted(aas, lettMSA.ravel()).reshape(lettMSA.shape)
    '''
    # ignore any vals(letters) beyond a.a.'s
    inds[inds == len(aas)] = 0
    mask = aas[inds] == lettMSA
    return np.where(mask, enums[inds], 0)
    '''
    return enums[inds]

def proc_MSA(MSA_file_name: str, query_seq_id: str) -> tuple[np.array, pd.DataFrame]:
    '''
    Encode an MSA into an enumeration of the amino acids,
    return as a NumPy array of dimensions
    (# seqs) x (aligned length of each seq).
    Takes Stockholm format MSAs, pfam sequence ID.
    '''
    # Read MSA
    msa = AlignIO.read(MSA_file_name, "stockholm")
    procd_seqs, survivor_seqs_idx = prune_seqs(msa, query_seq_id)
    procd_seqs = enum_seqs(procd_seqs, query_seq_id)
    return procd_seqs, survivor_seqs_idx

def calc_seq_weights(msa: np.array) -> np.array:
    '''
    Calculate weights to weigh each
    sequence in MSA by, based on number of
    matches in each column.
    One weight per sequence =>
    returns a column vector of length
    equal to number of sequences.
    '''
    # TODO: perhaps look into [using np.vectorize()](https://stackoverflow.com/a/58478502/10855624) for faster implementation
    weights_mtx = np.empty(msa.shape)
    # for each column
    for j in range(msa.shape[1]):
        aa_types, aa_counts = np.unique(msa[:,j], return_counts=True)
        inv_num_types = 1.0 / len(aa_types)
        # For quick lookup of corresponding aa
        counts_inds = {}
        for aa in aa_types:
            # np.unique() returns sorted unique array, need to get corresponding index
            unique_aatypes_ind = aa_types.searchsorted(aa)
            counts_inds[aa] = aa_counts[unique_aatypes_ind]
        for i in range(msa.shape[0]):
            # freq of seq i's aa in column j
            weights_mtx[i,j] = inv_num_types * (1.0 / counts_inds[msa[i,j]])
    # sum across columns within each sequence
    seq_weight_tots = np.sum(weights_mtx, axis=1)
    return (1.0/np.sum(seq_weight_tots) * seq_weight_tots)

def get_seqs_infos_idx(msa: AlignIO.MultipleSeqAlignment, seq_idx: np.array) -> pd.DataFrame:
    """
    returns sequence IDs, names and "source" species name for
    each sequence of index in the MSA specified in seqs_idx,
    which is a 1D vector of indices.
    """
    data = {"id": [], "name": [], "description": []}
    for npi in seq_idx:
        i = int(npi)
        data["id"].append(msa[i].id)
        data["name"].append(msa[i].name)
        data["description"].append(msa[i].description)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    parser = init_argparse()
    args = parser.parse_args()
    print(args)

    fam_id = args.PFAM_ID
    msas_dir = DATA_DIR / "external" / "MSA"

    enumd_mtx, survive_idx = proc_MSA(Path(args.MSA_file_path), args.query_seq_ID)
    print("enumerated alignment: ", enumd_mtx)
    with open(DATA_DIR / "processed" / "enumd_mtx_{}.npy".format(fam_id), "wb") as f:
        np.save(f, enumd_mtx)

    seq_weights = calc_seq_weights(enumd_mtx)
    print("weights: ", seq_weights)
    with open(DATA_DIR / "processed" / "seq_weights_{}.npy".format(fam_id), "wb") as f:
        np.save(f, seq_weights)

    seqs_infos = get_seqs_infos_idx(AlignIO.read(Path(args.MSA_file_path), "stockholm"), survive_idx)
    print("sequences metadata: ", seqs_infos)
    seqs_infos.to_csv(DATA_DIR / "processed" / "seq_infos_{}.csv".format(fam_id))

    print("Final alignment num sequences: {}, length: {}".format(enumd_mtx.shape[0], enumd_mtx.shape[1]))