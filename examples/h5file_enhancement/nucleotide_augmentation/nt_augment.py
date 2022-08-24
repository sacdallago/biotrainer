#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created 2022

@author: Mason Minot
"""
import pandas as pd
import numpy as np
import random
import itertools
from math import ceil
import time
from collections import defaultdict

# dictionary of nucleotide codons and their corresponding amino acids
nt_aa_dict = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'N': ['AAT', 'AAC'],
    'D': ['GAT', 'GAC'],
    'C': ['TGT', 'TGC'],
    'Q': ['CAA', 'CAG'],
    'E': ['GAA', 'GAG'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    'H': ['CAT', 'CAC'],
    'I': ['ATT', 'ATC', 'ATA'],
    'L': ['CTT', 'CTC', 'CTA', 'CTG', 'TTA', 'TTG'],
    'K': ['AAA', 'AAG'],
    'M': ['ATG'],
    'F': ['TTT', 'TTC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'W': ['TGG'],
    'Y': ['TAT', 'TAC'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    '*': ['TAA', 'TGA', 'TAG']
}


def aa_to_nt(seqs_and_targets_dict, aug_factor):
    '''
    Function which produces aug_factor unique nucleotide sequences for each amino acid sequence in
    seq_list. The appropriate targets (labels) are maintained for the augmented sequences.

    Parameters
    ----------
    seq_list : list or pandas.core.series.Series
        list or series of amino acid sequences
    target_list : list or pandas.core.series.Series
        list or series of 'targets' or class labels corresponding to the sequences
    aug_factor : int
        the augmentation factor. the number of unique nucleotide sequences to create per protein sequence

    Returns
    -------
    out_df : pandas.core.frame.DataFrame
        pandas dataframe containing augmented nucleotide sequences

    '''
    seq_by_id = {idx: str(sequence) for idx, (sequence, target) in seqs_and_targets_dict.items()}
    target_by_id = {idx: target for idx, (sequence, target) in seqs_and_targets_dict.items()}
    seq_dict = {}
    target_dict = {}
    for key in seqs_and_targets_dict.keys():
        seq_dict[key] = []
        nt_codons_per_residue = {}
        for i in range(len(seq_by_id[key])):
            # determine possible nt codons per aa position
            nt_codons_per_residue[str(i)] = nt_aa_dict[seq_by_id[key][i]]

        # use itertools product function to create a list of all possible combinations of nt codons  for given aa seq
        nucleotides = list(itertools.islice(itertools.product(*nt_codons_per_residue.values()), aug_factor))
        # convert list of tuples to list of strings
        nucleotides = list(map(''.join, nucleotides))
        tmp_target_list = []
        for j in range(len(nucleotides)):
            tmp_target_list.append(target_by_id[key])
        seq_dict[key] = (nucleotides)
        target_dict[key] = tmp_target_list

    return seq_dict, target_dict


def nt_augmentation(input_seqs, final_data_len=2e5, is_val_set=False):
    '''
    Wrapper function to setup nucleotide augmentation based on a desired augmented data length. If
    is_val_set = True, then sequences will be backtranslated (from amino acids to nucleotides) without
    augmentation

    Parameters
    ----------
    input_seqs : list or pandas.core.series.Series
        list or series of amino acid sequences
    final_data_len : int
        desired length of final data set
    is_val_set : bool
        whether or not input_seqs is a validation set. If is_val_set = True, backtranslation without
        augmentation is performed.

    Returns
    -------
    out_df : pandas.core.frame.DataFrame
        pandas dataframe containing augmented nucleotide sequences
    '''
    data_len = len(input_seqs['aaseq'])

    # round the calculated fraction to a whole number to get qty for first augmentation
    # this will augment the data greater than is necessary. i.e. for desired aug of 1.5, it will augment 2x
    calculated_aug_factor = int(ceil(final_data_len / data_len))

    if calculated_aug_factor == 0 or is_val_set == True:
        calculated_aug_factor = 1
    # for her2 negatives, augmenting by 2 then subsampling will decrease the sequence diveristy of the majority class
    # essentially acting as a majority class downsampling effect
    # elif is_val_set =='her2_neg': calculated_aug_factor = 2

    data = input_seqs.copy()
    aa_seq_list = data['aaseq']
    target_list = data['target']
    seq_dict, target_dict = aa_to_nt(aa_seq_list, target_list=target_list, aug_factor=calculated_aug_factor)

    # randomly downsample augmented data set to desired length
    if is_val_set == False:
        truncate_factor = final_data_len / data_len
        len_seq_dict = sum([len(x) for x in seq_dict.values()])  # number of total nucleotide sequences in dictionary

        # downsample augmented sequences by iteratively dropping one augmented nt seq from each
        # aa seq until desired data size is reached
        if final_data_len < len_seq_dict:
            num_seqs_to_drop = int(len_seq_dict - final_data_len)
            for i in range(num_seqs_to_drop):
                seq_dict[i] = np.random.choice(seq_dict[i], len(seq_dict[i]) - 1, replace=False)
                target_dict[i] = np.random.choice(target_dict[i], len(target_dict[i]) - 1, replace=False)

    seq_out_list = []
    target_out_list = []
    for key in seq_dict:
        for seq_entry in seq_dict[key]:
            seq_out_list.append(seq_entry)
        for target_entry in target_dict[key]:
            target_out_list.append(target_entry)

    out_df = pd.DataFrame(seq_out_list)
    out_df.columns = ['dnaseq']
    out_df['target'] = target_out_list

    return out_df
