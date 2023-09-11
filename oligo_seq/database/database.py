from typing import *

import torch
from torch .utils import data
from torch.nn.utils import rnn
import pandas as pd




class MLPDataset(data.Dataset):


    def __init__(self, path: str) -> None:
        super().__init__()
        database = pd.read_csv(path, index_col=0) # dictionary with sequences, features and labels tensors
        oligo_length = database["oligo_length"][0]
        encodings = torch.empty((len(database), 8*oligo_length))
        for i, row in database.iterrows():
            oligo = row["oligo_sequence"]
            off_target = row["off_target_sequence"]
            encodings[i,:] = torch.cat([self.encode_sequence(oligo), self.encode_sequence(off_target)])
        self.labels = torch.tensor(database["duplexing_log_score"], dtype=torch.double)
        self.mean = self.labels.mean()
        self.std = self.labels.std()
        self.labels = (self.labels - self.mean)/self.std # normalize
        features = torch.tensor(database[["oligo_length", "oligo_GC_content", "off_target_legth", "off_target_GC_content", "tm_diff", "number_mismatches"]].to_numpy(), dtype=torch.double)
        self.data = torch.cat([encodings, features], dim=1)


    def __len__(self):
        return self.data.shape[0]
    

    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor]:
        return self.data[index,:], self.labels[index]
    

    def encode_sequence(self, sequence: str):
        sequence_encoding = torch.empty((4*len(sequence)))
        for i, nt in enumerate(sequence):
            #encoding order is A, T, C. G
            if nt == 'A' or nt == 'a':
                nt_encoding = torch.tensor([1,0,0,0])
            elif nt == 'C' or nt == 'c':
                nt_encoding = torch.tensor([0,1,0,0])
            elif nt == 'G' or nt == 'g':
                nt_encoding = torch.tensor([0,0,1,0])
            elif nt == 'T' or nt == 't':
                nt_encoding = torch.tensor([0,0,0,1])
            else:
                Warning(f"Nucleotide {nt} not recognized.")
            sequence_encoding[4*i:4*(i+1)] = nt_encoding
        return sequence_encoding.double()
    



class RNNDataset(data.Dataset):


    def __init__(self, path: str) -> None:
        super().__init__()
        database = pd.read_csv(path, index_col=0) # dictionary with sequences, features and labels tensors
        self.sequences = [] # list of tensors
        for i, row in database.iterrows():
            oligo = row["oligo_sequence"]
            off_target = row["off_target_sequence"]
            encoding = torch.empty((len(oligo), 8), dtype=torch.double) #update for deletions and insertions
            for j in range(len(oligo)):
                encoding[j,:] = torch.tensor(self.encode_nt(oligo[j]) + self.encode_nt(off_target[j]), dtype=torch.double)
            self.sequences.append(encoding)
        self.labels = torch.tensor(database["duplexing_log_score"], dtype=torch.double)
        self.mean = self.labels.mean()
        self.std = self.labels.std()
        self.labels = (self.labels - self.mean)/self.std # normalize
        self.features = torch.tensor(database[["oligo_length", "oligo_GC_content", "off_target_legth", "off_target_GC_content", "tm_diff", "number_mismatches"]].to_numpy(), dtype=torch.double)
        

    def __len__(self):
        return self.labels.shape[0]
    

    def __getitem__(self, index) -> Tuple[torch.tensor, torch.tensor, torch.torch.tensor]:
        # sequences is a dictionary of tensors since the length of each sequence can change
        return self.sequences[index], self.features[index,:], self.labels[index] 
    

    def encode_nt(self, nt:str) -> list[str]:
            """_summary_

            Args:
                nt (str): _description_

            Returns:
                list[str]: _description_
            """
            #encoding order is A, C, T, G
            if nt == 'A' or nt == 'a':
                nt_encoding = [1,0,0,0]
            elif nt == 'C' or nt == 'c':
                nt_encoding = [0,1,0,0]
            elif nt == 'T' or nt == 't':
                nt_encoding = [0,0,1,0]
            elif nt == 'G' or nt == 'g':
                nt_encoding = [0,0,0,1]
            else:
                Warning(f"Nucleotide {nt} not recognized.")
            return nt_encoding
    



def pack_collate(batch: list) -> Tuple[rnn.PackedSequence, torch.Tensor, torch.Tensor]:
    """Colalte function for the RNNDataset that generates a PackedSequence for the a list of sequences encodings in the batch.
    In this way can still train the input in batches and explot the speed-ups that batching delivers through vectorization.

    Args:
        batch (list): Output of the batch sampler.

    Returns:
        tuple: Tuple containig the batch elements, namely that encoded sequences, the additional features and the groud-truth labels.
    """

    sequences, features, labels = data._utils.collate.default_collate(batch)
    sequences = [sequences[i,:,:] for i in range(len(sequences))]
    # sort the sequences in decreasing length order
    lengths = torch.tensor(list(map(len, sequences)))
    lengths, perm_idx = lengths.sort(0, descending=True)
    sorted_sequences = [sequences[i] for i in perm_idx]
    padded_sequences = rnn.pad_sequence(sequences=sorted_sequences, batch_first=True)
    return rnn.pack_padded_sequence(input=padded_sequences, lengths=lengths, batch_first=True), features[perm_idx,:], labels[perm_idx] # reorder according to the original ordering