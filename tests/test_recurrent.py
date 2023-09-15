import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

import pytest
import torch

from oligo_seq.database import pack_collate
from torch.nn.utils.rnn import PackedSequence

def test_collate():
    seq1 = torch.tensor([[1,0,0,0], [0,0,1,0,], [0,1,0,0]])
    seq2 = torch.tensor([[1,0,0,0], [0,1,0,0,], [0,0,1,0], [0,0,0,1], [0,0,0,0]])
    feat1 = torch.tensor([1,2])
    feat2 = torch.tensor([3,4])
    label1 = torch.tensor([1])
    label2 = torch.tensor([2])
    seq, feats, labels = pack_collate([[seq1, feat1, label1], [seq2, feat2, label2]])
    true_seq = PackedSequence(data = torch.tensor([[1,0,0,0], [1,0,0,0,], [0,1,0,0], [0,0,1,0], [0,0,1,0], [0,1,0,0], [0,0,0,1], [0,0,0,0]]), batch_sizes=torch.tensor([2,2,2,1,1]))
    true_feats = torch.tensor([[3,4], [1,2]])
    true_labels = torch.tensor([[2],[1]])



    assert torch.equal(seq.data, true_seq.data) and torch.equal(seq.batch_sizes, true_seq.batch_sizes), "The packed seqeunces classes do not correspond"
    assert torch.equal(feats, true_feats), "The features do not correspond"
    assert torch.equal(labels, true_labels), "The labels do not correspond"
