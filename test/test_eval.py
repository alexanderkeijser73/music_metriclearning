import pytest
import torch
from itertools import product

from eval import check_song_level_constraint
from train_utils import get_triplet_preds
from models.metricnet import MetricNet
from models.featurenet import FeatureNet

def test_check_song_level_constraint():
    # IT FAILED ONCE!
    mtr_net = MetricNet()
    ftr_net = FeatureNet()
    for i in range(1000):
        q, p, n = torch.randn(3, 3, 50, 80)
        slc = check_song_level_constraint(ftr_net, mtr_net, q, p, n)
        slc_stupid = check_song_level_constraint_stupid(ftr_net, mtr_net, q, p, n)
    assert slc_stupid == slc


def check_song_level_constraint_stupid(ftr_net, metr_net, query_song,
                                       pos_song, neg_song):
    # get query-pos, query-neg similarities, dim 0 = query frame idx, dim 1 = pos/neg frame idx
    pos_sims, neg_sims = get_triplet_preds(ftr_net, metr_net, (query_song, pos_song, neg_song))
    # [N, M] with N=num query patches, M= num pos/neg patches

    frame_level_constraints = []
    # loop through query patches to get corresponding pos/neg sims
    for q_patch in zip(pos_sims, neg_sims):
        pos, neg = q_patch
        for p in pos:
            for n in neg:
                frame_level_constraints.append(p < n)
    assert len(frame_level_constraints) == pos_sims.size(1) * pos_sims.size(
        1) * neg_sims.size(1)
    song_level_constraint = (sum(frame_level_constraints) / len(
        frame_level_constraints) > 0.5)
    return song_level_constraint