import sys

sys.path.append('..')

from train_utils import get_triplet_preds

import torch
import torch.nn.functional as F
from itertools import product

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def check_song_level_constraint(ftr_net, metr_net, query_song, pos_song, neg_song):
    # get query-pos, query-neg similarities, dim 0 = query frame idx, dim 1 = pos/neg frame idx
    pos_sims, neg_sims = get_triplet_preds(ftr_net, metr_net, (query_song, pos_song, neg_song))
    # create all query-pos-neg frame triplet combinations
    idx_tuples = product(range(pos_sims.size(1)), range(neg_sims.size(1)))
    pos_idxs, neg_idxs = list(zip(*idx_tuples))
    pos = torch.index_select(pos_sims, dim=1, index=torch.LongTensor(pos_idxs))
    neg = torch.index_select(neg_sims, dim=1, index=torch.LongTensor(neg_idxs))
    # perform majority voting to evaluate song-level constraint
    frame_level_constraint = pos > neg
    song_level_constraint = (frame_level_constraint.sum() > frame_level_constraint.numel()/2).item()
    return song_level_constraint


def eval(test_dataloader, ftr_net, metr_net, verbose=False):

    n_constraints_satisfied = 0
    for i, triplet in enumerate(test_dataloader):
        if verbose:
            print(f'Evaluating triplet {i+1}')

        query, pos, neg = triplet
        query, pos, neg = query.to(device), pos.to(device), neg.to(device)

        constraint_satisfied = check_song_level_constraint(ftr_net, metr_net, query, pos, neg)

        if constraint_satisfied:
            n_constraints_satisfied += 1
            if verbose:
                print(f'{n_constraints_satisfied} out of {i+1} song-level constraints satisfied!')



    constraint_fulfillment_rate = n_constraints_satisfied / len(test_dataloader)

    print('---------------------------------------------------')
    print(f'Total constraint fulfillment rate: {constraint_fulfillment_rate * 100:.1f}%')

    return constraint_fulfillment_rate


