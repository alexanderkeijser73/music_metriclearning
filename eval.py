from train_utils import get_triplet_preds, load_checkpoint
from models.featurenet import FeatureNet
from models.metricnet import MetricNet
from data.dataloader import transforms_list
from data.k_fold_cross_validation import PreprocessorKFold, FoldDataset


import torch
from torchvision import transforms
from itertools import product
import os
import re

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def check_song_level_constraint(ftr_net, metr_net, query_song, pos_song, neg_song):
    # get query-pos, query-neg similarities, dim 0 = query frame idx, dim 1 = pos/neg frame idx
    pos_sims, neg_sims = get_triplet_preds(ftr_net, metr_net, (query_song, pos_song, neg_song))
    # [N, M] with N=num query patches, M= num pos/neg patches
    # create all query-pos-neg frame triplet combinations
    # TODO: THIS IS MAYBE WHERE IT GOES WRONG, ALL SHAPES BEFORE WORK!
    idx_tuples = product(range(pos_sims.size(1)), range(neg_sims.size(1)))
    pos_idxs, neg_idxs = list(zip(*idx_tuples))
    pos = torch.index_select(pos_sims, dim=1, index=torch.LongTensor(pos_idxs).to(device))
    neg = torch.index_select(neg_sims, dim=1, index=torch.LongTensor(neg_idxs).to(device))
    # perform majority voting to evaluate song-level constraint
    # todo: is this correct?
    frame_level_constraint = pos < neg
    # todo: use get_triplet_preds
    # todo: check if this actually solved it
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

def eval_trained_model(config,
                       ftr_net_checkpoint_path,
                       metr_net_checkpoint_path,
                       fold=None):
    p = re.compile('(?<=fold)[0-9]+')
    fold = int(p.findall(ftr_net_checkpoint_path)[0])
    ftr_net = load_checkpoint(FeatureNet(),
                    os.path.join(config.checkpoint_path,
                                 ftr_net_checkpoint_path))
    metr_net = load_checkpoint(MetricNet(),
                    os.path.join(config.checkpoint_path,
                                 metr_net_checkpoint_path))


    # Analyse relative similarity votes using graph and split into K folds
    pr = PreprocessorKFold(config, transforms.Compose(transforms_list))

    for _ in range(fold):
        _, _, test_dl = pr.get_next_fold()

    eval(test_dl, ftr_net, metr_net, verbose=True)