import os
from itertools import product

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler


def load_checkpoint(model_object, checkpoint_path):
    cuda = torch.cuda.is_available()
    map_location = 'cpu' if not cuda else None
    cp = torch.load(checkpoint_path, map_location)
    model_state_dict = cp['model']
    model_object.load_state_dict(model_state_dict)
    model_object.eval()
    return model_object

def split_train_valid(dataset, shuffle_dataset=True, validation_split=0.2, test_split=None):
    random_seed =42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    valid_split = int(np.floor(validation_split * dataset_size))

    if test_split:
        test_split = int(np.floor(test_split * dataset_size))
        train_indices, valid_indices, test_indices = indices[valid_split+test_split:], \
                                                     indices[:valid_split], \
                                                     indices[valid_split:valid_split+test_split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        return train_sampler, valid_sampler, test_indices

    else:
        train_indices, valid_indices= indices[valid_split:], indices[:valid_split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(valid_indices)

        return train_sampler, valid_sampler


def save_checkpoint(model, optimizer, rootdir, filename=None, valid_loss=None):
    """ Save the trained model checkpoint
    """
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)
    path = os.path.join(rootdir, filename)
    checkpoint = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'valid_loss': valid_loss
    }
    torch.save(checkpoint, path)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_k_fold_cv(train_function, test_function, net, dataset, n_folds=10, train_batch_size=4, valid_batch_size=4, **kwargs):

    kf = KFold(n_splits=n_folds, shuffle=True)
    idx_generator = kf.split(dataset)

    per_fold_score = []

    # Save untrained parameters
    torch.save(net.state_dict(), './initial_parameters.pt.tar')
    for fold_idx, (train_indices, test_indices) in enumerate(idx_generator):

        # Re-initialize network
        net.load_state_dict(torch.load('./initial_parameters.pt.tar'))

        print('--------------------------------------------------')
        print(f'processing fold {fold_idx + 1}.....')


        train_dl = torch.utils.data.DataLoader(dataset, train_batch_size, sampler=SubsetRandomSampler(train_indices))
        valid_dl = torch.utils.data.DataLoader(dataset, valid_batch_size, sampler=SubsetRandomSampler(test_indices))
        test_dl = torch.utils.data.DataLoader(dataset, 1, sampler=SubsetRandomSampler(test_indices))

        print('training....')

        checkpoint_fname = train_function(net=net, train_dataloader=train_dl, valid_dataloader=valid_dl, **kwargs)

        print(f'checkpoint for fold {fold_idx + 1} saved as: {checkpoint_fname}')

        print('testing....')

        net = load_checkpoint(net, checkpoint_fname)
        score = test_function(net=net, test_dataloader=test_dl)
        per_fold_score.append(score)

    print('--------------------------------------------------')
    print('--------------------------------------------------')
    print(f'Mean score: {sum(per_fold_score) / len(per_fold_score) * 100:.2f}%')

def get_patch_tuples(query_fts, comp_fts):
    idx_tuples = product(range(query_fts.size(0)), range(comp_fts.size(0)))
    query_idxs, comp_idxs = list(zip(*idx_tuples))
    query = torch.index_select(query_fts, dim=0, index=torch.LongTensor(query_idxs))
    comp = torch.index_select(comp_fts, dim=0, index=torch.LongTensor(comp_idxs))
    return query, comp