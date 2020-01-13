import os
from itertools import product

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import SubsetRandomSampler
import yaml
import configargparse

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def print_config(config_obj):
    for k, v in vars(config_obj).items():
        print(f'{k}: {v}')
    print('-' * 79)

def get_triplet_preds(ft_net, mtr_net, batch):
    # Configure input
    query, pos, neg = batch
    query, pos, neg = query.squeeze().to(device), pos.squeeze().to(device), neg.squeeze().to(device)

    # Calculate features
    query_fts, pos_fts, neg_fts = ft_net(query), ft_net(pos), ft_net(neg)

    # Get all combinations of patches to be fed into metricnet
    # and calculate patch similarities
    pos_input = torch.cat(get_patch_tuples(query_fts, pos_fts), dim=1)
    neg_input = torch.cat(get_patch_tuples(query_fts, neg_fts), dim=1)
    pos_sims = mtr_net(pos_input)
    neg_sims = mtr_net(neg_input)
    pos_sims = pos_sims.view(query_fts.size(0), -1)
    neg_sims = neg_sims.view(query_fts.size(0), -1)
    return pos_sims, neg_sims

def parse_args_config():
      p = configargparse.ArgParser(default_config_files=[], config_file_parser_class=configargparse.YAMLConfigFileParser)
      p.add('-c', '--my-config', required=False, is_config_file=True, help='config file path')
      p.add('--checkpoint_path', required=False, type=str)
      p.add('--best_checkpoint', required=False, type=str)
      p.add('--save', required=False, action='store_true')
      p.add('--comparisons_file', required=False, type=str)
      p.add('--clips_dir', required=False, type=str)
      p.add('--stft_dir', required=False)
      p.add('--verbose', required=False, type=bool)
      p.add('--validate_every', required=False, type=int)
      p.add('--lr', required=False, type=float)
      p.add('--sr', required=False, type=int)
      p.add('--n_folds', required=False, type=int)
      p.add('--tensorboard_logdir', required=False)
      p.add('--lr_patience', required=False, type=int)
      p.add('--lr_decrease_factor', required=False, type=float)
      p.add('--batch_size', required=False, type=int)
      p.add('--valid_batch_size', required=False, type=int)
      p.add('--n_epochs', required=False, type=int)
      p.add('--debugging', required=False, action='store_true')
      # _, remaining_argv = p.parse_known_args()
      config = p.parse_args()
      print(config.debugging)
      return config


class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

def load_config(yaml_file='config.yaml'):
    """
    Loads configuration parameters from yaml file into Config object
    :param yaml_file: location of config.yaml file containing configuration parameters
    :return: config object containing all configuration parameters
    """
    config_dict = yaml.load(open(yaml_file, 'r'))
    config_obj = Config(**config_dict)
    return config_obj

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
    query = torch.index_select(query_fts, dim=0, index=torch.LongTensor(query_idxs).to(device))
    comp = torch.index_select(comp_fts, dim=0, index=torch.LongTensor(comp_idxs).to(device))
    return query, comp