import sys
sys.path.append('..')
from tqdm import tqdm
import os

import torch
from torchvision import transforms
from music_metriclearning.data.k_fold_cross_validation import PreprocessorKFold, FoldDataset
from music_metriclearning.data.dataloader import ToMel, NormFreqBands, Chunk
from music_metriclearning.train_utils import save_checkpoint, load_checkpoint, load_config, parse_args_config, get_triplet_preds, print_config

# hide shitty warnings
import warnings
warnings.simplefilter("ignore")
# Transforms applied by dataloader
transforms_list = [
    ToMel(n_mels=80, hop=512, f_min=0., f_max=8000., sr=16000, num_n_fft=3, start_n_fft=1024),
    NormFreqBands()
]
config_file = 'config_local.yaml'

# Handle config file/cli args
config = load_config(config_file)

# Analyse relative similarity votes using graph and split into K folds
pr = PreprocessorKFold(config, [])
pr.stft_dir = None

if not os.path.exists(config.stft_dir):
    os.makedirs(config.stft_dir)

train_triplets, test_triplets = pr.get_next_fold()

train_dl = FoldDataset(pr, train_triplets)
test_dl = FoldDataset(pr, test_triplets)

for i, (query, pos, neg) in enumerate(train_dl):
    query_id, pos_id, neg_id = train_dl.triplets[i]
    fname = os.path.join(config.stft_dir, str(query_id) + '.pt')
    if not os.path.exists(fname):
        torch.save(query, fname)

    fname = os.path.join(config.stft_dir, str(pos_id) + '.pt')
    if not os.path.exists(fname):
        torch.save(pos, fname)

    fname = os.path.join(config.stft_dir, str(neg_id) + '.pt')
    if not os.path.exists(fname):
        torch.save(neg, fname)
    print(f'processed {i}/{len(train_dl)} triplets')

for i, (query, pos, neg) in enumerate(test_dl):
    query_id, pos_id, neg_id = train_dl.triplets[i]
    fname = os.path.join(config.stft_dir, str(query_id) + '.pt')
    if not os.path.exists(fname):
        torch.save(query, fname)

    fname = os.path.join(config.stft_dir, str(pos_id) + '.pt')
    if not os.path.exists(fname):
        torch.save(pos, fname)

    fname = os.path.join(config.stft_dir, str(neg_id) + '.pt')
    if not os.path.exists(fname):
        torch.save(neg, fname)
    print(f'processed {i}/{len(test_dl)} triplets')

