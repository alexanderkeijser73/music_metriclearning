import sys
sys.path.append('..')
from tqdm import tqdm

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
    NormFreqBands(),
    Chunk()
]
config_file = 'config_local.yaml'

# Handle config file/cli args
config = load_config(config_file)

# Analyse relative similarity votes using graph and split into K folds
pr = PreprocessorKFold(config, [])

try:
    train_triplets, test_triplets = pr.get_next_fold()

    train_dl = torch.utils.data.DataLoader(FoldDataset(pr, train_triplets, siamese=False), config.batch_size)
    test_dl = torch.utils.data.DataLoader(FoldDataset(pr, test_triplets), 1)
    for batch in tqdm(train_dl):
        query, pos, neg = batch
        # print(query.size(), pos.size(), neg.size())
        pass
    for batch in tqdm(test_dl):
        pass
except Exception as e:
    print(e)


