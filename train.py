import sys
sys.path.append('..')
import os

from .data.k_fold_cross_validation import PreprocessorKFold, FoldDataset
from .data.dataloader import ToMel, NormFreqBands, Chunk
from .models.featurenet import FeatureNet
from .models.metricnet import MetricNet
from .loss import TripletLoss
from .eval import eval
from .train_utils import save_checkpoint, load_checkpoint, load_config, parse_args_config, get_triplet_preds, print_config

import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import time

# hide shitty warnings
import warnings
warnings.simplefilter("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def set_start_timestamp():
    global start_timestamp
    start_timestamp = time.strftime('%d_%b_%H_%M_%S')

def get_batch_loss(ft_net, mtr_net, batch, criterion):

    pos_sims, neg_sims = get_triplet_preds(ft_net, mtr_net, batch)
    # Calculate loss
    loss = criterion(pos_sims, neg_sims)

    return loss

def validate(ft_net, mtr_net, valid_batch, criterion):
    ft_net.eval()
    mtr_net.eval()

    valid_loss = get_batch_loss(ft_net, mtr_net, valid_batch, criterion)

    ft_net.train()
    mtr_net.train()
    return valid_loss

def train_k_fold_cv(config, pr):

    per_fold_cfr = []
    global summary_writer
    summary_writer = SummaryWriter(os.path.join(config.tensorboard_logdir, start_timestamp))

    for fold_idx in range(config.n_folds):
        # Initialize feature extractor net
        feature_extractor = FeatureNet()
        feature_extractor = feature_extractor.to(device)

        # Initialize metricnet
        metricnet = MetricNet()
        metricnet = metricnet.to(device)

        criterion = TripletLoss()

        # Optimizer
        optimizer = torch.optim.Adam(list(feature_extractor.parameters()) + list(metricnet.parameters()))

        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=config.lr_decrease_factor,
                                                               verbose=True,
                                                               patience=config.lr_patience)

        print('--------------------------------------------------')
        print(f'processing fold {fold_idx + 1}.....')

        train_triplets, test_triplets = pr.get_next_fold()

        train_dl = torch.utils.data.DataLoader(FoldDataset(pr, train_triplets, siamese=False), config.batch_size)
        valid_dl = torch.utils.data.DataLoader(FoldDataset(pr, test_triplets, siamese=False), config.valid_batch_size)
        test_dl = torch.utils.data.DataLoader(FoldDataset(pr, test_triplets), 1)

        print('training....')

        train_fold(config, fold_idx + 1, feature_extractor, metricnet, train_dl, valid_dl, criterion, optimizer, scheduler)

        print('testing....')

        cfr = test_fold(config, fold_idx + 1, test_dl, feature_extractor, metricnet)
        per_fold_cfr.append(cfr)

    mean_cfr = sum(per_fold_cfr) / len(per_fold_cfr) * 100
    summary_writer.add_hparams(hparam_dict=vars(config), metric_dict={'hparam/mean_cfr': mean_cfr})
    summary_writer.close()
    print('--------------------------------------------------')
    print('--------------------------------------------------')
    print(f'Mean CFR: {mean_cfr:.4f}%')



def train_fold(config,
               fold,
               ft_net,
               mtr_net,
               train_dataloader,
               test_dataloader,
               criterion,
               optimizer,
               scheduler):

        best_loss = 1e10
        for epoch in range(config.n_epochs):
            best_loss = train_epoch(config,
                                    fold,
                                    epoch,
                                    best_loss,
                                    ft_net,
                                    mtr_net,
                                    criterion,
                                    optimizer,
                                    train_dataloader,
                                    test_dataloader)
            print(f'Epoch {epoch} best valid loss: {best_loss}')
            scheduler.step(best_loss)



def test_fold(config, fold, test_dl, ftr_net, mtr_net):
    ftr_net_checkpoint = os.path.join(config.checkpoint_path, f'best_model_{start_timestamp}_fold{fold}_ftr_net.pt.tar')
    mtr_net_checkpoint = os.path.join(config.checkpoint_path, f'best_model_{start_timestamp}_fold{fold}_mtr_net.pt.tar')
    mtr_net = load_checkpoint(mtr_net, mtr_net_checkpoint)
    ftr_net = load_checkpoint(mtr_net, ftr_net_checkpoint)
    print('Checkpoint loaded...')
    cfr = eval(test_dl, ftr_net, mtr_net, config.verbose)
    return cfr


def train_epoch(config,
                fold,
                epoch,
                best_loss,
                ftr_net,
                mtr_net,
                criterion,
                optimizer,
                train_dataloader,
                valid_dataloader):

    for i, batch in enumerate(train_dataloader):
        total_steps = i + epoch * len(train_dataloader)

        # Only for time measurement of step through network
        t1 = time.time()

        optimizer.zero_grad()

        loss = get_batch_loss(ftr_net, mtr_net, batch, criterion)

        loss.backward()
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size / float(t2 - t1)

        if config.verbose:
            print(
                "[Epoch %d/%d] "
                "[Batch %d/%d] "
                "[total loss: %f] "
                "[Examples/Sec = %.2f]"
                % (epoch, config.n_epochs, i, len(train_dataloader), loss.item(), examples_per_second)
            )

        summary_writer.add_scalar(f'data/train loss fold {fold}', loss.item(), total_steps)

        if config.validate_every:
            if i % config.validate_every == 0:
                valid_batch = iter(valid_dataloader).next()

                valid_loss = validate(ftr_net, mtr_net, valid_batch, criterion)

                if config.verbose:
                    print(f'VALID LOSS: {valid_loss} -----------------------------------')

                summary_writer.add_scalar(f'data/valid loss fold {fold}', valid_loss.item(), total_steps)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    if config.save:
                        save_checkpoint(ftr_net,
                                        optimizer,
                                        config.checkpoint_path,
                                        filename=f'best_model_{start_timestamp}_fold{fold}_ftr_net.pt.tar',
                                        valid_loss=valid_loss.item())
                        save_checkpoint(mtr_net,
                                        optimizer,
                                        config.checkpoint_path,
                                        filename=f'best_model_{start_timestamp}_fold{fold}_mtr_net.pt.tar',
                                        valid_loss=valid_loss.item())



    return best_loss


def main(config_file='config_local.yaml', parse_args=False):
    print('-' * 79)
    print(f'Using device: {device.type}')
    print('-' * 79)

    set_start_timestamp()

    # Handle config file/cli args
    if parse_args:
        config = parse_args_config()
    else:
        config = load_config(config_file)
    print_config(config)

    # Transforms applied by dataloader
    transforms_list = [
        ToMel(n_mels=80, hop=512, f_min=0., f_max=8000., sr=16000, num_n_fft=3, start_n_fft=1024),
        NormFreqBands(),
        Chunk()
    ]

    # Analyse relative similarity votes using graph and split into K folds
    pr =  PreprocessorKFold(config,
                            transforms.Compose(transforms_list))

    train_k_fold_cv(config, pr)