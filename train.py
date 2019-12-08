import sys
sys.path.append('..')
import os

from .data.k_fold_cross_validation import PreprocessorKFold, FoldDataset
from .data.dataloader import ToMel, NormFreqBands, Chunk
from .models.featurenet import FeatureNet
from .models.metricnet import MetricNet
from .loss import TripletLoss

from .testing_song_level import test
from .train_utils import save_checkpoint, load_checkpoint, get_patch_tuples

from sacred import Experiment
from sacred.observers import MongoObserver
import torch
from torchvision import transforms
import time



def config():
    ex.add_config('config.yaml')

def set_start_timestamp():
    global time_now
    time_now = time.strftime('%d_%b_%H_%M_%S')

def get_batch_loss(ft_net, mtr_net, batch, criterion):
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

    # Calculate loss
    loss = criterion(pos_sims, neg_sims, query_size=query_fts.size(0))

    return loss


def validate(ft_net, mtr_net, valid_batch, criterion):
    ft_net.eval()
    mtr_net.eval()

    valid_loss = get_batch_loss(ft_net, mtr_net, valid_batch, criterion)

    ft_net.train()
    mtr_net.train()
    return valid_loss

def train_k_fold_cv(pr, n_folds=10, batch_size=4, valid_batch_size=4, lr_decrease_factor=None, lr_patience=None):

    per_fold_cfr = []

    for fold_idx in range(n_folds):

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
                                                               factor=lr_decrease_factor,
                                                               verbose=True,
                                                               patience=lr_patience)

        print('--------------------------------------------------')
        print(f'processing fold {fold_idx + 1}.....')

        train_triplets, test_triplets = pr.get_next_fold()

        # TODO

        train_dl = torch.utils.data.DataLoader(FoldDataset(pr, train_triplets, siamese=False), batch_size)
        valid_dl = torch.utils.data.DataLoader(FoldDataset(pr, test_triplets, siamese=False), valid_batch_size)
        # test_dl = torch.utils.data.DataLoader(FoldDataset(pr, test_triplets), 1)

        print('training....')

        train_fold(fold_idx + 1, feature_extractor, metricnet, train_dl, valid_dl, criterion, optimizer, scheduler)

        print('testing....')

        # cfr = test_fold(fold_idx + 1, net, test_dl)
        # per_fold_cfr.append(cfr)

    print('--------------------------------------------------')
    print('--------------------------------------------------')
    print(f'Mean CFR: {sum(per_fold_cfr) / len(per_fold_cfr) * 100:.4f}%')


def train_fold(fold,
               ft_net,
               mtr_net,
               train_dataloader,
               test_dataloader,
                criterion,
                optimizer,
                scheduler,
                batch_size,
                validate_every,
                n_epochs):

    best_loss = 1e10
    for epoch in range(n_epochs):
        best_loss = train_epoch(fold, epoch, best_loss, ft_net, mtr_net, criterion, optimizer, train_dataloader,
                                test_dataloader, batch_size, validate_every)
        print(f'Epoch {epoch} best valid loss: {best_loss}')
        scheduler.step(best_loss)



def test_fold(fold, net, test_dl, checkpoint_path=None, verbose=None):
    checkpoint = os.path.join(checkpoint_path, f'best_model_{time_now}_fold{fold}.pt.tar')
    net = load_checkpoint(net, checkpoint)
    print('Checkpoint loaded...')
    cfr = test(test_dl, net, False, verbose=verbose)
    return cfr


def train_epoch(fold, epoch, best_loss, ft_net, mtr_net, criterion, optimizer, train_dataloader, valid_dataloader,
                batch_size, validate_every, n_epochs, checkpoint_path, save, verbose):
    for i, batch in enumerate(train_dataloader):
        total_steps = i + epoch * len(train_dataloader)

        # Only for time measurement of step through network
        t1 = time.time()

        optimizer.zero_grad()

        loss = get_batch_loss(ft_net, mtr_net, batch, criterion)

        loss.backward()
        optimizer.step()

        # Just for time measurement
        t2 = time.time()
        examples_per_second = batch_size / float(t2 - t1)

        if verbose:
            print(
                "[Epoch %d/%d] "
                "[Batch %d/%d] "
                "[total loss: %f] "
                "[Examples/Sec = %.2f]"
                % (epoch, n_epochs, i, len(train_dataloader), loss.item(), examples_per_second)
            )

        ex.log_scalar(f'train loss fold {fold}', loss.item(), step=total_steps)

        if validate_every:
            if i % validate_every == 0:
                valid_batch = next(iter(valid_dataloader))

                valid_loss = validate(ft_net, mtr_net, valid_batch, criterion)

                if verbose:
                    print(f'VALID LOSS: {valid_loss} -----------------------------------')

                ex.log_scalar(f'valid loss fold {fold}', valid_loss.item(), step=total_steps)

                if valid_loss < best_loss:
                    best_loss = valid_loss
                    if save:
                        save_checkpoint(ft_net,
                                        optimizer,
                                        checkpoint_path,
                                        filename=f'best_model_{time_now}_fold{fold}.pt.tar',
                                        valid_loss=valid_loss.item())


    return best_loss


def main(comparisons_file,
         n_folds,
         clips_dir,
         stft_dir):

    set_start_timestamp()

    transforms_list = [
        ToMel(n_mels=80, hop=512, f_min=0., f_max=8000., sr=16000, num_n_fft=3, start_n_fft=1024),
        NormFreqBands(),
        Chunk()
    ]

    pr =  PreprocessorKFold(clips_dir,
                            comparisons_file,
                            n_folds,
                            transforms.Compose(transforms_list),
                            stft_dir)

    train_k_fold_cv(pr)