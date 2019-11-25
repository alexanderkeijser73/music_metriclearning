import sys
import os
sys.path.append('..')

import torch
import torch.nn.functional as F
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
from sacred import Experiment

from train_utils import split_train_valid, load_checkpoint
from data.dataloader import ToMel, NormFreqBands
from models.featurenet import FeatureNet
from models.featurenet_alternative_filter_shape import FeatureNetSquareKernel, FeatureNetBandKernel
from models.patchcorrelationnet import PatchCorrelationNet
from models.siamesenet import SiameseNet
from models.multi_timescale_cnn import MultiTimescaleCNN
from models.pnet_onet import CombinedNet
from models.nips_cnn import NipsCNN

ex = Experiment('test_metricnet_songlevel')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@ex.config
def config():
    ex.add_config('config.yaml')

def song_level_constraint(net, query_song, pos_song, neg_song, threshold=0.1, cos_similarity=True):

    if isinstance(net, SiameseNet):
        similarity_positive = net(query_song, pos_song)
        similarity_negative = net(query_song, neg_song)
        return similarity_positive > similarity_negative
    if isinstance(net, PatchCorrelationNet):
        distance_positive = net(query_song, pos_song)
        distance_negative = net(query_song, neg_song)
    else:
        query_fts = net(query_song)
        pos_fts = net(pos_song)
        neg_fts = net(neg_song)
        if cos_similarity:
            distance_positive = F.cosine_similarity(query_fts, pos_fts)
            distance_negative = F.cosine_similarity(query_fts, neg_fts)
        else:
            distance_positive = (query_fts - pos_fts).pow(2).sum(1)  # .pow(.5)
            distance_negative = (query_fts - neg_fts).pow(2).sum(1)  # .pow(.5)

    # return distance_negative - distance_positive > threshold
    return distance_negative > distance_positive


@ex.capture
def test(test_dataloader, net, cos_similarity=False, verbose=False):

    n_constraints_satisfied = 0
    for i, triplet in enumerate(test_dataloader):
        if verbose:
            print(f'Evaluating triplet {i+1}')

        query, pos, neg = triplet
        query, pos, neg = query.to(device), pos.to(device), neg.to(device)

        constraint_satisfied = song_level_constraint(net, query, pos, neg, cos_similarity=cos_similarity)

        if constraint_satisfied:
            n_constraints_satisfied += 1
            if verbose:
                print(f'{n_constraints_satisfied} out of {i+1} song-level constraints satisfied!')



    constraint_fulfillment_rate = n_constraints_satisfied / len(test_dataloader)

    print('---------------------------------------------------')
    print(f'Total constraint fulfillment rate: {constraint_fulfillment_rate * 100:.1f}%')

    return constraint_fulfillment_rate


# @ex.automain
# def main(checkpoint_path,
#          best_checkpoint_featurenet,
#          best_checkpoint_nips,
#          best_checkpoint_pcnet,
#          best_checkpoint_multitimescalecnn,
#          best_checkpoint_siamese,
#          clips_dir,
#          comparisons_file,
#          stft_dir,
#          cos_similarity,
#          model,
#          test_on_valid,
#          featurenet_embedding_dim,
#          chunk_size,
#          overlap,
#          pcnet_reduction_method,
#          feature_extractor_model,
#          pnet_onet_n_fc,
#          featurenet_square_kernel,
#          featurenet_band_kernel,
#          loss_function):
#
#     # Model
#     if model == 'nips':
#         net = NipsCNN()
#         transforms_list = [
#             ToMel(n_mels=128, num_n_fft=1, start_n_fft=1024, sr=16000, f_max=None)
#         ]
#         stft_dir = None
#         checkpoint = os.path.join(checkpoint_path, best_checkpoint_nips)
#     elif model == 'featurenet':
#         net = FeatureNet()
#         transforms_list = [
#             ToMel(n_mels=80, hop=512, f_min=0, f_max=8000, sr=16000, num_n_fft=3, start_n_fft=1024),
#             NormFreqBands()
#         ]
#         checkpoint = os.path.join(checkpoint_path, best_checkpoint_featurenet)
#     elif model == 'pcnet' or model == 'siamese':
#         feature_extractor = None
#         if feature_extractor_model == 'featurenet':
#             if featurenet_square_kernel and featurenet_band_kernel:
#                 raise ValueError(f'Cannot use featurenet_band_kernel={featurenet_band_kernel} and '
#                                  f'featurenet_square_kernel={featurenet_square_kernel}. Please edit config.yaml')
#             if featurenet_square_kernel:
#                 feature_extractor = FeatureNetSquareKernel(n_outputs=featurenet_embedding_dim,
#                                                             sigmoid=True if loss_function == 'losslesstripletloss' else False)
#             elif featurenet_band_kernel:
#                 feature_extractor = FeatureNetBandKernel(n_outputs=featurenet_embedding_dim,
#                                                          sigmoid=True if loss_function == 'losslesstripletloss' else False)
#             else:
#                 feature_extractor = FeatureNet(n_outputs=featurenet_embedding_dim,
#                                                sigmoid=True if loss_function == 'losslesstripletloss' else False)
#             transforms_list = [
#                 ToMel(n_mels=80, hop=512, f_min=0, f_max=8000, sr=16000, num_n_fft=3, start_n_fft=1024),
#                 NormFreqBands()
#             ]
#         elif feature_extractor_model == 'pnet_onet':
#             feature_extractor = CombinedNet(sr=16000, hop=512, input_width=chunk_size, n_fc=pnet_onet_n_fc)
#             transforms_list = [
#                 ToMel(n_mels=80, hop=512, f_min=0, f_max=8000, sr=16000, num_n_fft=1, start_n_fft=1024),
#                 NormFreqBands()
#             ]
#         elif feature_extractor_model == 'nips':
#             feature_extractor = NipsCNN()
#             transforms_list = [
#                 ToMel(n_mels=128, num_n_fft=1, start_n_fft=1024, sr=16000, f_max=None)
#             ]
#             stft_dir = None
#
#         elif feature_extractor_model == 'multitimescalecnn':
#             feature_extractor = MultiTimescaleCNN()
#             transforms_list = [
#                 ToMel(n_mels=80, hop=512, f_min=0, f_max=8000, sr=16000, num_n_fft=3, start_n_fft=1024),
#                 NormFreqBands()
#             ]
#
#         if model == 'pcnet':
#             net = PatchCorrelationNet(feature_extractor, chunk_size=chunk_size, overlap=overlap,
#                                       pcnet_reduction_method=pcnet_reduction_method)
#             checkpoint = os.path.join(checkpoint_path, best_checkpoint_pcnet)
#
#         elif model == 'siamese':
#             net = SiameseNet(feature_extractor)
#             checkpoint = os.path.join(checkpoint_path, best_checkpoint_siamese)
#
#
#     elif model == 'multitimescalecnn':
#         net = MultiTimescaleCNN()
#         transforms_list = [
#             ToMel(n_mels=80, hop=512, f_min=0, f_max=8000, sr=16000, num_n_fft=3, start_n_fft=1024),
#             NormFreqBands()
#         ]
#         checkpoint=os.path.join(checkpoint_path, best_checkpoint_multitimescalecnn)
#     else:
#         raise ValueError(f'Model {model} not supported. '
#                          f'Choose between \'nips\'\'featurenet\'\'pcnet\'')
#
#     net = load_checkpoint(net, checkpoint)
#     net = net.to(device)
#     print('Checkpoint loaded...')
#
#     dataset = MTATTripletDataset(clips_dir=clips_dir,
#                                  comparisons_file=comparisons_file,
#                                  transform=transforms.Compose(transforms_list), stft_dir=stft_dir)
#
#     if test_on_valid:
#         _, sampler = split_train_valid(dataset)
#     else:
#         test_indices = torch.load(os.path.join(checkpoint_path, 'test_indices.pt'))
#         sampler = SubsetRandomSampler(test_indices)
#
#     test_dataloader = DataLoader(dataset,
#                                   batch_size=1,
#                                   shuffle=False,
#                                   sampler=sampler,
#                                   num_workers=0,
#                                   pin_memory=False)
#
#     test_set_size = len(test_dataloader)
#     print(f'Testing on test set of size {test_set_size}...')
#
#     cfr = test(test_dataloader, net, cos_similarity=cos_similarity)
#
