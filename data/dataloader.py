import os

import librosa
import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram
from torchvision import transforms

from .similarity_graph import SimilarityGraph

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class MTATTripletDataset(Dataset):

    def __init__(self, clips_dir, comparisons_file, transform=None, stft_dir=None, sr=None):
        super(MTATTripletDataset, self).__init__()
        self.clips_dir = clips_dir
        self.graph,  = SimilarityGraph(comparisons_file)
        self.triplets, _= self.graph.get_triplets_and_clip_pairs()
        self.id2path = self.graph.id2path
        self.transform = transform
        self.stft_dir = stft_dir
        self.sr = sr

    def __getitem__(self, idx):
        # get a triplet from the graph
        triplet = self.triplets[idx]

        query, pos, neg = None, None, None

        if self.stft_dir is None:

            query = torch.from_numpy(
                librosa.load(
                    os.path.join(self.clips_dir, self.id2path[triplet.query_id]),
                    mono=True,
                    sr=self.sr
                )[0]
            )

            pos = torch.from_numpy(
                librosa.load(
                    os.path.join(self.clips_dir, self.id2path[triplet.pos_id]),
                    mono=True,
                    sr=self.sr
                )[0]
            )

            neg = torch.from_numpy(
                librosa.load(
                    os.path.join(self.clips_dir, self.id2path[triplet.neg_id]),
                    mono=True,
                    sr=self.sr)
                [0]
            )

            if self.transform:
                query, pos, neg = query.to(device), pos.to(device), neg.to(device)
                query = self.transform(query)
                pos = self.transform(pos)
                neg = self.transform(neg)
        else:
            fname = os.path.join(self.stft_dir, str(triplet.query_id) + '.pt')
            try:
                query = torch.load(fname, map_location=device.type)
            except:
                raise RuntimeError(f'File {fname} could not be found')
            fname = os.path.join(self.stft_dir, str(triplet.pos_id) + '.pt')
            try:
                pos = torch.load(fname, map_location=device.type)
            except:
                raise RuntimeError(f'File {fname} could not be found')
            fname = os.path.join(self.stft_dir, str(triplet.neg_id) + '.pt')
            try:
                neg = torch.load(fname, map_location=device.type)
            except:
                raise RuntimeError(f'File {fname} could not be found')

        assert not query is None and not pos is None and not neg is None, 'Found None sample'
        return query, pos, neg

    def __len__(self):
        return len(self.triplets)



# class MTATSiameseDataset(Dataset):
#
#     def __init__(self, clips_dir, comparisons_file, transform=None, stft_dir=None, sr=None):
#         super(MTATSiameseDataset, self).__init__()
#         self.clips_dir = clips_dir
#         self.graph = create_comparison_graph(comparisons_file)
#         self.triplets, _= self.graph.get_triplets_and_clip_pairs()
#         self.id2path = self.graph.id2path
#         self.transform = transform
#         self.stft_dir = stft_dir
#         self.sr = sr
#
#     def __getitem__(self, idx):
#         # get a triplet from the graph
#         triplet = self.triplets[idx//2]
#
#         # whether to choose the similar or dissimilar pair from the triplet
#         label = idx % 2
#
#         query, pos, neg = None, None, None
#
#         if self.stft_dir is None:
#
#             query = torch.from_numpy(
#                 librosa.load(
#                     os.path.join(self.clips_dir, self.id2path[triplet.query_id]),
#                     mono=True,
#                     sr=self.sr
#                 )[0]
#             )
#
#             if label == 1:
#                 comp = torch.from_numpy(
#                     librosa.load(
#                         os.path.join(self.clips_dir, self.id2path[triplet.pos_id]),
#                         mono=True,
#                         sr=self.sr
#                     )[0]
#                 )
#
#             elif label == 0:
#                 comp = torch.from_numpy(
#                     librosa.load(
#                         os.path.join(self.clips_dir, self.id2path[triplet.neg_id]),
#                         mono=True,
#                         sr=self.sr)
#                     [0]
#                 )
#
#             else:
#                 raise RuntimeError(f'Unexpected induced label ({label}) . Should be 0 or 1.')
#
#             if self.transform:
#                 query, comp = query.to(device), comp.to(device)
#                 query = self.transform(query)
#                 comp = self.transform(comp)
#
#         else:
#             fname = os.path.join(self.stft_dir, str(triplet.query_id) + '.pt')
#             try:
#                 query = torch.load(fname, map_location=device.type)
#             except:
#                 raise RuntimeError(f'File {fname} could not be found')
#
#             if label == 1:
#                 fname = os.path.join(self.stft_dir, str(triplet.pos_id) + '.pt')
#                 try:
#                     comp = torch.load(fname, map_location=device.type)
#                 except:
#                     raise RuntimeError(f'File {fname} could not be found')
#
#             elif label == 0:
#                 fname = os.path.join(self.stft_dir, str(triplet.neg_id) + '.pt')
#                 try:
#                     comp = torch.load(fname, map_location=device.type)
#                 except:
#                     raise RuntimeError(f'File {fname} could not be found')
#
#             else:
#                 raise RuntimeError(f'Unexpected induced label ({label}) . Should be 0 or 1.')
#
#         assert not query is None and not comp is None
#         return query, comp, label
#
#     def __len__(self):
#         return 2*len(self.triplets)


class ToMel(object):

    def __init__(self, n_mels=80, hop=512, f_min=0., f_max=8000., sr=16000, num_n_fft=3, start_n_fft=1024):
        self.n_mels = n_mels
        self.hop = hop
        self.f_min = f_min
        self.f_max = f_max
        self.sr = sr

        self.num_n_fft = num_n_fft
        self.start_n_fft = start_n_fft

    def __call__(self, sample):
        cat_sample = []
        # concatenate mel-spectrograms with different window sizes (doubled each time)
        for i in range(self.num_n_fft):
            n_fft = self.start_n_fft * 2 ** i
            cat_sample.append(MelSpectrogram(n_mels=self.n_mels,
                                             hop_length=self.hop,
                                             f_min=self.f_min,
                                             f_max=self.f_max,
                                             sample_rate=self.sr, n_fft=n_fft
                                             )(sample.unsqueeze(0))
                              )
        # transpose to shape Nx3xTxF
        return torch.cat(cat_sample).transpose(-1,-2)


class NormFreqBands(object):

    def __call__(self, sample):
        mean = torch.mean(sample, dim=1, keepdim=True)
        std = torch.std(sample, dim=1, keepdim=True)
        sample = (sample - mean) / std
        return sample

class LogCompress(object):

    def __call__(self, sample):
        return torch.log(1 + sample)

class Chunk(object):

    def __call__(selfs, sample):
        chunks = torch.split(sample, 50, dim=-2)
        if chunks[-1].size(-2) < chunks[0].size(-2):
            chunks = chunks[:-1]
        return torch.stack(chunks)

def get_dataloader(clips_dir, comparisons_file, batch_size=64, shuffle=True, sampler=None):
    dataset = MTATTripletDataset(clips_dir, comparisons_file, transform=transforms.Compose([ToMel(), NormFreqBands()]))
    dataloader = DataLoader(dataset, batch_size, shuffle, sampler)
    return dataloader