import os

import librosa
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from .similarity_graph import SimilarityGraph

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class PreprocessorKFold():

    def __init__(self, config, transform=None, ):
        self.df = pd.read_csv(config.comparisons_file, delimiter='\t')
        self.config = config
        self.clips_dir = config.clips_dir
        self.transform = transform
        self.stft_dir = config.stft_dir
        self.sr = config.sr
        self.sg = SimilarityGraph(self.df)
        self.kf = KFold(n_splits=config.n_folds)
        self.folds = iter(self.kf.split(self.sg.subgraphs))
        self.id2path = self.sg.id2path


    def get_next_fold(self):
        """

        :return: Tuple of two lists (train and test) for next fold.
                Lists contain triplets of clip_ids, which are used
                to load songs.
        """
        train, test  = next(self.folds)
        train_subgraphs = [sg for i,sg in enumerate(self.sg.subgraphs) if i in train]
        test_subgraphs = [sg for i, sg in enumerate(self.sg.subgraphs) if i in test]
        nodes_in_fold = lambda fold: [node for sg in fold for node in sg]
        edges_in_fold = lambda sg, fold: [edges for node in nodes_in_fold(fold) for edges in sg.graph.edges(node)]
        train_triplets = [SimilarityGraph.triplet_from_edge(edge) for edge in edges_in_fold(self.sg, train_subgraphs)]
        test_triplets = [SimilarityGraph.triplet_from_edge(edge) for edge in edges_in_fold(self.sg, test_subgraphs)]
        train_dl = DataLoader(FoldDataset(self, train_triplets), self.config.batch_size)
        valid_dl = DataLoader(FoldDataset(self, test_triplets), self.config.valid_batch_size)
        test_dl = DataLoader(FoldDataset(self, test_triplets), 1)
        return train_dl, valid_dl, test_dl



class FoldDataset():


    def __init__(self, preprocessor, triplets, siamese=False):
        """
        Bla
        :param preprocessor:
        :param triplets:
        :param siamese:
        """
        self.preprocessor = preprocessor
        self.id2path = preprocessor.id2path
        self.triplets = triplets
        self.siamese = siamese

    def __getitem__(self, item):
        query_id, pos_id, neg_id = self.triplets[item]

        if self.preprocessor.stft_dir is None or self.preprocessor.stft_dir == 'None':
            query = torch.from_numpy(
                librosa.load(
                    os.path.join(self.preprocessor.clips_dir, self.preprocessor.id2path[query_id]),
                    mono=True,
                    sr=self.preprocessor.sr
                )[0]
            )

            pos = torch.from_numpy(
                librosa.load(
                    os.path.join(self.preprocessor.clips_dir, self.preprocessor.id2path[pos_id]),
                    mono=True,
                    sr=self.preprocessor.sr
                )[0]
            )

            neg = torch.from_numpy(
                librosa.load(
                    os.path.join(self.preprocessor.clips_dir, self.preprocessor.id2path[neg_id]),
                    mono=True,
                    sr=self.preprocessor.sr)
                [0]
            )

            if self.preprocessor.transform:
                query, pos, neg = query.to(device), pos.to(device), neg.to(device)
                query = self.preprocessor.transform(query)
                pos = self.preprocessor.transform(pos)
                neg = self.preprocessor.transform(neg)
        else:
            fname = os.path.join(self.preprocessor.stft_dir, str(query_id) + '.pt')
            try:
                query = torch.load(fname, map_location=device.type)

            except:
                raise RuntimeError(f'File {fname} could not be found')

            fname = os.path.join(self.preprocessor.stft_dir, str(pos_id) + '.pt')
            try:
                pos = torch.load(fname, map_location=device.type)
            except:
                raise RuntimeError(f'File {fname} could not be found')

            fname = os.path.join(self.preprocessor.stft_dir, str(neg_id) + '.pt')
            try:
                neg = torch.load(fname, map_location=device.type)
            except:
                raise RuntimeError(f'File {fname} could not be found')
        return query, pos, neg


    def __len__(self):
        if self.siamese:
            return 2 * len(self.triplets)
        else:
            return len(self.triplets)
