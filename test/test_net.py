import torch
from train_utils import get_patch_tuples, get_triplet_preds
from models.featurenet import FeatureNet
from models.metricnet import MetricNet
from itertools import product

def test_get_patch_tuples_shape():
    for i in range(5):
        emb_dim = torch.randint(20, 600,(1,)).item()
        q_n_patches, p_n_patches, n_n_patches = torch.randint(2, 100, (3,))
        query_fts = torch.rand(q_n_patches, emb_dim)
        pos_fts = torch.rand(p_n_patches, emb_dim)
        neg_fts = torch.rand(n_n_patches, emb_dim)
        pos_input = torch.cat(get_patch_tuples(query_fts, pos_fts), dim=1)
        neg_input = torch.cat(get_patch_tuples(query_fts, neg_fts), dim=1)
        assert pos_input.size() == (q_n_patches*p_n_patches, 2*emb_dim)
        assert neg_input.size() == (q_n_patches * n_n_patches, 2 * emb_dim)

# todo: probably obsolete
def test_get_patch_tuples_axes():
    idx_tuples = product(range(10), range(10,20))
    query_idxs, comp_idxs = list(zip(*idx_tuples))
    query_idxs = torch.tensor(query_idxs).view(10, -1)
    comp_idxs = torch.tensor(comp_idxs).view(10, -1)
    print(query_idxs)
    print(comp_idxs)
    # find a way to check this automatically

def test_get_triplet_preds():
    ft_net = FeatureNet()
    mtr_net = MetricNet()
    for i in range(5):
        C = 3
        T_q, T_p, T_n =  torch.randint(50, 5000, (3,))
        F = 80
        q, p, n = torch.rand(C, T_q, F), torch.rand(C, T_p, F), torch.rand(
            C, T_n, F)
        pos_sims, neg_sims = get_triplet_preds(ft_net, mtr_net, (q, p, n))
        assert pos_sims.size() == (T_q//50, T_p//50)
        assert neg_sims.size() == (T_q // 50, T_n // 50)
