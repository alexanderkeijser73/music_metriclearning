from torch import nn
import torch

class TripletLoss(nn.Module):

    def __init__(self):
        super(TripletLoss, self).__init__()

    def forward(self, pos_patch_sims, neg_patch_sims):
        """ Vectorized implementation of loss proposed in "Deep ranking:
        Triplet MatchNet for music metric learning"
        https://ieeexplore.ieee.org/document/7952130

        :param pos_patch_sims: similarities [0,1] for all combinations (
                itertools product) of frames patches from the query and
                positive example
        :param neg_patch_sims: similarities [0,1] for all combinations (
                itertools product) of frames patches from the query and
                negative example
        :return: loss (float)
        """
        d_plus_max = pos_patch_sims.max(dim=-1, keepdims=True)[0]
        psi = torch.clamp(d_plus_max - neg_patch_sims, min=0).mean()
        log_pos = torch.log(1 - pos_patch_sims)
        log_neg = torch.log(neg_patch_sims)
        phi = -(log_pos.mean() + log_neg.mean())
        loss = phi  + psi
        return loss
