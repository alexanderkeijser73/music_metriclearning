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

# def loss(pos_patch_sims, neg_patch_sims, query_size=None):
#     pos = pos_patch_sims.view(query_size, -1)
#     neg = neg_patch_sims.view(query_size, -1)
#     d_plus_max = pos.max(dim=-1, keepdims=True)[0]
#     psi = torch.clamp(d_plus_max - neg, min=0).mean()
#     log_pos = torch.log(1 - pos_patch_sims)
#     log_neg = torch.log(neg_patch_sims)
#     phi = -(log_pos.mean() + log_neg.mean())
#     return phi + psi

def loss_stupid(pos, neg):
    """ Inefficient but certainly correct implementation of loss proposed in
    "Deep ranking: Triplet MatchNet for music metric learning"
    https://ieeexplore.ieee.org/document/7952130

    :param pos_patch_sims: similarities [0,1] for all combinations (
            itertools product) of frames patches from the query and
            positive example
    :param neg_patch_sims: similarities [0,1] for all combinations (
            itertools product) of frames patches from the query and
            negative example
    :return: loss (float)
    """
    loss = []
    for x in range(len(pos)):
        pos_x = pos[x].squeeze()
        neg_x = neg[x].squeeze()
        #calculate phi
        d_plus_max = pos_x.max()
        psi = 0
        for xmin in neg_x:
            psi += torch.clamp(d_plus_max - xmin, min=0)
        psi /= neg_x.size(-1)
        # calculate psi
        phi = 0
        for xplus in pos_x:
            for xmin in neg_x:
                phi -= (torch.log(1-xplus) + torch.log(xmin))
        phi /= (pos_x.size(-1) * neg_x.size(-1))
        loss.append((phi + psi).item())
    return sum(loss) / len(loss)

# if __name__ == '__main__':
#     pos = torch.nn.functional.sigmoid(torch.randn(1, 324))
#     neg = torch.nn.functional.sigmoid(torch.randn(1, 360))
#     print(loss(pos, neg, query_size=18))
#     print(loss_stupid(pos, neg))
