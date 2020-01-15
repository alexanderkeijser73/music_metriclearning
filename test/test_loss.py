from loss import TripletLoss
import torch

def test_loss_formula():
    eps = 0.001
    vectorized = TripletLoss()
    for i in range(10):
        q_size, p_size, n_size = torch.randint(10, 40, (3, ))
        pos_sims = torch.rand(q_size * p_size).view(q_size, -1)
        neg_sims = torch.rand(q_size * n_size).view(q_size, -1)
        stupid = loss_stupid(pos_sims, neg_sims)
        vect = vectorized(pos_sims, neg_sims)
        assert abs(stupid - vect) < eps

def test_loss_direction():
    pos = torch.FloatTensor(10, 10).fill_(0.4)
    neg = torch.FloatTensor(10, 10).fill_(0.5)
    loss_fn = TripletLoss()
    assert loss_fn(pos, neg) < loss_fn(neg, pos)

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
        pos_x = pos[x]
        neg_x = neg[x]
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