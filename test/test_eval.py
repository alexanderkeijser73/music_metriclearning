# import pytest
# import torch
# from eval import check_song_level_constraint
# from models.metricnet import MetricNet
# from models.featurenet import FeatureNet
#
# def test_check_song_level_constraint():
#     mtr_net = MetricNet()
#     ftr_net = FeatureNet()
#     mel_tensor = torch.randn(18, 3, 50, 80)
#     slc = check_song_level_constraint(ftr_net, mtr_net, mel_tensor, mel_tensor, mel_tensor)
#     assert type(slc) == bool


