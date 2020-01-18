import sys
sys.path.append('..')
from train import main as train
from train_utils import parse_args_config
from eval import eval_trained_model

if __name__ == '__main__':
    config = parse_args_config()
    if config.eval:
        eval_trained_model(config,
                           config.ftr_net_checkpoint,
                           config.mtr_net_checkpoint)
    else:
        train(config)
