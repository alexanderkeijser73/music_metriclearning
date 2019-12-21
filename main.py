import sys
sys.path.append('..')
from music_metriclearning.train import main as train

if __name__ == '__main__':
    train(parse_args=True)