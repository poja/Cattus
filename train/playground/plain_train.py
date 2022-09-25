
import copy
import json
from pathlib import Path
from train.hex import Hex
from train.tictactoe import TicTacToe

from train.train_process import TrainProcess


WORKDIR = Path(r'/Users/yishai/work/RL/workarea_nettest/')
TRAIN_DATA_DIR = WORKDIR / 'uniq_moves'
CFG = Path('/Users/yishai/work/RL/train/config_nettest.json')
BASE_MODEL_PATH = WORKDIR / 'models' / 'model_220924_154327'
OUTPUT = WORKDIR / 'models' / 'myfit'


class PlainTrain(TrainProcess):
    
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        
        self.game = TicTacToe()
        self.net_type = self.cfg["model"]["type"]

        
    def train(self):
        output_model = self._train(BASE_MODEL_PATH, TRAIN_DATA_DIR)
        output_model.save(OUTPUT, save_format='tf')

        

def main():
    with open(CFG, 'rb') as f:
        cfg = json.load(f)
    plain_train = PlainTrain(cfg)
    plain_train.train()


if __name__ == '__main__':
    main()
    