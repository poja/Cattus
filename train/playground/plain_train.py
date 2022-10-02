
import copy
import json
import os
from pathlib import Path
import shutil
from train.hex import Hex
from train.tictactoe import TicTacToe

from train.train_process import TrainProcess


WORKDIR = Path(r'/Users/yishai/work/RL/workarea_nettest/')
TRAIN_DATA_DIR = WORKDIR / 'uniq_moves'
CFG = Path('/Users/yishai/work/RL/train/config_nettest.json')
# BASE_MODEL_PATH = WORKDIR / 'models' / 'model_220924_154327'
OUTPUT = WORKDIR / 'models' / 'myfit2'


class PlainTrain(TrainProcess):
    
    def __init__(self, cfg):
        self.cfg = copy.deepcopy(cfg)
        
        self.game = TicTacToe()
        self.net_type = self.cfg["model"]["type"]
        self.cfg['models_dir'] = WORKDIR / 'models'

        
    def train(self):
        base_model = self.game.create_model(self.net_type, self.cfg)
        base_model_path = WORKDIR / 'models' / 'base'
        base_model.save(base_model_path, save_format='tf')

        output_path, _ = self._train(base_model_path, TRAIN_DATA_DIR)
        shutil.rmtree(OUTPUT)
        shutil.copytree(output_path, OUTPUT)

        

def main():
    with open(CFG, 'rb') as f:
        cfg = json.load(f)
    plain_train = PlainTrain(cfg)
    plain_train.train()


if __name__ == '__main__':
    main()
    