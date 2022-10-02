from collections import Counter
from itertools import groupby
import json

import numpy as np
import tensorflow as tf
from train import net_utils
from train.data_parser import DataParser
from train.playground.ttt_representations import Ttt
from train.tictactoe import TicTacToe


training_games_dir = '/Users/yishai/work/RL/workarea_nettest/uniq_moves'
cfg = json.load(open('/Users/yishai/work/RL/train/config_nettest.json'))
game = TicTacToe()
parser = DataParser(game, training_games_dir, cfg)
train_dataset = tf.data.Dataset.from_generator(
    parser.generator, output_types=(tf.string, tf.string, tf.string))
train_dataset = train_dataset.map(parser.get_parse_func())
# train_dataset = train_dataset.batch(32, drop_remainder=True)
train_dataset = train_dataset.prefetch(4)


model_path = '/Users/yishai/work/RL/workarea_nettest/models/myfit2'
custom_objects = {
                # "loss_const_0": net_utils.loss_const_0,
                "loss_cross_entropy": net_utils.loss_cross_entropy,
                "policy_head_accuracy": net_utils.policy_head_accuracy}
model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)


# Build overfitter:
hashable_dataset = [(np.array(pl).tobytes(), w) for pl, w in train_dataset]
hashable_dataset.sort()
winners_per_board = dict()
for board_planes, board_data in groupby(hashable_dataset, key=lambda x: x[0]):
    winners_per_board[board_planes] = [x[1] for x in board_data]

def overfitter(pl):
    pl_bytes = np.array(pl).tobytes() 
    assert pl_bytes in winners_per_board
    return float(max(winners_per_board[pl_bytes], key=winners_per_board[pl_bytes].count))


truth, predictions, overfitter_predictions = [], [], []
for (board_planes, winner) in train_dataset:
    truth.append(float(winner))
    predictions.append(float(model(board_planes)[0]))
    overfitter_predictions.append(overfitter(board_planes))
    
    # print(model(planes)[0])
    # print(winner)
    
    # guessed_probs = model(planes)[1]
    # loss = net_utils.loss_cross_entropy(groundtruth_probs, guessed_probs)
    # print(loss)
    # if loss < 0:
    #     import IPython; IPython.embed()
    # if loss == 0:
    #     print(groundtruth_probs)
    #     print(guessed_probs)
    
    

    
    
loss = tf.losses.MeanSquaredError()
print(loss(truth, predictions))
import IPython; IPython.embed()
