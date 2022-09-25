from train.data_parser import DataParser
import tensorflow as tf
from train.tictactoe import TicTacToe

tf.config.run_functions_eagerly(True)

training_games_dir = '/Users/yishai/work/RL/workarea_nettest/uniq_moves'
game = TicTacToe()
parser = DataParser(game, training_games_dir, 10000)
train_dataset = tf.data.Dataset.from_generator(
    parser.generator, output_types=(tf.string, tf.string, tf.string))
train_dataset = train_dataset.map(parser.get_parse_func())
# train_dataset = train_dataset.batch(32, drop_remainder=True)
train_dataset = train_dataset.prefetch(4)
print(next(iter(train_dataset)))

