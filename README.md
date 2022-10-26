

# Cattus
[![All tests](https://github.com/poja/RL/actions/workflows/all-tests.yml/badge.svg)](https://github.com/poja/RL/actions/workflows/all-tests.yml)

Cattus is a chess engine based on DeepMind [AlphaZero paper](https://arxiv.org/abs/1712.01815), written in Rust. It uses a neural network to evaluate positions, and MCTS as a search algorithm.

The neural network is trained by self-play of the engine itself, in iterations. The initial network may perform badly on a state, but using the search algorithm on top of the network output, a more accurate evaluation of the state is achieved, which is then fed into the network as training data.

## UCI

The engine exectuable support the [Universal Chess Interface](https://en.wikipedia.org/wiki/Universal_Chess_Interface).

By using [python-chess](https://pypi.org/project/python-chess/0.15.0/) the engine can be used in Python:
```python
import chess
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci(["cattus.exe", "--sim-num", "10000"])

# Let Cattus play against itself
board = chess.Board()
while  not board.is_game_over() and  not board.can_claim_draw():
	result = engine.play(board, chess.engine.Limit(time=20))
	board.push(result.move)

engine.quit()
```

## Training

To run the training process, first one should install the python project on `venv`:
```bash
pip install poetry

mkdir ~/.venvs; cd ~/.venvs
python -m venv cattus

pip install -e $CATTUS_TOP
```

Activate the `venv` and start the training process:
```bash
~/.venvs/cattus/activate

python $CATTUS_TOP/train/main.py --config $CATTUS_TOP/train/config.json
```
