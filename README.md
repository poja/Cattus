# Cattus

Cattus is a chess engine based on DeepMind [AlphaZero](https://www.deepmind.com/blog/alphazero-shedding-new-light-on-chess-shogi-and-go) written in Rust. Its uses a neural network to evaluate positions, and MCTS as a search algorithm.

The neural network is trained by self-play of the engine itself, in iterations. The initial network may perform badly on a state, but using the search algorithm on top of the network output, a more accurate evaluation of the state is achieved, which is then fed into the network as training data.

## UCI

The engine exectuable support the [Universal Chess Interface](https://en.wikipedia.org/wiki/Universal_Chess_Interface).

Using the engine from Python, using [python-chess](https://pypi.org/project/python-chess/0.15.0/):
```python
import chess
import chess.engine

# Let Cattus play against itself

engine = chess.engine.SimpleEngine.popen_uci(["cattus.exe", "--sim-num", "10000"])
board = chess.Board()

while  not board.is_game_over() and  not board.can_claim_draw():
	result = engine.play(board, chess.engine.Limit(time=20))
	board.push(result.move)
engine.quit()
```
