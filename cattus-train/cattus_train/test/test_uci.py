import os
from pathlib import Path

import chess
import chess.engine

TESTS_DIR = Path(os.path.realpath(__file__)).parent
CATTUS_ENGINE_TOP = Path(TESTS_DIR).parent.parent.parent / "cattus-engine"


def test_works_with_python_library_chess():
    engine = chess.engine.SimpleEngine.popen_uci(
        "cargo run --bin cattus -- --sim-num 100".split(" "),
        cwd=CATTUS_ENGINE_TOP,
    )

    board = chess.Board()
    while not board.is_game_over() and not board.can_claim_draw():
        result = engine.play(board, chess.engine.Limit(time=20))
        board.push(result.move)

        for r in reversed(range(8)):
            for f in range(8):
                sq = chess.square(f, r)
                piece = board.piece_at(sq)
                s = "_" if piece is None else str(piece)
                print(s, " ", end="")
            print()
        print()
        print()

    engine.quit()


if __name__ == "__main__":
    test_works_with_python_library_chess()
