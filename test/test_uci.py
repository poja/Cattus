import os
import chess
import chess.engine

DEBUG = False
TESTS_DIR = os.path.dirname(os.path.realpath(__file__))
CATTUS_TOP = os.path.abspath(os.path.join(TESTS_DIR, ".."))
EXE_DIR = os.path.join(CATTUS_TOP, "target", "debug" if DEBUG else "release")
UCI_EXE = os.path.join(EXE_DIR, "cattus.exe")


def test_uci():
    engine = chess.engine.SimpleEngine.popen_uci(
        [UCI_EXE, "--sim-num", "10000"])

    board = chess.Board()
    while not board.is_game_over() and not board.can_claim_draw():
        result = engine.play(board, chess.engine.Limit(time=20))
        board.push(result.move)

        for r in reversed(range(8)):
            for f in range(8):
                sq = chess.square(f, r)
                piece = board.piece_at(sq)
                s = '_' if piece is None else str(piece)
                print(s, ' ', end="")
            print()
        print()
        print()

    engine.quit()

