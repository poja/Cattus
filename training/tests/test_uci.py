import subprocess
from pathlib import Path

import chess
import chess.engine

TESTS_DIR = Path(__file__).parent.resolve()
CATTUS_ENGINE_TOP = TESTS_DIR.parent.parent / "engine"


def test_works_with_python_library_chess():
    subprocess.check_call(
        ["cargo", "build", "--features=onnx-ort", "--bin=cattus", "-q", "--profile=release"],
        cwd=CATTUS_ENGINE_TOP,
    )
    cattus_exe = CATTUS_ENGINE_TOP / "target" / "release" / "cattus"
    assert cattus_exe.exists()

    engine = chess.engine.SimpleEngine.popen_uci(
        [str(cattus_exe), "--sim-num=100"],
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
