
from pathlib import Path
import shutil


WORKDIR = Path(r'/Users/yishai/work/RL/workarea_nettest3/')

def get_all_moves():
    training_iteration : Path = WORKDIR / 'games' / '221002_171659_00000_4418645149476772613'
    for f in training_iteration.iterdir():
        yield f
        
def main():
    dest_folder = WORKDIR / 'uniq_moves'
    dest_folder.mkdir(exist_ok=True)
    all_seen_moves = set()
    for f in get_all_moves():
        content = open(f, 'rb').read()
        if content in all_seen_moves:
            continue
        all_seen_moves.add(content)
        shutil.copy(f, dest_folder)


if __name__ == '__main__':
    main()