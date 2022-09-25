
from pathlib import Path
import shutil


WORKDIR = Path(r'/Users/yishai/work/RL/workarea_nettest/')

def get_all_moves():
    training_iteration : Path = WORKDIR / 'games' / '220924_154327_00000_11915216593680640006'
    for f in training_iteration.iterdir():
        yield f
        
def main():
    dest_folder = WORKDIR / 'uniq_moves'
    all_seen_moves = set()
    for f in get_all_moves():
        content = open(f, 'rb').read()
        if content in all_seen_moves:
            continue
        all_seen_moves.add(content)
        shutil.copy(f, dest_folder)


if __name__ == '__main__':
    main()