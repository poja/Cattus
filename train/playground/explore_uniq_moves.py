from collections import Counter
import json
from pathlib import Path


UNIQ_MOVES_DIR = Path('/Users/yishai/work/RL/workarea_nettest2/uniq_moves')

def load_uniq_moves():
    """Load json files from uniq_moves directory."""
    moves = []
    for f in UNIQ_MOVES_DIR.iterdir():
        with open(f, 'r') as f:
            moves.append(json.load(f))
    return moves


def main():
    moves = load_uniq_moves()
    
    pairs = [(tuple(m['planes']), m['winner']) for m in moves]
    print(f'len(pairs): {len(pairs)}')
    pairs = set(pairs)
    print(f'len(pairs): {len(pairs)}')
    
    c = Counter([x[0] for x in pairs])
    c2 = Counter(c.values())
    print(c2)
    



if __name__ == '__main__':
    main()
    