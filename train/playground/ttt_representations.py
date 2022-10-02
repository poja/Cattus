from re import T
import numpy as np

class Ttt:
    
    def __init__(self):
        self.board = [ ' ' ] * 9
     
    @classmethod   
    def from_planes(cls, planes, cpu):
        assert cpu == True
        planes = np.transpose(planes, (2, 0, 1))
        
        t = cls()
        planes = planes.reshape((27,))
        for i, c in enumerate(planes[:9]):
            if c == 1:
                t.board[i] = 'X'
                
        for i, c in enumerate(planes[9:18]):
            if c == 1:
                assert t.board[i] == ' '
                t.board[i] = 'O'
        
        return t    
    
    def to_planes(self, cpu):
        assert cpu == True
        planes = np.array(
            [int(c == 'X') for c in self.board] +
            [int(c == 'O') for c in self.board] +
            [1] * 9
        ).reshape((1,3,3,3))
        return np.transpose(planes, (0, 2, 3, 1))

    
    @classmethod
    def from_str(cls, s):
        t = cls()
        i = 0
        for c in s:
            if c == 'X':
                t.board[i] = 'X'
            elif c == 'O':
                t.board[i] = 'O'
            elif c == ' ' or c == '.':
                t.board[i] = ' '
            else:
                i -= 1
            i += 1
        return t
    
    def __str__(self):
        b = self.board
        return f'{b[0]}{b[1]}{b[2]}\n{b[3]}{b[4]}{b[5]}\n{b[6]}{b[7]}{b[8]}' 
        
                
        
        