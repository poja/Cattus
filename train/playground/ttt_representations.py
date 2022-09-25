from re import T
import numpy as np

class Ttt:
    
    def __init__(self):
        self.board = [ ' ' ] * 9
     
    @classmethod   
    def from_planes(cls, planes):
        t = cls()
        planes = planes.reshape((27,))
        for i in planes[:9]:
            if planes[i] == 1:
                t.board = 'X'
                
        for i in planes[9:18]:
            if planes[9+i] == 1:
                t.board = 'O'
        
        return t    
    
    def to_planes(self):
        return np.array(
            [int(c == 'X') for c in self.board] +
            [int(c == 'O') for c in self.board] +
            [1] * 9
        ).reshape((1,3,3,3))

    
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
    
    @classmethod
    def __str__(self):
        b = self.board
        return f'{b[0]}{b[1]}{b[2]}\n{b[3]}{b[4]}{b[5]}\n{b[6]}{b[7]}{b[8]}' 
        
                
        
        