"""
Created on Sat Jun 24 17:38:57 2017

@author: Arjun
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time

def create_board():
    return np.zeros([3,3])

def place(board,player,position):
    if(board[position]==0):
        board[position] = player

def possibilities(board):
    return np.where(board==0)

def random_place(board,player):
    position = random.choice(possibilities(board))
    board[position] = player
    
def row_win(board,player):
    for i in range(3):
        if np.all(board[i]==player):
            return True
    return False

def col_win(board,player):
    for i in range(3):
        if np.all(board[:,i]==player):
            return True
    return False

def diag_win(board,player):
    xdiag = np.array([board[i,j] for i in range(3) for j in range(3) if i+j==2])
    if np.all(board.diagonal()==player):
        return True
    elif np.all(xdiag==player):
        return True
    else:
        return False

def evaluate(board):
    winner = 0
    for player in [1, 2]:
        if row_win(board,player) and col_win(board,player) and diag_win(board,player):
            winner = player
        
    if np.all(board != 0) and winner == 0:
        winner = -1
    return winner  

def play_game():
    board = create_board()
    player = 1
    for i in range(9):
        random_place(board,player)
        end = evaluate(board)
        if evaluate != -1:
            break
        if player == 1:
            player = 2
        else:
            player = 1

def play_strategic_game():
    board, winner = create_board(), 0
    board[1,1] = 1
    while winner == 0:
        for player in [2,1]:
            #what the shit
            # use `random_place` to play a game, and store as `board`.
            board = random_place(board,player)
            # use `evaluate(board)`, and store as `winner`.
            winner = evaluate(board)
            if winner != 0:
                break
    return winner


board = create_board()
player = 1
position = (0,0)
place(board,player,position)
possibilities(board)
player = 2
random_place(board,player)
evaluate(board)  

results=[]
start = time.time()
for i in range(1000):
    results.append(play_game())
end = time.time()
print(end-start)
plt.hist(results)
plt.show()

# starts from center
resultsS=[]
for i in range(1000):
    results.append(play_strategic_game())
plt.hist(results)
plt.show()
