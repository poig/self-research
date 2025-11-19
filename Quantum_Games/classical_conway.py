# -*- coding: utf-8 -*-
"""
The Game of Life (GoL) module named in honour of John Conway

This module defines the classes required for the GoL simulation.

Created on Tue Jan 15 12:21:17 2019

@author: shakes
"""
import numpy as np
from scipy import signal
import rle

class GameOfLife:
    '''
    Object for computing Conway's Game of Life (GoL) cellular machine/automata
    '''
    def __init__(self, N=256, finite=False, fastMode=False):
        self.grid = np.zeros((N,N), np.int64)
        self.neighborhood = np.ones((3,3), np.int64) # 8 connected kernel
        self.neighborhood[1,1] = 0 #do not count centre pixel
        self.finite = finite
        self.fastMode = fastMode
        self.aliveValue = 1
        self.deadValue = 0
        
    def getStates(self):
        '''
        Returns the current states of the cells
        '''
        return self.grid
    
    def getGrid(self):
        '''
        Same as getStates()
        '''
        return self.getStates()
               
    def evolve(self):
        '''
        Given the current states of the cells, apply the GoL rules:
        - Any live cell with fewer than two live neighbors dies, as if by underpopulation.
        - Any live cell with two or three live neighbors lives on to the next generation.
        - Any live cell with more than three live neighbors dies, as if by overpopulation.
        - Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction
        '''
        #get weighted sum of neighbors
        #PART A & E CODE HERE
        if self.fastMode:
            #sum_neighbors = signal.fftconvolve(self.grid, self.neighborhood, mode='same')
            #sum_neighbors = np.round(sum_neighbors,0)
#
            ## only if there are 3 neigbhour or 2 neigbhour and alive is True
            ##new_grid = ((sum_neighbors == 3) | ((sum_neighbors == 2) & (self.grid == 1))).astype(np.int64)  
#
            ## simplfy the boolean algebra expression: n_3 or g and n_2
            #new_grid= (sum_neighbors==3) + self.grid * (sum_neighbors==2)
            #self.grid = new_grid
            #return ()

            # all in one line, but not readable fftconvolve/oaconvolve
            self.grid = (lambda n: ((n == 3) + self.grid * (n == 2)))(np.round(signal.fftconvolve(self.grid, self.neighborhood, mode='same'),0))
            return self.grid
        else:
            sum_neighbors = np.zeros((self.grid.shape[0],self.grid.shape[1]), np.int64)
            for idx, _ in np.ndenumerate(self.grid): 
                # summarize the neighbors with range max(0,n-1),min(grid_size,n+2)
                sum_neighbors[idx] = np.sum(self.grid[max(0,idx[0]-1):min(len(self.grid),idx[0]+2), max(0,idx[1]-1):min(len(self.grid),idx[1]+2)]) - self.grid[idx]

        #implement the GoL rules by thresholding the weights
        #PART A CODE HERE
        Weight_thresehold = ( # [neighbor count, new state]
            (1, -1), # rule 1, dead with 0 or 1 neighbors
            (2, 0), # rule 2, alive with 2 neighbors
            (3, 1), # rule 4, reproduction or keep alive with 3 neighbors
            (np.inf, -1) # rule 3, dead with > 3 neighebors
        )
        update_grid = np.zeros((self.grid.shape[0],self.grid.shape[1]), np.int64)

        # use enumerate specifically design for numpy array, to save double loop enumerate
        for idx, sum_ in np.ndenumerate(sum_neighbors): 
            for weight in Weight_thresehold: 
                # in this structure we can avoid 3 if else statements
                if sum_ <= weight[0]:
                    #update the grid, live if more than 0
                    update_grid[idx] = int(self.grid[idx] + weight[1] > 0)
                    break
        # or we can use use numpy vectorize to save time, doing the same thing
        
        #update the grid
        self.grid = update_grid #UNCOMMENT THIS WITH YOUR UPDATED GRID
    
    def insertBlinker(self, index=(0,0)):
        '''
        Insert a blinker oscillator construct at the index position
        '''
        self.grid[index[0], index[1]+1] = self.aliveValue
        self.grid[index[0]+1, index[1]+1] = self.aliveValue
        self.grid[index[0]+2, index[1]+1] = self.aliveValue
        
    def insertGlider(self, index=(0,0)):
        '''
        Insert a glider construct at the index position
        '''
        self.grid[index[0], index[1]+1] = self.aliveValue
        self.grid[index[0]+1, index[1]+2] = self.aliveValue
        self.grid[index[0]+2, index[1]] = self.aliveValue
        self.grid[index[0]+2, index[1]+1] = self.aliveValue
        self.grid[index[0]+2, index[1]+2] = self.aliveValue
        
    def insertGliderGun(self, index=(0,0)):
        '''
        Insert a glider construct at the index position
        '''
        self.grid[index[0]+1, index[1]+25] = self.aliveValue
        
        self.grid[index[0]+2, index[1]+23] = self.aliveValue
        self.grid[index[0]+2, index[1]+25] = self.aliveValue
        
        self.grid[index[0]+3, index[1]+13] = self.aliveValue
        self.grid[index[0]+3, index[1]+14] = self.aliveValue
        self.grid[index[0]+3, index[1]+21] = self.aliveValue
        self.grid[index[0]+3, index[1]+22] = self.aliveValue
        self.grid[index[0]+3, index[1]+35] = self.aliveValue
        self.grid[index[0]+3, index[1]+36] = self.aliveValue
        
        self.grid[index[0]+4, index[1]+12] = self.aliveValue
        self.grid[index[0]+4, index[1]+16] = self.aliveValue
        self.grid[index[0]+4, index[1]+21] = self.aliveValue
        self.grid[index[0]+4, index[1]+22] = self.aliveValue
        self.grid[index[0]+4, index[1]+35] = self.aliveValue
        self.grid[index[0]+4, index[1]+36] = self.aliveValue
        
        self.grid[index[0]+5, index[1]+1] = self.aliveValue
        self.grid[index[0]+5, index[1]+2] = self.aliveValue
        self.grid[index[0]+5, index[1]+11] = self.aliveValue
        self.grid[index[0]+5, index[1]+17] = self.aliveValue
        self.grid[index[0]+5, index[1]+21] = self.aliveValue
        self.grid[index[0]+5, index[1]+22] = self.aliveValue
        
        self.grid[index[0]+6, index[1]+1] = self.aliveValue
        self.grid[index[0]+6, index[1]+2] = self.aliveValue
        self.grid[index[0]+6, index[1]+11] = self.aliveValue
        self.grid[index[0]+6, index[1]+15] = self.aliveValue
        self.grid[index[0]+6, index[1]+17] = self.aliveValue
        self.grid[index[0]+6, index[1]+18] = self.aliveValue #
        self.grid[index[0]+6, index[1]+23] = self.aliveValue
        self.grid[index[0]+6, index[1]+25] = self.aliveValue
        
        self.grid[index[0]+7, index[1]+11] = self.aliveValue
        self.grid[index[0]+7, index[1]+17] = self.aliveValue
        self.grid[index[0]+7, index[1]+25] = self.aliveValue
        
        self.grid[index[0]+8, index[1]+12] = self.aliveValue
        self.grid[index[0]+8, index[1]+16] = self.aliveValue
        
        self.grid[index[0]+9, index[1]+13] = self.aliveValue
        self.grid[index[0]+9, index[1]+14] = self.aliveValue
        
    def insertFromPlainText(self, txtString, pad=0):
        '''
        Assumes txtString contains the entire pattern as a human readable pattern without comments
        '''
        text= []
        for line in txtString.split('\n'):
            text.append(line)

        description = []
        y = 0
        for idx, f in enumerate(text): 
            #read each line
            if len(f) == 0:
                continue

            if f[0] == "!":
                print(text[idx])
                description.append(text[idx])
            elif "." in f or "O" in f:
                x=0
                for t in text[idx]: 
                    if t == "O":
                        self.grid[y][x] = 1
                    elif t == ".":
                        self.grid[y][x] = 0
                    x += 1
                y += 1
            elif text[idx][0] == " " or text[idx][0] == "\n":
                pass
        print("size:",x,y)
        self.grid = np.pad(self.grid, pad_width=pad)


    def insertFromRLE(self, rleString, pad=0):
        '''
        Given string loaded from RLE file, populate the game grid
        '''
        parser = rle.RunLengthEncodedParser(rle_string= rleString)
        print(parser.name)
        print(parser._comments)
        print("size:", parser.size_x, parser.size_y)
        #rle_info = parser.__format__(None)
        new_grid = np.zeros((parser.size_y,parser.size_x), np.int64)
        alive = 0

        for idx, x in np.ndenumerate(parser.pattern_2d_array):
            if x == "b":
                new_grid[idx] = 0
            elif x == "o":
                new_grid[idx] = 1
                alive+=1
        print(alive, "initial alive cells")

        # default rule: b3/s23
        # we ignore rle rule here as its not a task and which will require to chnage the initial value
        #if parser.rule != "b3/s23":
        #    print("Warning: RLE rule is not b3/s23, ignoring it")

        self.grid = np.pad(new_grid, pad_width=pad)


        