import gymnasium as gym
import json
import random
from typing import Literal
from scipy.signal import convolve2d
from gymnasium import spaces
import matplotlib.pyplot as plt
import numpy as np
import pygame
from pygame import gfxdraw
import numpy as np
from typing import List, Tuple
import math
from math import cos, sin, tan, pi

def blit(base, arr, index=(0,0), add=False, inplace=False):
    """ "blit" one matrix onto another at a certain index """
    if not inplace:
        base = base.copy()
    if add:
        base[index[0]:index[0]+arr.shape[0], index[1]:index[1]+arr.shape[1]] = arr.astype(base.dtype)
    else:
        base[index[0]:index[0]+arr.shape[0], index[1]:index[1]+arr.shape[1]] += arr.astype(base.dtype)
    return base

def crop2content(arr, val=0):
    """ Removes all rows and columns of a 2D matrix which are only the specified value """
    x = arr.T[~np.all(arr.T == val, axis=1)].T
    return x[~np.all(x == val, axis=1)]

def get_pieces(file, unique=False):
    rtn = []
    with open(file, 'r') as f:
        for p in json.load(f):
            for r in range(4):
                rtn.append(blit(np.zeros((5,5)), np.rot90(p, k=r)))
    if unique:
        raise NotImplementedError('Only unique pieces isnt impelemented yet')
    return np.array(rtn)


class BlockdokuEnv(gym.Env):
    metadata = {"render_modes": ['pygame', 'interactive'], "render_fps": 4}
    _all_pieces = get_pieces('pieces.json')

    def __init__(self, render_mode=None, max_steps=None, limit_invalid=10, flatten=True):
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.score = 0
        self.prev_score = 0
        self.steps = 0
        self.max_steps = max_steps
        self.offset = 50
        self.rectSize = 20
        size = (self.rectSize*9) + (self.offset*2)
        self.screen_size = np.array((size, size + 5*(self.rectSize/3) + 100))
        self.screen = None
        self.surf = None
        self.userSurf = None
        self.userSurfOffset = 0
        self._userPrinted = False
        self.font = None
        self._flat = flatten
        self.render_mode = render_mode
        self.invalid_limit = limit_invalid
        self._invalid_count = 0

        ### Define the spaces ###
        if self._flat:
            self.observation_space = spaces.MultiBinary((156,))
        else:
            self.observation_space = spaces.Tuple([spaces.MultiBinary((9,9))] + [spaces.MultiBinary((5,5))]*3)

        self.action_space = spaces.Tuple((spaces.Discrete(3), spaces.Discrete(81)))

        self.board = np.zeros((9,9), np.uint8)
        self.pieces = self.peices = []
        self.refill_pieces()

    def _get_obs(self):
        if self._flat:
            rtn = self.board.flatten().tolist()
            for i in self.pieces:
                rtn += i.flatten().tolist()

            # Fill the rest with 0's
            return np.append(rtn, np.zeros((156 - len(rtn),)))
        else:
            pieces = self.pieces.copy()
            while len(pieces) < 3:
                pieces.append(np.zeros((5,5)))
            return [self.board] + pieces

    def _get_info(self):
        return {}

    def _get_terminated(self):
        if self.max_steps is not None and self.steps > self.max_steps:
            return True

        if self.invalid_limit is not None and self.invalid_limit > 0 and self._invalid_count > self.invalid_limit:
            return True

        return not np.any(self.pieces_fit())

    def _get_reward(self):
        return self.score - self.prev_score

    def reset(self, seed=None, start_valid=True, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        random.seed(seed)
        self.steps = 0

        self.board = np.zeros((9,9), np.uint8)

        self.pieces = self.peices = []
        self.refill_pieces()

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.steps += 1
        self.prev_score = self.score

        # This is a rare instance of when a goto statement would be genuinely useful
        # I'm using a placeholder function to mimic one
        def checkValid():
            pieceIndex, pos = action

            # Check that that peice exists
            if pieceIndex >= len(self.pieces):
                return False
            index = np.unravel_index(pos, (9,9))
            piece = self.pieces[pieceIndex]

            # Make sure the move is valid
            # If it's just an empty placeholder piece, don't do anything
            if not np.any(piece):
                return False
            p = crop2content(piece)
            # If this is not the case, we're trying to put the piece part way off the board, which isn't allowed
            if self.board[index[0]:index[0]+p.shape[0], index[1]:index[1]+p.shape[1]].shape != p.shape:
                return False

            # In case our action is invalid and we need to reset the board
            validBoard = self.board.copy()
            # Add the piece onto the board
            self.board[index[0]:index[0]+p.shape[0], index[1]:index[1]+p.shape[1]] += p.astype(np.uint8)

            # Since we're adding, not replacing, if a peice got added on top of a piece that's already there, it'll add to 2
            if self.board.max() > 1:
                self.board = validBoard
                return False

            self.pieces[pieceIndex] = np.zeros((5,5))
            self.updateBoard()

            # If we actually placed a piece, then add to the score
            self.score += np.sum(p)
            return True

        valid = checkValid()
        if not valid:
            # self.print('Invalid move!')
            self._invalid_count += 1

        return self._get_obs(), self._get_reward() if valid else -2, self._get_terminated(), False, self._get_info()

    def updateBoard(self):
        self.refill_pieces()
        cnt = 0
        # Find & replace the rows
        cnt += self.board[np.all(self.board==1, axis=1)].shape[0]
        self.board[np.all(self.board==1, axis=1)] = 0
        # Find & replace the columns
        cnt += self.board.T[np.all(self.board==1, axis=0)].shape[0]
        self.board.T[np.all(self.board==1, axis=0)] = 0
        # Find & replace the squares
        for row in range(3):
            for col in range(3):
                matches = np.all(self.board[row*3:row*3+3, col*3:col*3+3])
                if matches:
                    self.board[row*3:row*3+3, col*3:col*3+3] = 0
                    cnt += matches

        self.score += cnt * 9

    def _init_pygame(self):
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('Blockdoku')
            self.screen = pygame.display.set_mode(self.screen_size)

        if self.font is None:
            self.font = pygame.font.SysFont("Verdana", 10)

        if self.surf is None:
            self.surf = pygame.Surface(self.screen_size)
            self.surf.convert()
            self.surf.fill((255, 255, 255))

        if self.userSurf is None:
            self.userSurf = pygame.Surface(self.screen_size)
            self.userSurf.convert()
            self.userSurf.fill((255, 255, 255))

    def refill_pieces(self):
        if not np.any(np.array(self.pieces)):
            self.pieces = random.choices(self._all_pieces, k=3, )

    def render(self):
        if self.render_mode == 'pygame':
            self._init_pygame()

            # This doesn't need to be in self, but it is because of the way Python interacts with pygame (I think)
            self.surf.fill((255, 255, 255))

            # Draw the user's print text
            self.surf.blit(self.userSurf, (0, 290))
            self.userSurf.fill((255, 255, 255, 255))
            self.userSurfOffset = 0

            filled = (200,200,200)
            unfilled = (20,20,20)

            # Draw the board
            for row, _row in enumerate(self.board):
                for col, i in enumerate(_row):
                    topLeft = (self.offset + col*self.rectSize, self.offset + row*self.rectSize)
                    pygame.draw.rect(self.surf, filled if i else unfilled, (topLeft, [self.rectSize]*2))

            # Draw the pieces
            gap = 55
            for offset, pice in enumerate(self.pieces):
                for row, _row in enumerate(pice):
                    for col, i in enumerate(_row):
                        topLeft = (
                            self.offset + (gap*offset) + col*(self.rectSize/2),
                            (self.offset) + 10 + (9*self.rectSize) + row*(self.rectSize/2)
                        )
                        pygame.draw.rect(self.surf, filled if i else unfilled, (topLeft, [self.rectSize/2]*2))

            # Draw the helpful texts
            strings = (
                f'Step:           {self.steps}',
                f'Score:          {self.score}',
                f'Running Reward: {self._get_reward()}',
                f'Invalid Moves:  {self._invalid_count}',
            )
            # For some dumb error I don't understand
            try:
                for offset, string in enumerate(strings):
                    self.surf.blit(self.font.render(string, True, (0,0,0)), (5, 5 + offset*10))
            except:
                self.font = pygame.font.SysFont("Verdana", 10)
                for offset, string in enumerate(strings):
                    self.surf.blit(self.font.render(string, True, (0,0,0)), (5, 5 + offset*10))

            # I don't remember what this does
            self.surf = pygame.transform.scale(self.surf, self.screen_size)

            # Display to screen
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            pygame.display.flip()

        elif self.render_mode == 'interactive':
            raise NotImplementedError()

        else:
            raise TypeError(f"Unknown render mode {self.render_mode}")

    def print(self, string):
        self._init_pygame()
        # self.userSurf.blit(self.font.render(str(string), True, (0,0,0)), (5 + ((self.userSurfOffset // 40) * 100), 5 + (self.userSurfOffset % 40)))
        self.userSurf.blit(self.font.render(str(string), True, (0,0,0)), (5, 5 + self.userSurfOffset))
        self.userSurfOffset += 10

    def pieces_fit(self):
        return [self.can_fit(p) for p in self.pieces]

    def can_fit(self, shape):
        # The logical_not here is because we want to fit in the gaps, not what we already have there
        # I have NO idea why flipping across the y axis is required. I'm very baffled. Took me forever to figure out
        return convolve2d(np.logical_not(self.board), np.flip(shape,1)).max() == np.sum(shape)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.font = None
