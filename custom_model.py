# custom_model.py

import numpy as np
from copy import deepcopy
from piece import Piece
from board import Board
import os

class CUSTOM_AI_MODEL:
    def __init__(self, weights_path='trained_model.npy'):
        """
        Initializes the AI model by loading trained weights.
        If weights are not found, initializes with random weights.
        """
        if os.path.exists(weights_path):
            self.genotype = np.load(weights_path)
            print(f"Loaded trained weights from {weights_path}")
        else:
            # Initialize with random weights if no trained model is found
            self.genotype = np.random.uniform(-1, 1, 9)
            print("Initialized with random weights.")
    
    def save_weights(self, weights_path='trained_model.npy'):
        """
        Saves the current weights to a file.
        """
        np.save(weights_path, self.genotype)
        print(f"Saved trained weights to {weights_path}")
    
    def load_weights(self, weights_path='trained_model.npy'):
        """
        Loads weights from a file.
        """
        if os.path.exists(weights_path):
            self.genotype = np.load(weights_path)
            print(f"Loaded trained weights from {weights_path}")
        else:
            raise FileNotFoundError(f"No weights file found at {weights_path}")
    
    def valuate(self, board):
        """
        Evaluates the board using the current weights.
        """
        peaks = self.get_peaks(board)
        highest_peak = np.max(peaks)
        holes = self.get_holes(peaks, board)
        wells = self.get_wells(peaks)
    
        # Define feature functions
        features = {
            'agg_height': np.sum(peaks),
            'n_holes': np.sum(holes),
            'bumpiness': self.get_bumpiness(peaks),
            'num_pits': np.count_nonzero(np.count_nonzero(board, axis=0) == 0),
            'max_wells': np.max(wells),
            'n_cols_with_holes': np.count_nonzero(np.array(holes) > 0),
            'row_transitions': self.get_row_transition(board, highest_peak),
            'col_transitions': self.get_col_transition(board, peaks),
            'cleared': np.count_nonzero(np.mean(board, axis=1))
        }
    
        # Order the features consistently with the genotype
        feature_values = np.array([
            features['agg_height'],
            features['n_holes'],
            features['bumpiness'],
            features['num_pits'],
            features['max_wells'],
            features['n_cols_with_holes'],
            features['row_transitions'],
            features['col_transitions'],
            features['cleared']
        ])
    
        # Calculate the aggregate score
        aggregate_rating = np.dot(self.genotype, feature_values)
    
        return aggregate_rating
    
    def get_best_move(self, board, piece):
    """
    Determines the best move by evaluating all possible placements of the piece
    using the current AI model's scoring function.
    """
    best_x = -1
    best_piece = None
    max_score = -np.inf  # Start with the lowest possible score

    for rotation in range(4):  # Explore all 4 rotations
        rotated_piece = piece.get_next_rotation()
        for x in range(board.width):  # Explore all horizontal positions
            try:
                y = board.drop_height(rotated_piece, x)  # Get drop height for this placement
            except Exception:
                continue  # Skip invalid placements

            # Simulate placing the piece
            board_copy = deepcopy(board)
            try:
                board_copy.place(x, y, rotated_piece)
            except Exception:
                continue  # Skip if the piece can't be placed

            # Convert the board to a numpy array for evaluation
            np_board = self.bool_to_np(board_copy.board)

            # Evaluate the board using the AI model
            score = self.valuate(np_board)

            # Keep track of the best move
            if score > max_score:
                max_score = score
                best_x = x
                best_piece = deepcopy(rotated_piece)  # Save the best piece configuration

    return best_x, best_piece

    
    @staticmethod
    def bool_to_np(board):
        """
        Converts a boolean board to a numpy array of integers.
        """
        return np.array([[1 if cell else 0 for cell in row] for row in board])
    
    @staticmethod
    def get_peaks(board):
        """
        Identifies the peak (highest filled cell) in each column.
        """
        peaks = np.zeros(board.shape[1], dtype=int)
        for col in range(board.shape[1]):
            filled = np.where(board[:, col] == 1)[0]
            if filled.size > 0:
                peaks[col] = filled.max() + 1  # +1 to represent height
            else:
                peaks[col] = 0
        return peaks
    
    @staticmethod
    def get_holes(peaks, board):
        """
        Counts the number of holes (empty cells with at least one filled cell above them) in each column.
        """
        holes = np.zeros(board.shape[1], dtype=int)
        for col in range(board.shape[1]):
            if peaks[col] > 0:
                column = board[:peaks[col], col]
                holes[col] = np.sum(column == 0)
        return holes
    
    @staticmethod
    def get_bumpiness(peaks):
        """
        Calculates the bumpiness of the board based on the peaks.
        """
        bumpiness = 0
        for i in range(len(peaks) - 1):
            bumpiness += abs(peaks[i] - peaks[i + 1])
        return bumpiness
    
    @staticmethod
    def get_wells(peaks):
        """
        Calculates the number of wells in the board.
        """
        wells = np.zeros(len(peaks), dtype=int)
        for i in range(len(peaks)):
            if i == 0:
                if peaks[i] < peaks[i + 1]:
                    wells[i] = peaks[i + 1] - peaks[i]
            elif i == len(peaks) - 1:
                if peaks[i] < peaks[i - 1]:
                    wells[i] = peaks[i - 1] - peaks[i]
            else:
                left = peaks[i - 1] - peaks[i]
                right = peaks[i + 1] - peaks[i]
                if left > 0 and right > 0:
                    wells[i] = min(left, right)
        return wells
    
    @staticmethod
    def get_row_transition(board, highest_peak):
        """
        Counts the number of row transitions in the board.
        """
        transitions = 0
        for row in range(int(board.shape[0] - highest_peak), board.shape[0]):
            for col in range(1, board.shape[1]):
                if board[row, col] != board[row, col - 1]:
                    transitions += 1
        return transitions
    
    @staticmethod
    def get_col_transition(board, peaks):
        """
        Counts the number of column transitions in the board.
        """
        transitions = 0
        for col in range(board.shape[1]):
            if peaks[col] <= 1:
                continue
            for row in range(int(board.shape[0] - peaks[col]), board.shape[0] - 1):
                if board[row, col] != board[row + 1, col]:
                    transitions += 1
        return transitions
