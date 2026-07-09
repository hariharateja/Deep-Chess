"""
NNUE-style Evaluation Network for Chess.

Architecture:
  Input (768) -> Hidden1 (256, ClippedReLU) -> Hidden2 (32, ClippedReLU) -> Output (1, tanh)

768 features = 12 piece types (6 per color) x 64 squares, one-hot encoding.
Output is scaled to centipawn range (~[-10000, 10000]).
"""

import chess
import math
import os
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from week3.chess_engine import ChessEngine


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

# Piece index mapping: (piece_type - 1) + 6 * color_int
# WHITE PAWN=0, WHITE KNIGHT=1, ..., WHITE KING=5, BLACK PAWN=6, ..., BLACK KING=11
# Feature index = piece_index * 64 + square

def board_to_features(board: chess.Board) -> np.ndarray:
    """Convert a chess board to a 768-dim binary feature vector."""
    features = np.zeros(768, dtype=np.float32)
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            piece_idx = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
            features[piece_idx * 64 + sq] = 1.0
    return features


def compute_feature_diff(board: chess.Board, move: chess.Move):
    """
    Compute which features are added/removed by a move.
    Call BEFORE board.push(move).
    Returns (removed_indices, added_indices) lists.
    """
    removed = []
    added = []

    from_sq = move.from_square
    to_sq = move.to_square
    moving_piece = board.piece_at(from_sq)

    if moving_piece is None:
        return removed, added

    piece_idx = (moving_piece.piece_type - 1) + (0 if moving_piece.color == chess.WHITE else 6)

    # Remove piece from origin
    removed.append(piece_idx * 64 + from_sq)

    # Capture: remove captured piece
    captured = board.piece_at(to_sq)
    if captured:
        cap_idx = (captured.piece_type - 1) + (0 if captured.color == chess.WHITE else 6)
        removed.append(cap_idx * 64 + to_sq)

    # En passant capture
    if board.is_en_passant(move):
        ep_sq = to_sq + (-8 if moving_piece.color == chess.WHITE else 8)
        ep_piece = board.piece_at(ep_sq)
        if ep_piece:
            ep_idx = (ep_piece.piece_type - 1) + (0 if ep_piece.color == chess.WHITE else 6)
            removed.append(ep_idx * 64 + ep_sq)

    # Promotion
    if move.promotion:
        promo_idx = (move.promotion - 1) + (0 if moving_piece.color == chess.WHITE else 6)
        added.append(promo_idx * 64 + to_sq)
    else:
        added.append(piece_idx * 64 + to_sq)

    # Castling: also move the rook
    if board.is_castling(move):
        if chess.square_file(to_sq) == 6:  # Kingside
            rook_from = to_sq + 1
            rook_to = to_sq - 1
        else:  # Queenside
            rook_from = to_sq - 2
            rook_to = to_sq + 1
        rook_idx = (chess.ROOK - 1) + (0 if moving_piece.color == chess.WHITE else 6)
        removed.append(rook_idx * 64 + rook_from)
        added.append(rook_idx * 64 + rook_to)

    return removed, added


# ---------------------------------------------------------------------------
# ClippedReLU
# ---------------------------------------------------------------------------

class ClippedReLU(nn.Module):
    """clamp(relu(x), 0, 1)"""
    def forward(self, x):
        return torch.clamp(x, 0.0, 1.0)


# ---------------------------------------------------------------------------
# NNUE Network
# ---------------------------------------------------------------------------

class NNUENet(nn.Module):
    """
    NNUE-style evaluation network.
    Input: 768 binary features
    Output: single scalar (tanh, scaled to centipawns)
    """

    SCALE = 10000.0  # Output range: [-SCALE, SCALE] centipawns

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(768, 256),
            ClippedReLU(),
            nn.Linear(256, 32),
            ClippedReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (batch, 768) float tensor
        Returns: (batch, 1) evaluation in [-SCALE, SCALE]
        """
        return self.net(x) * self.SCALE


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

class ChessDataset(Dataset):
    """Dataset of (board_fen, score) pairs for supervised training."""

    def __init__(self, data: list):
        """
        data: list of (fen_string, score_float) tuples.
        Scores should be in centipawns from White's perspective.
        """
        self.features = []
        self.scores = []
        for fen, score in data:
            board = chess.Board(fen)
            feat = board_to_features(board)
            self.features.append(feat)
            # Normalize score to [-1, 1] for tanh target
            self.scores.append(np.tanh(score / NNUENet.SCALE))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.features[idx], dtype=torch.float32),
            torch.tensor([self.scores[idx]], dtype=torch.float32),
        )


def train_nnue(model: NNUENet, data: list, epochs: int = 10, lr: float = 1e-3,
               batch_size: int = 64):
    """
    Train the NNUE model on (fen, score) pairs.

    Args:
        model: NNUENet instance
        data: list of (fen_string, centipawn_score) tuples
        epochs: number of training epochs
        lr: learning rate
        batch_size: batch size
    """
    dataset = ChessDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for feat_batch, score_batch in loader:
            optimizer.zero_grad()
            pred = model(feat_batch) / NNUENet.SCALE  # normalize to [-1,1]
            loss = loss_fn(pred, score_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")


# ---------------------------------------------------------------------------
# NNUE Chess Engine
# ---------------------------------------------------------------------------

class NNUEEngine(ChessEngine):
    """Chess engine that uses NNUE network for position evaluation."""

    def __init__(self, max_depth=4, model: NNUENet = None):
        super().__init__(max_depth=max_depth)
        if model is None:
            self.model = NNUENet()
        else:
            self.model = model
        self.model.eval()
        self._feature_cache = None

    def evaluate(self, board: chess.Board) -> float:
        """Evaluate position using NNUE network."""
        if board.is_checkmate():
            if board.turn == chess.WHITE:
                return -math.inf
            else:
                return math.inf
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        features = board_to_features(board)
        feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            score = self.model(feat_tensor).item()
        return score

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
        self.model.eval()
        print(f"Model loaded from {path}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("NNUE Evaluation Network Demo")
    print("=" * 60)

    # Create model
    model = NNUENet()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test feature extraction
    board = chess.Board()
    features = board_to_features(board)
    print(f"Feature vector size: {len(features)}")
    print(f"Active features (starting position): {int(features.sum())}")

    # Test evaluation
    feat_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        score = model(feat_tensor).item()
    print(f"Starting position eval (random weights): {score:.1f} cp")

    # Test incremental feature diff
    move = chess.Move.from_uci("e2e4")
    removed, added = compute_feature_diff(board, move)
    print(f"Move e2e4 -> removed features: {removed}, added features: {added}")

    # Generate some training data using material evaluation
    print("\nGenerating training data from material evaluation...")
    base_engine = ChessEngine(max_depth=1)
    training_data = []
    board = chess.Board()
    for _ in range(200):
        if board.is_game_over():
            board = chess.Board()
        fen = board.fen()
        score = base_engine.evaluate(board)
        if math.isfinite(score):
            training_data.append((fen, score))
        moves = list(board.legal_moves)
        if moves:
            import random
            board.push(random.choice(moves))

    print(f"Training data size: {len(training_data)}")

    # Train
    print("\nTraining NNUE model...")
    train_nnue(model, training_data, epochs=5, lr=1e-3)

    # Test NNUE engine
    print("\nTesting NNUE engine on starting position...")
    engine = NNUEEngine(max_depth=3, model=model)
    board = chess.Board()
    best = engine.find_best_move(board, depth=3)
    print(f"Best move (depth 3): {best}")
    print(f"Nodes searched: {engine.nodes_searched}")
