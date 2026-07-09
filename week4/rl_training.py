"""
Reinforcement Learning Training for Chess NNUE.

Self-play with Q-learning:
- Two engines play against each other
- Epsilon-greedy move selection
- TD(0) updates on the NNUE evaluation network
- Periodic model checkpoints
"""

import chess
import math
import os
import sys
import random
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from week4.nnue_eval import NNUENet, NNUEEngine, board_to_features


# ---------------------------------------------------------------------------
# Self-play game
# ---------------------------------------------------------------------------

MAX_MOVES = 200


def play_self_play_game(engine: NNUEEngine, epsilon: float = 0.1,
                        max_moves: int = MAX_MOVES, search_depth: int = 2):
    """
    Play a self-play game using epsilon-greedy move selection.

    Returns:
        experiences: list of (features, score) tuples for training
        result: 1.0 (white wins), -1.0 (black wins), 0.0 (draw)
    """
    board = chess.Board()
    states = []  # List of (features_array, turn_was_white)

    move_count = 0
    while not board.is_game_over() and move_count < max_moves:
        # Epsilon-greedy: with probability epsilon, pick a random move
        if random.random() < epsilon:
            move = random.choice(list(board.legal_moves))
        else:
            move = engine.find_best_move(board, depth=search_depth)
            if move is None:
                break

        features = board_to_features(board)
        states.append((features, board.turn == chess.WHITE))

        board.push(move)
        move_count += 1

    # Determine game result
    if board.is_checkmate():
        # The side to move is checkmated
        if board.turn == chess.WHITE:
            result = -1.0  # Black wins
        else:
            result = 1.0   # White wins
    else:
        result = 0.0  # Draw (stalemate, repetition, 50-move, or max moves)

    return states, result


# ---------------------------------------------------------------------------
# Q-learning / TD training
# ---------------------------------------------------------------------------

def compute_td_targets(states, result, gamma=0.99):
    """
    Compute TD targets for each state in the game.

    For the final state, the target is the game result.
    For earlier states, we use: target = gamma * next_target
    Targets are from White's perspective.

    Returns list of (features, target) tuples.
    """
    n = len(states)
    if n == 0:
        return []

    targets = [0.0] * n

    # Terminal reward
    targets[-1] = result

    # Backward pass: propagate with discount
    for i in range(n - 2, -1, -1):
        targets[i] = gamma * targets[i + 1]

    training_data = []
    for i in range(n):
        features = states[i][0]  # numpy array
        target = targets[i]
        training_data.append((features, target))

    return training_data


def train_on_experiences(model: NNUENet, experiences: list, lr: float = 1e-4):
    """
    Train the NNUE model on a batch of (features_array, target_score) pairs.
    Uses MSE loss between model output and TD target.
    """
    if not experiences:
        return 0.0

    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    features_batch = torch.tensor(
        np.array([e[0] for e in experiences]), dtype=torch.float32
    )
    # Targets are in [-1, 1] range (game results with discount)
    # Scale to match model output range
    targets_batch = torch.tensor(
        [[e[1] * NNUENet.SCALE] for e in experiences], dtype=torch.float32
    )

    model.train()
    optimizer.zero_grad()
    predictions = model(features_batch)
    loss = loss_fn(predictions, targets_batch)
    loss.backward()
    optimizer.step()
    model.eval()

    return loss.item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(num_games: int = 100, epochs_per_game: int = 1, search_depth: int = 2,
          epsilon_start: float = 0.3, epsilon_end: float = 0.05,
          gamma: float = 0.99, lr: float = 1e-4,
          checkpoint_dir: str = None, checkpoint_interval: int = 25,
          model_path: str = None):
    """
    Run the full RL training pipeline.

    Args:
        num_games: number of self-play games
        epochs_per_game: training passes per game's experiences
        search_depth: minimax search depth during self-play
        epsilon_start: initial exploration rate
        epsilon_end: final exploration rate
        gamma: discount factor
        lr: learning rate
        checkpoint_dir: directory for saving model checkpoints
        checkpoint_interval: save checkpoint every N games
        model_path: path to load a pretrained model
    """
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize model
    model = NNUENet()
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Loaded model from {model_path}")
    model.eval()

    engine = NNUEEngine(max_depth=search_depth, model=model)

    # Stats
    results = {1.0: 0, -1.0: 0, 0.0: 0}
    total_loss = 0.0
    total_moves = 0

    print(f"Starting RL training: {num_games} games, depth={search_depth}")
    print(f"Epsilon: {epsilon_start} -> {epsilon_end}, gamma={gamma}, lr={lr}")
    print("=" * 60)

    start_time = time.time()

    for game_num in range(1, num_games + 1):
        # Linear epsilon decay
        progress = game_num / num_games
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * progress

        # Play a game
        states, result = play_self_play_game(
            engine, epsilon=epsilon, search_depth=search_depth
        )
        results[result] += 1
        total_moves += len(states)

        # Compute TD targets and train
        experiences = compute_td_targets(states, result, gamma=gamma)

        game_loss = 0.0
        for _ in range(epochs_per_game):
            loss = train_on_experiences(model, experiences, lr=lr)
            game_loss += loss
        total_loss += game_loss

        # Progress report
        if game_num % 10 == 0 or game_num == 1:
            elapsed = time.time() - start_time
            avg_loss = total_loss / game_num
            w, b, d = results[1.0], results[-1.0], results[0.0]
            print(f"Game {game_num}/{num_games} | "
                  f"W:{w} B:{b} D:{d} | "
                  f"Moves:{len(states)} | "
                  f"Loss:{game_loss:.4f} | "
                  f"AvgLoss:{avg_loss:.4f} | "
                  f"Eps:{epsilon:.3f} | "
                  f"Time:{elapsed:.1f}s")

        # Checkpoint
        if game_num % checkpoint_interval == 0:
            path = os.path.join(checkpoint_dir, f"nnue_game_{game_num}.pt")
            torch.save(model.state_dict(), path)
            print(f"  -> Checkpoint saved: {path}")

    # Final save
    final_path = os.path.join(checkpoint_dir, "nnue_final.pt")
    torch.save(model.state_dict(), final_path)

    elapsed = time.time() - start_time
    w, b, d = results[1.0], results[-1.0], results[0.0]
    avg_moves = total_moves / num_games

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  Games: {num_games}")
    print(f"  Results: White {w}, Black {b}, Draw {d}")
    print(f"  Avg moves/game: {avg_moves:.1f}")
    print(f"  Final model: {final_path}")
    print(f"  Total time: {elapsed:.1f}s")

    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL Training for Chess NNUE")
    parser.add_argument("--games", type=int, default=100,
                        help="Number of self-play games (default: 100)")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Training epochs per game (default: 1)")
    parser.add_argument("--depth", type=int, default=2,
                        help="Search depth during self-play (default: 2)")
    parser.add_argument("--epsilon-start", type=float, default=0.3,
                        help="Initial exploration rate (default: 0.3)")
    parser.add_argument("--epsilon-end", type=float, default=0.05,
                        help="Final exploration rate (default: 0.05)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor (default: 0.99)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory for checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=25,
                        help="Save checkpoint every N games (default: 25)")
    parser.add_argument("--load-model", type=str, default=None,
                        help="Path to pretrained model to continue training")

    args = parser.parse_args()

    train(
        num_games=args.games,
        epochs_per_game=args.epochs,
        search_depth=args.depth,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        gamma=args.gamma,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        model_path=args.load_model,
    )
