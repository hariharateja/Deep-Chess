"""
Engine-vs-Engine Round-Robin Tournament.

Supports engine types: random, minimax (week3), zobrist (with TT), nnue (neural eval).
Each pair plays twice (once as white, once as black).
Prints results table with rankings.
"""

import chess
import random
import time
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from week3.chess_engine import ChessEngine
from week4.zobrist_engine import ZobristEngine
from week4.nnue_eval import NNUENet, NNUEEngine

MAX_MOVES = 200
MAX_DEPTH_MINIMAX = 4
MAX_DEPTH_ZOBRIST = 4
MAX_DEPTH_NNUE = 3


# ---------------------------------------------------------------------------
# Engine wrappers
# ---------------------------------------------------------------------------

class RandomEngine:
    """Picks a random legal move."""
    name = "Random"

    def find_best_move(self, board: chess.Board, depth=None) -> chess.Move:
        moves = list(board.legal_moves)
        return random.choice(moves) if moves else None


def make_engine(engine_type: str):
    """Create an engine instance by type name."""
    t = engine_type.lower()
    if t == "random":
        return RandomEngine()
    elif t == "minimax":
        e = ChessEngine(max_depth=MAX_DEPTH_MINIMAX)
        e.name = "Minimax"
        return e
    elif t == "zobrist":
        e = ZobristEngine(max_depth=MAX_DEPTH_ZOBRIST)
        e.name = "Zobrist"
        return e
    elif t == "nnue":
        # Try to load a trained model, fall back to random weights
        model = NNUENet()
        ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'checkpoints', 'nnue_final.pt')
        if os.path.exists(ckpt):
            import torch
            model.load_state_dict(torch.load(ckpt, weights_only=True))
            print(f"  NNUE: loaded weights from {ckpt}")
        else:
            print("  NNUE: using random weights (no checkpoint found)")
        e = NNUEEngine(max_depth=MAX_DEPTH_NNUE, model=model)
        e.name = "NNUE"
        return e
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")


# ---------------------------------------------------------------------------
# Play a single game
# ---------------------------------------------------------------------------

def play_game(white_engine, black_engine, max_moves=MAX_MOVES, verbose=False):
    """
    Play a game between two engines.

    Returns:
        result: 1.0 (white wins), 0.0 (black wins), 0.5 (draw)
        move_count: number of moves played
    """
    board = chess.Board()
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        if board.turn == chess.WHITE:
            engine = white_engine
        else:
            engine = black_engine

        move = engine.find_best_move(board)
        if move is None:
            break

        if verbose:
            print(f"  {move_count+1}. {'W' if board.turn == chess.WHITE else 'B'}: {board.san(move)}")

        board.push(move)
        move_count += 1

    # Determine result
    if board.is_checkmate():
        if board.turn == chess.WHITE:
            result = 0.0   # Black wins (white is checkmated)
        else:
            result = 1.0   # White wins (black is checkmated)
    else:
        result = 0.5  # Draw

    return result, move_count


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------

def run_tournament(engine_types=None, verbose=False):
    """Run a round-robin tournament between all engine types."""
    if engine_types is None:
        engine_types = ["random", "minimax", "zobrist", "nnue"]

    print("Creating engines...")
    engines = {}
    for etype in engine_types:
        engines[etype] = make_engine(etype)

    names = list(engines.keys())
    n = len(names)

    # Score tracking
    scores = {name: 0.0 for name in names}
    results_matrix = {a: {b: [] for b in names} for a in names}

    total_games = n * (n - 1)  # Each pair plays twice
    game_num = 0

    print(f"\nRound-Robin Tournament: {n} engines, {total_games} games")
    print("=" * 60)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            white_name = names[i]
            black_name = names[j]
            white_engine = engines[white_name]
            black_engine = engines[black_name]
            game_num += 1

            print(f"\nGame {game_num}/{total_games}: "
                  f"{white_name} (W) vs {black_name} (B)", end="", flush=True)

            start = time.time()
            result, moves = play_game(white_engine, black_engine,
                                       max_moves=MAX_MOVES, verbose=verbose)
            elapsed = time.time() - start

            if result == 1.0:
                result_str = f"{white_name} wins"
                scores[white_name] += 1.0
                scores[black_name] += 0.0
            elif result == 0.0:
                result_str = f"{black_name} wins"
                scores[white_name] += 0.0
                scores[black_name] += 1.0
            else:
                result_str = "Draw"
                scores[white_name] += 0.5
                scores[black_name] += 0.5

            results_matrix[white_name][black_name].append(result)

            print(f" -> {result_str} ({moves} moves, {elapsed:.1f}s)")

    # ---------------------------------------------------------------------------
    # Print results table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)

    # Header
    col_width = 10
    header = f"{'Engine':<{col_width}}"
    for name in names:
        header += f" {name:>{col_width}}"
    header += f" {'Score':>{col_width}}"
    print(header)
    print("-" * len(header))

    # Rows
    for name_a in names:
        row = f"{name_a:<{col_width}}"
        for name_b in names:
            if name_a == name_b:
                row += f" {'---':>{col_width}}"
            else:
                game_results = results_matrix[name_a][name_b]
                # Show W/D/L from name_a's perspective as white
                cell_parts = []
                for r in game_results:
                    if r == 1.0:
                        cell_parts.append("W")
                    elif r == 0.0:
                        cell_parts.append("L")
                    else:
                        cell_parts.append("D")
                cell = ",".join(cell_parts) if cell_parts else "-"
                row += f" {cell:>{col_width}}"
        row += f" {scores[name_a]:>{col_width}.1f}"
        print(row)

    # Rankings
    print("\n" + "-" * 40)
    print("RANKINGS")
    print("-" * 40)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for rank, (name, score) in enumerate(ranked, 1):
        max_score = (n - 1) * 2  # max possible score
        print(f"  {rank}. {name:<12} {score:.1f}/{max_score} points")

    print()
    return scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chess Engine Tournament")
    parser.add_argument("--engines", nargs="+",
                        default=["random", "minimax", "zobrist", "nnue"],
                        help="Engine types to include")
    parser.add_argument("--verbose", action="store_true",
                        help="Print individual moves")
    args = parser.parse_args()

    run_tournament(engine_types=args.engines, verbose=args.verbose)
