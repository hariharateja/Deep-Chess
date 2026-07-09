"""
Interactive Play-Against-the-Engine Mode.

Choose your color, pick an engine, and play chess in the terminal.
Moves can be entered in SAN (e.g., Nf3) or UCI (e.g., g1f3) format.
"""

import chess
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from week3.chess_engine import ChessEngine
from week4.zobrist_engine import ZobristEngine
from week4.nnue_eval import NNUENet, NNUEEngine


def create_engine(engine_type: str):
    """Create an engine by type."""
    t = engine_type.lower()
    if t == "minimax":
        return ChessEngine(max_depth=4)
    elif t == "zobrist":
        return ZobristEngine(max_depth=4)
    elif t == "nnue":
        model = NNUENet()
        ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'checkpoints', 'nnue_final.pt')
        if os.path.exists(ckpt):
            import torch
            model.load_state_dict(torch.load(ckpt, weights_only=True))
            print(f"NNUE: loaded weights from {ckpt}")
        else:
            print("NNUE: using random weights (no checkpoint found)")
        return NNUEEngine(max_depth=3, model=model)
    else:
        raise ValueError(f"Unknown engine type: {t}")


def parse_move(board: chess.Board, move_str: str) -> chess.Move:
    """Try to parse a move as SAN first, then as UCI."""
    move_str = move_str.strip()

    # Try SAN
    try:
        return board.parse_san(move_str)
    except (chess.InvalidMoveError, chess.IllegalMoveError, ValueError):
        pass

    # Try UCI
    try:
        move = chess.Move.from_uci(move_str)
        if move in board.legal_moves:
            return move
    except (ValueError, chess.InvalidMoveError):
        pass

    return None


def display_board(board: chess.Board, player_color: chess.Color):
    """Display the board oriented for the player."""
    print()
    if player_color == chess.BLACK:
        # Flip board for black
        print(board.transform(chess.flip_vertical).transform(chess.flip_horizontal))
    else:
        print(board)
    print()


def play():
    """Main interactive play loop."""
    print("=" * 50)
    print("  CHESS - Play Against the Engine")
    print("=" * 50)

    # Choose color
    while True:
        color_input = input("\nPlay as (w)hite or (b)lack? ").strip().lower()
        if color_input in ('w', 'white'):
            player_color = chess.WHITE
            print("You are playing as White.")
            break
        elif color_input in ('b', 'black'):
            player_color = chess.BLACK
            print("You are playing as Black.")
            break
        else:
            print("Please enter 'w' or 'b'.")

    # Choose engine
    while True:
        engine_input = input("\nEngine type (minimax/zobrist/nnue)? ").strip().lower()
        if engine_input in ('minimax', 'zobrist', 'nnue'):
            break
        else:
            print("Please enter 'minimax', 'zobrist', or 'nnue'.")

    print(f"\nCreating {engine_input} engine...")
    engine = create_engine(engine_input)

    board = chess.Board()
    print("\nGame started! Enter moves in SAN (e.g., Nf3) or UCI (e.g., g1f3).")
    print("Type 'quit' to resign, 'undo' to take back a move.\n")

    display_board(board, player_color)

    while not board.is_game_over():
        if board.turn == player_color:
            # Player's turn
            side_str = "White" if board.turn == chess.WHITE else "Black"
            move_input = input(f"Your move ({side_str}): ").strip()

            if move_input.lower() in ('quit', 'resign', 'q'):
                print("You resigned. Engine wins!")
                return

            if move_input.lower() == 'undo':
                if len(board.move_stack) >= 2:
                    board.pop()  # Undo engine's move
                    board.pop()  # Undo player's move
                    print("Took back last move pair.")
                    display_board(board, player_color)
                else:
                    print("Nothing to undo.")
                continue

            move = parse_move(board, move_input)
            if move is None:
                print(f"Invalid move: '{move_input}'. Legal moves: ", end="")
                legal = [board.san(m) for m in board.legal_moves]
                if len(legal) <= 20:
                    print(", ".join(legal))
                else:
                    print(", ".join(legal[:20]) + f"... ({len(legal)} total)")
                continue

            san = board.san(move)
            board.push(move)
            print(f"You played: {san}")
        else:
            # Engine's turn
            print("Engine thinking...", end="", flush=True)
            start = time.time()
            move = engine.find_best_move(board)
            elapsed = time.time() - start

            if move is None:
                print(" No move found!")
                break

            san = board.san(move)
            board.push(move)
            print(f" {san} ({elapsed:.1f}s)")

        display_board(board, player_color)

    # Game over
    print("=" * 50)
    result = board.result()
    if board.is_checkmate():
        winner = "Black" if board.turn == chess.WHITE else "White"
        if (winner == "White" and player_color == chess.WHITE) or \
           (winner == "Black" and player_color == chess.BLACK):
            print(f"Checkmate! You win! ({result})")
        else:
            print(f"Checkmate! Engine wins! ({result})")
    elif board.is_stalemate():
        print(f"Stalemate! Draw. ({result})")
    elif board.is_insufficient_material():
        print(f"Insufficient material. Draw. ({result})")
    elif board.is_fifty_moves():
        print(f"Fifty-move rule. Draw. ({result})")
    elif board.is_repetition():
        print(f"Threefold repetition. Draw. ({result})")
    else:
        print(f"Game over: {result}")
    print("=" * 50)


if __name__ == "__main__":
    play()
