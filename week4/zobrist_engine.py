"""
Zobrist Hashing + Transposition Table + Iterative Deepening Chess Engine.

Enhances the week3 minimax + alpha-beta engine with:
- Zobrist hashing for fast position identification
- Transposition table to avoid re-searching positions
- Iterative deepening to populate the table progressively
"""

import chess
import math
import json
import time
import random
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from week3.chess_engine import ChessEngine


# ---------------------------------------------------------------------------
# Zobrist Hashing
# ---------------------------------------------------------------------------

class ZobristHasher:
    """Generate and incrementally update Zobrist hash keys for chess positions."""

    def __init__(self, seed=42):
        rng = random.Random(seed)

        # piece_keys[piece_type 0..5][color 0..1][square 0..63]
        self.piece_keys = [
            [[rng.getrandbits(64) for _ in range(64)] for _ in range(2)]
            for _ in range(6)
        ]

        # Castling rights: 4 bits -> 16 entries
        self.castling_keys = [rng.getrandbits(64) for _ in range(16)]

        # En-passant file (0-7) + 1 for no ep = 9 entries
        self.ep_keys = [rng.getrandbits(64) for _ in range(9)]

        # Side to move
        self.side_key = rng.getrandbits(64)

    def hash_board(self, board: chess.Board) -> int:
        """Compute full Zobrist hash from scratch."""
        h = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                pt = piece.piece_type - 1  # 0-based
                co = int(piece.color)
                h ^= self.piece_keys[pt][co][sq]

        # Castling
        castling = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            castling |= 1
        if board.has_queenside_castling_rights(chess.WHITE):
            castling |= 2
        if board.has_kingside_castling_rights(chess.BLACK):
            castling |= 4
        if board.has_queenside_castling_rights(chess.BLACK):
            castling |= 8
        h ^= self.castling_keys[castling]

        # En passant
        if board.ep_square is not None:
            h ^= self.ep_keys[chess.square_file(board.ep_square)]
        else:
            h ^= self.ep_keys[8]

        # Side to move
        if board.turn == chess.BLACK:
            h ^= self.side_key

        return h

    def update_hash(self, h: int, board: chess.Board, move: chess.Move) -> int:
        """
        Incrementally update hash after a move is made on `board`.
        Call BEFORE board.push(move).
        Returns the new hash.
        """
        # We'll compute from scratch for correctness on special moves.
        # For a production engine you'd do this incrementally for every case,
        # but the from-scratch path is still fast and correct.
        # However, for normal moves we do it incrementally:

        from_sq = move.from_square
        to_sq = move.to_square
        moving_piece = board.piece_at(from_sq)
        captured_piece = board.piece_at(to_sq)

        if moving_piece is None:
            # Shouldn't happen with legal moves
            board.push(move)
            new_h = self.hash_board(board)
            board.pop()
            return new_h

        # Handle castling, en-passant capture, promotion via full recompute
        is_castling = board.is_castling(move)
        is_ep = board.is_en_passant(move)
        is_promotion = move.promotion is not None

        if is_castling or is_ep or is_promotion:
            board.push(move)
            new_h = self.hash_board(board)
            board.pop()
            return new_h

        # Normal move: incremental update
        pt = moving_piece.piece_type - 1
        co = int(moving_piece.color)

        # Remove piece from origin
        h ^= self.piece_keys[pt][co][from_sq]
        # Place piece at destination
        h ^= self.piece_keys[pt][co][to_sq]

        # Remove captured piece
        if captured_piece:
            cpt = captured_piece.piece_type - 1
            cco = int(captured_piece.color)
            h ^= self.piece_keys[cpt][cco][to_sq]

        # Flip side to move
        h ^= self.side_key

        # Update castling (recompute castling bits after move)
        # Remove old castling hash
        old_castling = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            old_castling |= 1
        if board.has_queenside_castling_rights(chess.WHITE):
            old_castling |= 2
        if board.has_kingside_castling_rights(chess.BLACK):
            old_castling |= 4
        if board.has_queenside_castling_rights(chess.BLACK):
            old_castling |= 8
        h ^= self.castling_keys[old_castling]

        # Remove old EP hash
        if board.ep_square is not None:
            h ^= self.ep_keys[chess.square_file(board.ep_square)]
        else:
            h ^= self.ep_keys[8]

        # Push to get new state, read castling/ep, then pop
        board.push(move)
        new_castling = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            new_castling |= 1
        if board.has_queenside_castling_rights(chess.WHITE):
            new_castling |= 2
        if board.has_kingside_castling_rights(chess.BLACK):
            new_castling |= 4
        if board.has_queenside_castling_rights(chess.BLACK):
            new_castling |= 8
        h ^= self.castling_keys[new_castling]

        if board.ep_square is not None:
            h ^= self.ep_keys[chess.square_file(board.ep_square)]
        else:
            h ^= self.ep_keys[8]
        board.pop()

        return h


# ---------------------------------------------------------------------------
# Transposition Table
# ---------------------------------------------------------------------------

EXACT = 0
ALPHA_FLAG = 1  # Upper bound (failed low)
BETA_FLAG = 2   # Lower bound (failed high)


class TranspositionTable:
    """Hash map storing previously evaluated positions."""

    def __init__(self, max_size=1 << 20):
        self.max_size = max_size
        self.table = {}
        self.hits = 0
        self.stores = 0

    def lookup(self, key: int):
        """Return entry dict or None."""
        entry = self.table.get(key)
        if entry is not None:
            self.hits += 1
        return entry

    def store(self, key: int, depth: int, score: float, flag: int, best_move: chess.Move):
        """Store an entry, replacing if new depth >= old depth."""
        existing = self.table.get(key)
        if existing is not None and existing['depth'] > depth:
            return  # Don't replace deeper entries
        self.table[key] = {
            'depth': depth,
            'score': score,
            'flag': flag,
            'best_move': best_move,
        }
        self.stores += 1
        # Simple eviction: if too large, clear half (crude but functional)
        if len(self.table) > self.max_size:
            keys = list(self.table.keys())
            for k in keys[:len(keys) // 2]:
                del self.table[k]

    def clear(self):
        self.table.clear()
        self.hits = 0
        self.stores = 0


# ---------------------------------------------------------------------------
# Zobrist Engine
# ---------------------------------------------------------------------------

class ZobristEngine(ChessEngine):
    """
    Chess engine with Zobrist hashing, transposition table,
    and iterative deepening on top of minimax + alpha-beta.
    """

    def __init__(self, max_depth=8):
        super().__init__(max_depth=max_depth)
        self.hasher = ZobristHasher()
        self.tt = TranspositionTable()
        self.current_hash = 0

    def order_moves(self, board: chess.Board, tt_best_move=None):
        """Order moves, putting TT best move first."""
        moves = list(board.legal_moves)

        def move_score(move):
            score = 0
            # TT move gets highest priority
            if tt_best_move and move == tt_best_move:
                return 100000
            if board.is_capture(move):
                victim = board.piece_type_at(move.to_square)
                attacker = board.piece_type_at(move.from_square)
                if victim:
                    score += 10 * self.PIECE_VALUES.get(victim, 0)
                if attacker:
                    score -= self.PIECE_VALUES.get(attacker, 0)
            board.push(move)
            if board.is_check():
                score += 5000
            board.pop()
            if move.promotion:
                score += 8000
            return score

        moves.sort(key=move_score, reverse=True)
        return moves

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float,
                maximizing: bool) -> float:
        self.nodes_searched += 1
        orig_alpha = alpha

        # TT lookup
        h = self.hasher.hash_board(board)
        entry = self.tt.lookup(h)
        tt_best_move = None

        if entry is not None and entry['depth'] >= depth:
            if entry['flag'] == EXACT:
                return entry['score']
            elif entry['flag'] == BETA_FLAG:
                alpha = max(alpha, entry['score'])
            elif entry['flag'] == ALPHA_FLAG:
                beta = min(beta, entry['score'])
            if alpha >= beta:
                return entry['score']

        if entry is not None:
            tt_best_move = entry.get('best_move')

        # Terminal or depth limit
        if depth == 0 or board.is_game_over():
            score = self.evaluate(board)
            self.tt.store(h, depth, score, EXACT, None)
            return score

        ordered_moves = self.order_moves(board, tt_best_move)
        best_move = ordered_moves[0] if ordered_moves else None

        if maximizing:
            max_eval = -math.inf
            for move in ordered_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            score = max_eval
        else:
            min_eval = math.inf
            for move in ordered_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            score = min_eval

        # Store in TT
        if score <= orig_alpha:
            flag = ALPHA_FLAG
        elif score >= beta:
            flag = BETA_FLAG
        else:
            flag = EXACT
        self.tt.store(h, depth, score, flag, best_move)

        return score

    def find_best_move(self, board: chess.Board, depth: int = None) -> chess.Move:
        """Find best move using iterative deepening."""
        if depth is None:
            depth = self.max_depth

        self.nodes_searched = 0
        best_move = None

        # Iterative deepening
        for d in range(1, depth + 1):
            maximizing = board.turn == chess.WHITE
            current_best = None
            current_eval = None

            ordered_moves = self.order_moves(board, best_move)

            for move in ordered_moves:
                board.push(move)
                eval_score = self.minimax(board, d - 1, -math.inf, math.inf, not maximizing)
                board.pop()

                if current_eval is None:
                    current_eval = eval_score
                    current_best = move
                elif maximizing and eval_score > current_eval:
                    current_eval = eval_score
                    current_best = move
                elif not maximizing and eval_score < current_eval:
                    current_eval = eval_score
                    current_best = move

            best_move = current_best

        return best_move

    def find_mate_sequence(self, board: chess.Board, depth: int = None) -> list:
        if depth is None:
            depth = self.max_depth

        sequence = []
        board_copy = board.copy()
        remaining_depth = depth

        while remaining_depth > 0 and not board_copy.is_game_over():
            move = self.find_best_move(board_copy, remaining_depth)
            if move is None:
                break
            sequence.append(move)
            board_copy.push(move)
            remaining_depth -= 1

        return sequence


# ---------------------------------------------------------------------------
# Puzzle solving
# ---------------------------------------------------------------------------

def solve_puzzles(json_path: str, mate_in_n: int, max_puzzles: int = None):
    depth = mate_in_n * 2
    with open(json_path, 'r') as f:
        puzzles = json.load(f)

    engine = ZobristEngine(max_depth=depth)
    solved = 0
    failed = 0
    total = 0

    for fen, expected_solution in puzzles.items():
        if max_puzzles and total >= max_puzzles:
            break
        total += 1

        board = chess.Board(fen)
        side = "White" if board.turn == chess.WHITE else "Black"

        print(f"\n{'='*60}")
        print(f"Puzzle #{total}: {fen}")
        print(f"  Side to move: {side}")
        print(f"  Expected: {expected_solution}")

        engine.tt.clear()
        start_time = time.time()
        engine.nodes_searched = 0
        sequence = engine.find_mate_sequence(board, depth)
        elapsed = time.time() - start_time

        if sequence:
            temp_board = board.copy()
            san_moves = []
            for move in sequence:
                san_moves.append(temp_board.san(move))
                temp_board.push(move)

            is_mate = temp_board.is_checkmate()
            move_str = " ".join(san_moves)

            if is_mate:
                print(f"  SOLVED: {move_str}")
                solved += 1
            else:
                print(f"  FOUND SEQUENCE (not mate): {move_str}")
                failed += 1
        else:
            print(f"  FAILED: No sequence found")
            failed += 1

        print(f"  Nodes searched: {engine.nodes_searched}")
        print(f"  TT hits: {engine.tt.hits}")
        print(f"  Time: {elapsed:.3f}s")

    print(f"\n{'='*60}")
    print(f"Results: {solved}/{total} solved, {failed} failed")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    week3_dir = os.path.join(script_dir, '..', 'week3')

    print("Zobrist Engine - Mate-in-2 puzzles demo")
    print("=" * 60)

    json_file = os.path.join(week3_dir, "mate_in_2.json")
    if os.path.exists(json_file):
        solve_puzzles(json_file, 2, max_puzzles=5)
    else:
        print(f"Puzzle file not found: {json_file}")
        print("Running on a sample position instead...")
        board = chess.Board("4r1rk/5K1b/7R/R7/8/8/8/8 w - - 0 1")
        engine = ZobristEngine(max_depth=4)
        seq = engine.find_mate_sequence(board, 4)
        if seq:
            temp = board.copy()
            sans = []
            for m in seq:
                sans.append(temp.san(m))
                temp.push(m)
            print(f"Mate sequence: {' '.join(sans)}")
            print(f"Checkmate: {temp.is_checkmate()}")
