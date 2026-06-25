import chess
import math
import json
import time


class ChessEngine:
    """
    Chess engine using Minimax with Alpha-Beta pruning to solve
    mate-in-2, mate-in-3, and mate-in-4 puzzles.
    """

    def __init__(self, max_depth=8):
        """
        :param max_depth: maximum search depth in plies (half-moves).
                          mate-in-2 = 4 plies, mate-in-3 = 6 plies, mate-in-4 = 8 plies
        """
        self.max_depth = max_depth
        self.nodes_searched = 0

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------

    PIECE_VALUES = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0,
    }

    def evaluate(self, board: chess.Board) -> float:
        """
        Static evaluation from White's perspective.
        Checkmate = +/- infinity, stalemate = 0.
        Otherwise a simple material count.
        """
        if board.is_checkmate():
            # The side to move is checkmated
            if board.turn == chess.WHITE:
                return -math.inf
            else:
                return math.inf

        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        score = 0.0
        for piece_type in self.PIECE_VALUES:
            score += len(board.pieces(piece_type, chess.WHITE)) * self.PIECE_VALUES[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * self.PIECE_VALUES[piece_type]
        return score

    # -------------------------------------------------------------------------
    # Move ordering (captures first, then checks, then rest)
    # -------------------------------------------------------------------------

    def order_moves(self, board: chess.Board):
        """Order moves to improve alpha-beta pruning efficiency."""
        moves = list(board.legal_moves)

        def move_score(move):
            score = 0
            # Captures first
            if board.is_capture(move):
                # MVV-LVA: value of captured piece - value of attacker
                victim = board.piece_type_at(move.to_square)
                attacker = board.piece_type_at(move.from_square)
                if victim:
                    score += 10 * self.PIECE_VALUES.get(victim, 0)
                if attacker:
                    score -= self.PIECE_VALUES.get(attacker, 0)
            # Checks next
            board.push(move)
            if board.is_check():
                score += 5000
            board.pop()
            # Promotions
            if move.promotion:
                score += 8000
            return score

        moves.sort(key=move_score, reverse=True)
        return moves

    # -------------------------------------------------------------------------
    # Minimax with Alpha-Beta Pruning
    # -------------------------------------------------------------------------

    def minimax(self, board: chess.Board, depth: int, alpha: float, beta: float,
                maximizing: bool) -> float:
        """
        Minimax with alpha-beta pruning.
        :param board: current board state
        :param depth: remaining depth to search
        :param alpha: best value maximizer can guarantee
        :param beta: best value minimizer can guarantee
        :param maximizing: True if it's the maximizing player's turn
        :return: evaluation score
        """
        self.nodes_searched += 1

        # Terminal or depth limit
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)

        ordered_moves = self.order_moves(board)

        if maximizing:
            max_eval = -math.inf
            for move in ordered_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # beta cutoff
            return max_eval
        else:
            min_eval = math.inf
            for move in ordered_moves:
                board.push(move)
                eval_score = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # alpha cutoff
            return min_eval

    # -------------------------------------------------------------------------
    # Find best move
    # -------------------------------------------------------------------------

    def find_best_move(self, board: chess.Board, depth: int = None) -> chess.Move:
        """
        Find the best move for the current side to move.
        :param board: current board position
        :param depth: search depth in plies (defaults to self.max_depth)
        :return: best Move
        """
        if depth is None:
            depth = self.max_depth

        self.nodes_searched = 0
        maximizing = board.turn == chess.WHITE

        best_move = None
        best_eval = None

        ordered_moves = self.order_moves(board)

        for move in ordered_moves:
            board.push(move)
            eval_score = self.minimax(board, depth - 1, -math.inf, math.inf, not maximizing)
            board.pop()

            if best_eval is None:
                best_eval = eval_score
                best_move = move
            elif maximizing and eval_score > best_eval:
                best_eval = eval_score
                best_move = move
            elif not maximizing and eval_score < best_eval:
                best_eval = eval_score
                best_move = move

        return best_move

    # -------------------------------------------------------------------------
    # Find forced mate sequence
    # -------------------------------------------------------------------------

    def find_mate_sequence(self, board: chess.Board, depth: int = None) -> list:
        """
        Find the full forced mate sequence (list of moves).
        Uses minimax to pick the best move for each side at every step.
        :param board: current board position
        :param depth: search depth in plies
        :return: list of Move objects forming the mate sequence
        """
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


# =============================================================================
# Solve puzzles from JSON files
# =============================================================================

def solve_puzzles(json_path: str, mate_in_n: int, max_puzzles: int = None):
    """
    Load puzzles from a JSON file and solve them.
    :param json_path: path to the puzzle JSON file
    :param mate_in_n: number of moves to mate (2, 3, or 4)
    :param max_puzzles: max number of puzzles to solve (None = all)
    """
    depth = mate_in_n * 2  # convert moves to plies

    with open(json_path, 'r') as f:
        puzzles = json.load(f)

    engine = ChessEngine(max_depth=depth)

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
        print(f"  Board:\n{board}\n")

        start_time = time.time()
        engine.nodes_searched = 0
        sequence = engine.find_mate_sequence(board, depth)
        elapsed = time.time() - start_time

        if sequence:
            # Format the move sequence in SAN
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
        print(f"  Time: {elapsed:.3f}s")

    print(f"\n{'='*60}")
    print(f"Results: {solved}/{total} solved, {failed} failed")


# =============================================================================
# Interactive mode
# =============================================================================

def interactive_mode():
    """
    Enter a FEN and get the engine's best move / mate sequence.
    """
    engine = ChessEngine(max_depth=8)

    print("Chess Engine - Mate in N Solver")
    print("Enter a FEN position, or 'quit' to exit.\n")

    while True:
        fen = input("FEN> ").strip()
        if fen.lower() in ('quit', 'exit', 'q'):
            break

        try:
            board = chess.Board(fen)
        except ValueError as e:
            print(f"Invalid FEN: {e}")
            continue

        depth_input = input("Search depth in plies (default=8): ").strip()
        depth = int(depth_input) if depth_input else 8

        print(f"\nBoard:\n{board}\n")
        print(f"Searching at depth {depth}...")

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
                print(f"Forced mate found: {move_str}")
            else:
                print(f"Best sequence: {move_str}")
        else:
            print("No move found.")

        print(f"Nodes searched: {engine.nodes_searched}")
        print(f"Time: {elapsed:.3f}s\n")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "interactive":
            interactive_mode()

        elif mode in ("2", "3", "4"):
            n = int(mode)
            json_file = os.path.join(script_dir, f"mate_in_{n}.json")
            max_puzzles = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            solve_puzzles(json_file, n, max_puzzles)

        elif mode == "all":
            for n in (2, 3, 4):
                json_file = os.path.join(script_dir, f"mate_in_{n}.json")
                if os.path.exists(json_file):
                    print(f"\n{'#'*60}")
                    print(f"  MATE IN {n} PUZZLES")
                    print(f"{'#'*60}")
                    solve_puzzles(json_file, n, max_puzzles=3)

        else:
            # Treat as a FEN string
            board = chess.Board(mode)
            depth = int(sys.argv[2]) if len(sys.argv) > 2 else 8
            engine = ChessEngine(max_depth=depth)
            sequence = engine.find_mate_sequence(board, depth)
            if sequence:
                temp = board.copy()
                sans = [temp.san(m) for m in sequence if not temp.push(m)]
                # re-do properly
                temp = board.copy()
                sans = []
                for m in sequence:
                    sans.append(temp.san(m))
                    temp.push(m)
                print(" ".join(sans))
                print(f"Checkmate: {temp.is_checkmate()}")
    else:
        print("Usage:")
        print("  python chess_engine.py 2          # solve mate-in-2 puzzles (first 5)")
        print("  python chess_engine.py 3 10       # solve mate-in-3 puzzles (first 10)")
        print("  python chess_engine.py 4           # solve mate-in-4 puzzles (first 5)")
        print("  python chess_engine.py all         # solve sample from each category")
        print("  python chess_engine.py interactive  # interactive mode")
        print('  python chess_engine.py "<FEN>" 8   # solve a specific position')
        print()
        # Default: run a quick demo
        print("Running demo on mate-in-2 puzzles (first 3)...\n")
        json_file = os.path.join(script_dir, "mate_in_2.json")
        if os.path.exists(json_file):
            solve_puzzles(json_file, 2, max_puzzles=3)
