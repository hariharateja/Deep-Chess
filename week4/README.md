# Week 4: Advanced Chess Engine

Building on the Week 3 minimax engine with Zobrist hashing, NNUE evaluation, reinforcement learning, and interactive play.

## Files

| File | Description |
|------|-------------|
| `zobrist_engine.py` | Minimax + alpha-beta with Zobrist hashing, transposition tables, and iterative deepening |
| `nnue_eval.py` | NNUE-style neural network evaluation (768 -> 256 -> 32 -> 1) using PyTorch |
| `rl_training.py` | Self-play reinforcement learning with Q-learning / TD(0) updates |
| `tournament.py` | Round-robin engine-vs-engine tournament with results table |
| `play.py` | Interactive play-against-the-engine mode in the terminal |

## Installation

```bash
pip install python-chess torch
```

## Usage

### Zobrist Engine (puzzle solving)
```bash
python zobrist_engine.py
```

### NNUE Evaluation (demo)
```bash
python nnue_eval.py
```

### RL Training
```bash
# Quick training run
python rl_training.py --games 100 --epochs 1 --depth 2

# Longer training
python rl_training.py --games 1000 --depth 3 --lr 0.0001

# Resume from checkpoint
python rl_training.py --games 500 --load-model checkpoints/nnue_final.pt
```

### Tournament
```bash
# Full tournament (all engines)
python tournament.py

# Select specific engines
python tournament.py --engines random minimax zobrist

# Verbose output (show every move)
python tournament.py --verbose
```

### Interactive Play
```bash
python play.py
```
Then choose your color and engine type. Enter moves in SAN (e.g., `Nf3`) or UCI (e.g., `g1f3`).

## Algorithms

- **Zobrist Hashing**: Random 64-bit keys XOR'd together for piece/square/castling/en-passant/turn. Enables fast position lookup in a transposition table.
- **Transposition Table**: Caches evaluated positions with depth, score, bound type (exact/alpha/beta), and best move. Avoids re-searching the same position.
- **Iterative Deepening**: Search depth 1, then 2, ..., up to max. Populates the TT so deeper searches benefit from shallower results.
- **NNUE**: Efficiently Updatable Neural Network for evaluation. 768 binary input features (12 piece types x 64 squares), two hidden layers with ClippedReLU, tanh output scaled to centipawns.
- **RL / TD Learning**: Self-play games with epsilon-greedy exploration. Game outcomes propagated backward with discount factor gamma. NNUE weights updated via MSE loss on TD targets.
