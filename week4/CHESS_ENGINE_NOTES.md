# Deep Chess Engine - Complete Guide

## The Big Picture

This project builds a chess engine in **4 layers**, each building on the previous:

1. **Minimax + Alpha-Beta Pruning** (Classical Search)
2. **Zobrist Hashing + Transposition Tables** (Caching)
3. **NNUE Neural Network** (Learned Evaluation)
4. **Reinforcement Learning** (Self-Play Training)

---

## Layer 1: Minimax + Alpha-Beta Pruning

> **File:** `week3/chess_engine.py`

### Minimax Algorithm

Chess is a **two-player zero-sum game** — what's good for you is bad for your opponent.

Minimax explores the game tree by alternating between two players:

- **Maximizer (White):** Picks the move with the **highest** score
- **Minimizer (Black):** Picks the move with the **lowest** score
- At the depth limit, a **static evaluation function** scores the position

```
         Max (White)
        /     |     \
      Min   Min    Min (Black)
     / \   / \    / \
    5   3 6   2  1   8

White picks: 6 (the best guaranteed outcome)
```

### Alpha-Beta Pruning

The key optimization that makes minimax practical.

```
alpha = best score the maximizer can guarantee so far
beta  = best score the minimizer can guarantee so far

Rule: If beta <= alpha → STOP searching this branch
```

**Why does this work?**

If Black already found a move scoring 3, and in another branch White can guarantee 5, Black will **never** go down that branch. So we skip it entirely.

**Performance impact:**

| Without pruning | With pruning |
| --- | --- |
| O(b^d) nodes | O(b^(d/2)) nodes |
| Depth 8 ≈ 16M nodes | Depth 8 ≈ 4K nodes |

This effectively **doubles your search depth** for free.

### Move Ordering

Alpha-beta pruning only works well if you search the **best moves first**. The engine orders moves by priority:

| Priority | Move Type | Why |
| --- | --- | --- |
| 1st | Promotions (+8000) | Creates a queen — huge material gain |
| 2nd | Checks (+5000) | Forces opponent to respond |
| 3rd | Captures (MVV-LVA) | Material gain, likely important |
| 4th | Quiet moves | Everything else |

**MVV-LVA** = Most Valuable Victim, Least Valuable Attacker

> Capturing a Queen with a Pawn scores higher than capturing a Pawn with a Queen.

### Static Evaluation

Simple material counting from White's perspective:

| Piece | Value (centipawns) |
| --- | --- |
| Pawn | 100 |
| Knight | 320 |
| Bishop | 330 |
| Rook | 500 |
| Queen | 900 |
| King | 0 (infinite in practice) |

Special cases:
- Checkmate = `±infinity`
- Stalemate = `0`

---

## Layer 2: Zobrist Hashing + Transposition Table

> **File:** `week4/zobrist_engine.py`

### The Problem

In chess, you can reach the **same position** via different move orders:

```
1. e4 e5 2. Nf3    →  same position
1. Nf3 e5 2. e4    →  same position
```

Without caching, the engine re-evaluates identical positions thousands of times.

### Zobrist Hashing

A clever way to compute a **unique 64-bit hash** for any chess position.

**Setup (one time):**

```python
# Generate random 64-bit numbers for every possible feature
piece_keys[piece_type][color][square]  # 6 × 2 × 64 = 768 random numbers
castling_keys[16]                       # 4 castling rights = 16 combos
ep_keys[9]                              # en passant file (0-7) + none
side_key                                # 1 random number for side to move
```

**Computing the hash:**

```python
hash = 0
for each piece on the board:
    hash XOR= piece_keys[piece][color][square]
hash XOR= castling_keys[current_castling_rights]
hash XOR= ep_keys[en_passant_file]
if black_to_move:
    hash XOR= side_key
```

**The XOR magic — incremental updates in O(1):**

```python
# After moving a piece from e2 to e4:
hash ^= piece_keys[PAWN][WHITE][e2]   # remove from origin
hash ^= piece_keys[PAWN][WHITE][e4]   # place at destination
hash ^= side_key                       # flip turn
# Done! No need to recompute from scratch
```

XOR properties that make this work:
- `A ^ A = 0` (XOR with itself cancels out)
- `A ^ 0 = A` (XOR with zero is identity)
- Order doesn't matter (commutative + associative)

### Transposition Table

A hash map that stores previously evaluated positions:

```
position_hash → { depth, score, flag, best_move }
```

**The three flag types are crucial:**

| Flag | Meaning | When stored |
| --- | --- | --- |
| `EXACT` | This is the true minimax value | Score is between alpha and beta |
| `ALPHA` (upper bound) | Real score is ≤ this value | Search failed low (score ≤ alpha) |
| `BETA` (lower bound) | Real score is ≥ this value | Search failed high (score ≥ beta) |

**How lookup works:**

```python
entry = tt.lookup(position_hash)

if entry.flag == EXACT:
    return entry.score              # we know the exact answer!

if entry.flag == BETA:
    alpha = max(alpha, entry.score) # tighten lower bound

if entry.flag == ALPHA:
    beta = min(beta, entry.score)   # tighten upper bound

if alpha >= beta:
    return entry.score              # cutoff!
```

### Iterative Deepening

Search depth 1, then depth 2, then depth 3… up to max_depth.

**Why not just search at max depth directly?**

| Benefit | Explanation |
| --- | --- |
| TT population | Shallow searches fill the transposition table with best moves |
| Better move ordering | Deeper searches use TT moves first → more pruning |
| Time management | Can stop anytime and return the best move found so far |
| Paradoxical speedup | Searching 1+2+3+4 is often **faster** than just searching 4 |

---

## Layer 3: NNUE Neural Network Evaluation

> **File:** `week4/nnue_eval.py`

### The Problem

Material counting is too simplistic. It doesn't understand:
- Pawn structure (doubled, isolated, passed pawns)
- King safety
- Piece activity and coordination
- Positional concepts (outposts, open files, etc.)

### NNUE = Efficiently Updatable Neural Network

Invented by the Stockfish developers. The key insight: use a neural network for evaluation, but design it so the first layer can be **incrementally updated** when pieces move.

### Feature Vector (Input)

768 binary features = 12 piece types × 64 squares

```
Index mapping:
  WHITE PAWN   on a1 → index 0
  WHITE PAWN   on b1 → index 1
  ...
  WHITE PAWN   on h8 → index 63
  WHITE KNIGHT on a1 → index 64
  ...
  BLACK KING   on h8 → index 767

Formula: index = (piece_type - 1 + 6 * color) * 64 + square
```

For the starting position: 32 features are set to 1 (one per piece), 736 are 0.

### Network Architecture

```
Input (768 binary) 
    ↓
Linear(768, 256) + ClippedReLU    ← 196,864 parameters
    ↓
Linear(256, 32) + ClippedReLU     ← 8,224 parameters
    ↓
Linear(32, 1) + Tanh × 10000     ← 33 parameters
    ↓
Output: single score in [-10000, +10000] centipawns
```

**Total: ~205,000 parameters**

### ClippedReLU

```python
ClippedReLU(x) = clamp(x, 0, 1)
```

Why clipped? Bounds activations to [0, 1], which:
- Stabilizes training
- Allows efficient quantization (important for production engines)
- Keeps inputs to the next layer in a known range

### Why "Efficiently Updatable"?

When a move is made, only **2-4 features change**:

```
Move: White Pawn e2 → e4

Removed features: [WPAWN on e2]  → index 28 flips 1→0
Added features:   [WPAWN on e4]  → index 36 flips 0→1

Only 2 out of 768 inputs changed!
```

For the first layer (768 → 256), instead of recomputing:
```
new_output = W × new_input     ← 768 × 256 = 196,608 multiplications
```

You can do:
```
new_output = old_output 
           - W[:, removed_idx]  ← subtract column for removed feature
           + W[:, added_idx]    ← add column for added feature
```

This is just **256 + 256 = 512 operations** instead of 196,608. That's a **384x speedup** for the most expensive layer.

### Kaiming Initialization

```python
nn.init.kaiming_normal_(weight, nonlinearity='relu')
```

Weights are initialized with variance `2/fan_in` to prevent vanishing/exploding gradients in ReLU networks. Named after Kaiming He (of ResNet fame).

---

## Layer 4: Reinforcement Learning (Self-Play)

> **File:** `week4/rl_training.py`

### The Goal

Train the NNUE weights through **self-play**, so the network learns evaluation from **experience** rather than human-defined rules.

### Self-Play Loop

```
engine_white = NNUE engine
engine_black = same NNUE engine (plays both sides)

while game not over (max 200 moves):
    if random() < epsilon:
        move = random_move()        ← EXPLORE: try something new
    else:
        move = engine.best_move()   ← EXPLOIT: use current knowledge
    
    record(board_features, whose_turn)
    play the move

result = +1 (white wins) / -1 (black wins) / 0 (draw)
```

### Epsilon-Greedy Exploration

| Phase | Epsilon | Behavior |
| --- | --- | --- |
| Early training | 0.30 | 30% random moves → lots of exploration |
| Mid training | 0.15 | Balanced exploration/exploitation |
| Late training | 0.05 | 5% random → mostly using learned policy |

The epsilon decays linearly from `epsilon_start` to `epsilon_end` over all games.

**Why explore?**
Without random moves, the engine only sees positions reachable by its current (bad) policy. It gets stuck in a local optimum. Random moves force it to experience diverse positions.

### TD(0) Learning — Temporal Difference

The core RL update. The idea: **propagate the game result backwards** through all positions in the game.

```
Game moves:    pos1 → pos2 → pos3 → ... → posN → RESULT

TD targets (working backwards):
  target[N]   = result                    (e.g., +1 for white win)
  target[N-1] = gamma × result           (= 0.99)
  target[N-2] = gamma² × result          (= 0.98)
  target[N-3] = gamma³ × result          (= 0.97)
  ...
  target[1]   = gamma^(N-1) × result     (= 0.99^(N-1))
```

### Gamma (Discount Factor) = 0.99

```
gamma = 0.99 means:
  A win in 1 move  is worth  1.00
  A win in 10 moves is worth 0.90
  A win in 50 moves is worth 0.61
  A win in 100 moves is worth 0.37
```

**What this teaches the network:**
- Positions closer to a win should evaluate higher
- Encourages the engine to find **shorter** paths to victory
- Earlier positions still get credit, but discounted

### Training Step (Gradient Descent)

```python
for each (board_features, td_target) in game_experiences:
    prediction = NNUE(board_features)       # what the network thinks
    target = td_target × SCALE              # what actually happened
    loss = MSE(prediction, target)          # squared error
    loss.backward()                          # compute gradients
    optimizer.step()                         # update weights (Adam)
```

Over thousands of games:
- Positions that led to wins → evaluated **higher**
- Positions that led to losses → evaluated **lower**
- The network learns patterns humans didn't explicitly program

### Training Hyperparameters

| Parameter | Default | Purpose |
| --- | --- | --- |
| `num_games` | 100 | Number of self-play games |
| `search_depth` | 2 | Minimax depth during self-play |
| `epsilon_start` | 0.3 | Initial exploration rate |
| `epsilon_end` | 0.05 | Final exploration rate |
| `gamma` | 0.99 | Discount factor |
| `lr` | 0.0001 | Adam learning rate |
| `checkpoint_interval` | 25 | Save model every N games |

---

## Tournament System

> **File:** `week4/tournament.py`

### Engine Types

| Engine | Evaluation | Search | Expected Strength |
| --- | --- | --- | --- |
| `random` | None (random moves) | None | Weakest |
| `minimax` | Material counting | Alpha-Beta, depth 4 | Medium |
| `zobrist` | Material counting | Alpha-Beta + TT + Iterative Deepening | Strong |
| `nnue` | Neural network | Alpha-Beta, depth 3 | Depends on training |

### Scoring

- Win = **1 point**
- Draw = **0.5 points**
- Loss = **0 points**
- Each pair plays **2 games** (one as White, one as Black)
- Max **200 moves** per game (draw if exceeded)

---

## The Complete Pipeline

```
Position
    ↓
Feature Extraction (768-dim binary vector)
    ↓
NNUE Evaluation (neural network → centipawn score)
    ↓
Minimax Search + Alpha-Beta Pruning
    ↓
Zobrist Hash → Transposition Table (lookup/store)
    ↓
Move Ordering (TT move → captures → checks → quiet)
    ↓
Best Move
```

### Training Pipeline

```
Initialize random NNUE weights
    ↓
Self-play games (epsilon-greedy)
    ↓
Collect (position, outcome) experiences
    ↓
Compute TD targets (discounted game results)
    ↓
Train NNUE via gradient descent (MSE loss)
    ↓
Repeat for N games
    ↓
Evaluate in tournament
```

---

## Key Concepts Summary

| Concept | What It Does | Why It Matters |
| --- | --- | --- |
| **Minimax** | Searches game tree assuming optimal play | Foundation of all chess engines |
| **Alpha-Beta** | Prunes branches that can't affect the result | Doubles effective search depth |
| **Move Ordering** | Searches best moves first | Makes alpha-beta pruning effective |
| **Zobrist Hashing** | O(1) position identification via XOR | Enables transposition table |
| **Transposition Table** | Caches evaluated positions (EXACT/ALPHA/BETA) | Avoids redundant computation |
| **Iterative Deepening** | Search depth 1, 2, 3… up to max | Populates TT, enables time control |
| **NNUE** | Neural network for position evaluation | Learns patterns beyond material |
| **Incremental Update** | Only recompute changed features | 384x speedup on first layer |
| **Epsilon-Greedy** | Balance exploration vs exploitation | Prevents getting stuck in local optima |
| **TD Learning** | Propagate game results backwards | Teaches position evaluation from outcomes |
| **Gamma (discount)** | Future rewards worth less than immediate | Encourages shorter paths to victory |

---

## How to Run

```bash
# Install dependencies
pip install python-chess torch

# Train the NNUE with self-play
python3 week4/rl_training.py --games 100 --depth 2

# Run the tournament
python3 week4/tournament.py

# Play against the engine
python3 week4/play.py

# Test Zobrist engine on mate puzzles
python3 week4/zobrist_engine.py
```
